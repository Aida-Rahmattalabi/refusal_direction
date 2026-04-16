import json
import os
import torch

from typing import List
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm
from transformers import GenerationConfig

from pipeline.utils.hook_utils import add_hooks
from pipeline.model_utils.model_base import ModelBase


def generate_harmful_responses(
    model,
    tokenizer,
    tokenize_instructions_fn,
    instructions: List[str],
    batch_size: int = 8,
    max_new_tokens: int = 64,
    forced_prefix: str = "Sure, here is",
) -> List[str]:
    """
    Generate compliant responses for harmful instructions by forcing a non-refusal prefix.

    The forced_prefix ("Sure, here is") is prepended to the assistant turn before
    generation, steering the model away from its trained refusal behavior. The returned
    responses include the forced prefix followed by the model's continuation.
    """
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    responses = []

    for i in tqdm(range(0, len(instructions), batch_size), desc="Generating harmful responses"):
        batch_instr = instructions[i : i + batch_size]

        # Tokenize with the forced prefix already in the assistant turn
        tokenized = tokenize_instructions_fn(
            instructions=batch_instr,
            outputs=[forced_prefix] * len(batch_instr),
        )

        input_ids = tokenized.input_ids.to(model.device)
        attention_mask = tokenized.attention_mask.to(model.device)
        prefix_len = input_ids.shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
            )

        # Decode only the newly generated tokens; prepend the forced prefix to
        # reconstruct the full response text.
        for j in range(len(batch_instr)):
            continuation = tokenizer.decode(
                output_ids[j, prefix_len:], skip_special_tokens=True
            ).strip()
            responses.append(forced_prefix + " " + continuation)

    return responses


def generate_refusal_responses(
    model,
    tokenizer,
    tokenize_instructions_fn,
    instructions: List[str],
    batch_size: int = 8,
    max_new_tokens: int = 64,
) -> List[str]:
    """
    Generate natural refusal responses by running the model normally on harmful
    instructions. Since the prompts are harmful, the safety-trained model will
    refuse them, producing the "harmless" side of the contrast pair.
    """
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    responses = []

    for i in tqdm(range(0, len(instructions), batch_size), desc="Generating refusal responses"):
        batch_instr = instructions[i : i + batch_size]
        tokenized = tokenize_instructions_fn(instructions=batch_instr)

        input_ids = tokenized.input_ids.to(model.device)
        attention_mask = tokenized.attention_mask.to(model.device)
        prefix_len = input_ids.shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
            )

        for j in range(len(batch_instr)):
            response = tokenizer.decode(
                output_ids[j, prefix_len:], skip_special_tokens=True
            ).strip()
            responses.append(response)

    return responses


def get_mean_activations_with_responses(
    model,
    tokenizer,
    tokenize_instructions_fn,
    instructions: List[str],
    responses: List[str],
    block_modules: List[torch.nn.Module],
    batch_size: int = 8,
    n_eoi_positions: int = 5,
) -> Float[Tensor, "n_positions n_layers d_model"]:
    """
    Compute mean residual-stream activations at the end-of-instruction (EOI) positions
    when teacher-forcing the model on instruction+response sequences.

    Activations are recorded at the last n_eoi_positions tokens of the instruction
    (just before the response begins). This mirrors the positions used in the original
    generate_directions pipeline but now conditions on the model seeing the full response.

    With left-padding, for sample j in a batch of max_seq_len:
        eoi_end   = max_seq_len - resp_lengths[j]
        eoi_start = eoi_end - n_eoi_positions
    where resp_lengths[j] = len(response tokens for sample j).
    """
    torch.cuda.empty_cache()

    n_layers = model.config.num_hidden_layers
    n_samples = len(instructions)
    d_model = model.config.hidden_size

    mean_activations = torch.zeros(
        (n_eoi_positions, n_layers, d_model),
        dtype=torch.float64,
        device=model.device,
    )

    for i in tqdm(range(0, n_samples, batch_size), desc="Computing activations"):
        batch_instr = instructions[i : i + batch_size]
        batch_resp = responses[i : i + batch_size]

        # Tokenize instruction only to measure how many real tokens each instruction has
        tok_instr = tokenize_instructions_fn(instructions=batch_instr)
        instr_lengths = tok_instr.attention_mask.sum(dim=1)  # [bs]

        # Tokenize instruction+response for the teacher-forced forward pass
        tok_full = tokenize_instructions_fn(instructions=batch_instr, outputs=batch_resp)
        full_lengths = tok_full.attention_mask.sum(dim=1)  # [bs]

        # Number of response tokens per sample
        resp_lengths = full_lengths - instr_lengths  # [bs]
        max_seq_len = tok_full.input_ids.shape[1]

        # Collect EOI activations per layer for this batch
        layer_acts: dict[int, list] = {l: [] for l in range(n_layers)}

        def make_hook(layer_idx):
            def hook_fn(module, input):
                # input[0]: [bs, seq_len, d_model]
                act = input[0].clone().to(dtype=torch.float64)
                for j in range(act.shape[0]):
                    resp_len = int(resp_lengths[j].item())
                    eoi_end = max_seq_len - resp_len
                    eoi_start = eoi_end - n_eoi_positions
                    layer_acts[layer_idx].append(act[j, eoi_start:eoi_end, :])
            return hook_fn

        fwd_pre_hooks = [
            (block_modules[layer], make_hook(layer)) for layer in range(n_layers)
        ]

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            model(
                input_ids=tok_full.input_ids.to(model.device),
                attention_mask=tok_full.attention_mask.to(model.device),
            )

        # Accumulate into running mean (divide by total n_samples, not batch size)
        for layer_idx in range(n_layers):
            for act in layer_acts[layer_idx]:  # act: [n_eoi_positions, d_model]
                mean_activations[:, layer_idx, :] += act.to(mean_activations.device) / n_samples

    return mean_activations


def get_mean_diff_causal(
    model,
    tokenizer,
    tokenize_instructions_fn,
    instructions: List[str],
    harmful_responses: List[str],
    refusal_responses: List[str],
    block_modules: List[torch.nn.Module],
    batch_size: int = 8,
    n_eoi_positions: int = 5,
) -> Float[Tensor, "n_positions n_layers d_model"]:
    """
    Causal mean-diff:
        mean_activations(instruction + harmful_response)
      - mean_activations(instruction + refusal_response)

    Both activation sets are computed at EOI positions of the same harmful instructions,
    so the only thing that varies between the two runs is the response the model sees.
    This isolates the activation signature of compliance vs. refusal from content
    differences in the prompts.
    """
    mean_harmful = get_mean_activations_with_responses(
        model, tokenizer, tokenize_instructions_fn,
        instructions, harmful_responses, block_modules,
        batch_size=batch_size, n_eoi_positions=n_eoi_positions,
    )
    mean_refusal = get_mean_activations_with_responses(
        model, tokenizer, tokenize_instructions_fn,
        instructions, refusal_responses, block_modules,
        batch_size=batch_size, n_eoi_positions=n_eoi_positions,
    )

    mean_diff: Float[Tensor, "n_positions n_layers d_model"] = mean_harmful - mean_refusal
    return mean_diff


def generate_directions_causal(
    model_base: ModelBase,
    harmful_instructions: List[str],
    artifact_dir: str,
    batch_size: int = 8,
    max_new_tokens: int = 64,
    forced_prefix: str = "Sure, here is",
):
    """
    Causal variant of generate_directions.

    Instead of contrasting activations from harmful prompts against activations from a
    separate harmless prompt set, this method uses only the harmful prompt set and
    contrasts activations conditioned on two different continuations of the *same* prompt:

      - Compliant (harmful) response: generated by forcing the model to begin with
        `forced_prefix` (e.g. "Sure, here is"), which bypasses refusal training and
        produces a cooperative continuation.
      - Refusal response: generated by running the model normally; since the prompts
        are harmful, the model naturally refuses.

    Activations are recorded at EOI positions (last len(eoi_toks) tokens of the
    instruction) using teacher forcing on the full instruction+response sequence.

    This gives a "causal" direction because the prompt is held constant — only the
    response changes — so the mean diff captures the internal state difference between
    the model being in "compliance mode" vs. "refusal mode".
    """
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    n_eoi_positions = len(model_base.eoi_toks)

    # Step 1: generate both response types for each harmful instruction
    harmful_responses = generate_harmful_responses(
        model_base.model, model_base.tokenizer, model_base.tokenize_instructions_fn,
        harmful_instructions,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        forced_prefix=forced_prefix,
    )
    refusal_responses = generate_refusal_responses(
        model_base.model, model_base.tokenizer, model_base.tokenize_instructions_fn,
        harmful_instructions,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )

    # Save generated responses for inspection / reuse
    with open(os.path.join(artifact_dir, "harmful_responses.json"), "w") as f:
        json.dump(
            [{"instruction": inst, "response": resp}
             for inst, resp in zip(harmful_instructions, harmful_responses)],
            f, indent=4,
        )
    with open(os.path.join(artifact_dir, "refusal_responses.json"), "w") as f:
        json.dump(
            [{"instruction": inst, "response": resp}
             for inst, resp in zip(harmful_instructions, refusal_responses)],
            f, indent=4,
        )

    # Step 2: compute the causal mean diff
    mean_diffs = get_mean_diff_causal(
        model_base.model, model_base.tokenizer, model_base.tokenize_instructions_fn,
        harmful_instructions, harmful_responses, refusal_responses,
        model_base.model_block_modules,
        batch_size=batch_size,
        n_eoi_positions=n_eoi_positions,
    )

    assert mean_diffs.shape == (
        n_eoi_positions,
        model_base.model.config.num_hidden_layers,
        model_base.model.config.hidden_size,
    )
    assert not mean_diffs.isnan().any()

    torch.save(mean_diffs, os.path.join(artifact_dir, "mean_diffs.pt"))

    return mean_diffs
