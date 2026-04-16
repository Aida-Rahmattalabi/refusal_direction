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


def get_activations_at_eoi(
    model,
    tokenize_instructions_fn,
    instructions: List[str],
    responses: List[str],
    block_modules: List[torch.nn.Module],
    n_eoi_positions: int,
    batch_size: int = 8,
) -> Float[Tensor, "n_samples n_positions n_layers d_model"]:
    """
    Teacher-force the model on instruction+response sequences and record activations
    at the EOI positions (last n_eoi_positions tokens of the instruction) for every
    sample individually.

    Returns shape: (n_samples, n_eoi_positions, n_layers, d_model)

    With left-padding the EOI slice for sample j in a batch of max_seq_len is:
        eoi_end   = max_seq_len - resp_lengths[j]
        eoi_start = eoi_end - n_eoi_positions
    """
    torch.cuda.empty_cache()

    n_samples = len(instructions)
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size

    # Pre-allocate per-sample activation storage
    all_activations = torch.zeros(
        (n_samples, n_eoi_positions, n_layers, d_model),
        dtype=torch.float64,
    )

    for i in tqdm(range(0, n_samples, batch_size), desc="Computing activations"):
        batch_instr = instructions[i : i + batch_size]
        batch_resp  = responses[i : i + batch_size]
        bs = len(batch_instr)

        # Instruction-only lengths (for locating EOI positions in the full sequence)
        tok_instr = tokenize_instructions_fn(instructions=batch_instr)
        instr_lengths = tok_instr.attention_mask.sum(dim=1)  # [bs]

        # Full instruction+response lengths
        tok_full = tokenize_instructions_fn(instructions=batch_instr, outputs=batch_resp)
        full_lengths = tok_full.attention_mask.sum(dim=1)  # [bs]

        resp_lengths = full_lengths - instr_lengths  # [bs]
        max_seq_len  = tok_full.input_ids.shape[1]

        # layer -> list of per-sample tensors [n_eoi_positions, d_model]
        layer_acts: dict[int, list] = {l: [] for l in range(n_layers)}

        def make_hook(layer_idx):
            def hook_fn(module, input):
                act = input[0].clone().to(dtype=torch.float64)  # [bs, seq_len, d_model]
                for j in range(act.shape[0]):
                    resp_len  = int(resp_lengths[j].item())
                    eoi_end   = max_seq_len - resp_len
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

        # Store into all_activations[sample_idx, :, layer, :]
        for layer_idx in range(n_layers):
            for j, act in enumerate(layer_acts[layer_idx]):  # act: [n_eoi_positions, d_model]
                all_activations[i + j, :, layer_idx, :] = act.cpu()

    return all_activations


def get_mean_diff_causal(
    model,
    tokenize_instructions_fn,
    instructions: List[str],
    harmful_responses: List[str],
    refusal_responses: List[str],
    block_modules: List[torch.nn.Module],
    n_eoi_positions: int,
    batch_size: int = 8,
) -> Float[Tensor, "n_positions n_layers d_model"]:
    """
    Causal mean-diff computed as the average over all harmful prompts of the
    per-prompt activation difference:

        mean_over_prompts [ act(instruction + harmful_response)
                          - act(instruction + refusal_response) ]

    Both activation sets are at EOI positions, so the only variable between
    the two runs is the response text — isolating the compliance-vs-refusal
    direction from prompt-content variation.
    """
    acts_harmful = get_activations_at_eoi(
        model, tokenize_instructions_fn,
        instructions, harmful_responses, block_modules,
        n_eoi_positions=n_eoi_positions, batch_size=batch_size,
    )  # (n_samples, n_positions, n_layers, d_model)

    acts_refusal = get_activations_at_eoi(
        model, tokenize_instructions_fn,
        instructions, refusal_responses, block_modules,
        n_eoi_positions=n_eoi_positions, batch_size=batch_size,
    )  # (n_samples, n_positions, n_layers, d_model)

    # Per-prompt difference, then average across prompts
    per_prompt_diff = acts_harmful - acts_refusal  # (n_samples, n_positions, n_layers, d_model)
    mean_diff = per_prompt_diff.mean(dim=0)        # (n_positions, n_layers, d_model)

    return mean_diff


def generate_directions_causal(
    model_base: ModelBase,
    harmful_instructions: List[str],
    harmful_responses: List[str],
    artifact_dir: str,
    batch_size: int = 8,
    max_new_tokens: int = 64,
):
    """
    Causal variant of generate_directions.

    Instead of contrasting activations from harmful prompts against a separate
    harmless prompt set, this method uses only the harmful prompt set and contrasts
    activations conditioned on two different continuations of the *same* prompt:

      - Compliant (harmful) response: provided externally via `harmful_responses`.
        These should come from an uncensored model or a curated example dataset
        (see dataset/example_harmful_responses.json for test fixtures).
      - Refusal response: generated by running the model normally; since the prompts
        are harmful, the model naturally refuses.

    The direction is computed as the average over all prompts of the per-prompt
    activation difference at EOI positions (teacher-forced on the full
    instruction+response sequence). This is more causally clean than the original
    mean-diff because prompt content is held constant across the two conditions.
    """
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    assert len(harmful_instructions) == len(harmful_responses), (
        f"Got {len(harmful_instructions)} instructions but {len(harmful_responses)} responses"
    )

    n_eoi_positions = len(model_base.eoi_toks)

    # Generate refusal responses from the model normally
    refusal_responses = generate_refusal_responses(
        model_base.model, model_base.tokenizer, model_base.tokenize_instructions_fn,
        harmful_instructions,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )

    # Save both response sets for inspection / reuse
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

    # Compute causal mean diff
    mean_diffs = get_mean_diff_causal(
        model_base.model, model_base.tokenize_instructions_fn,
        harmful_instructions, harmful_responses, refusal_responses,
        model_base.model_block_modules,
        n_eoi_positions=n_eoi_positions,
        batch_size=batch_size,
    )

    assert mean_diffs.shape == (
        n_eoi_positions,
        model_base.model.config.num_hidden_layers,
        model_base.model.config.hidden_size,
    )
    assert not mean_diffs.isnan().any()

    torch.save(mean_diffs, os.path.join(artifact_dir, "mean_diffs.pt"))

    return mean_diffs
