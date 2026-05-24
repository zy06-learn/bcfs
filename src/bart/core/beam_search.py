"""Beam-search helpers aligned with Hugging Face BART generation.

The project-specific difference from the standalone HF baseline is the beam
width. All other decode parameters should come from the checkpoint's official
`generation_config`.
"""

from __future__ import annotations

import torch

from core.config import LENGTH_PENALTY, MAX_DECODE_STEPS, MIN_DECODE_STEPS, NO_REPEAT_NGRAM


def _resolve_generation_kwargs(model, beam_size: int) -> dict:
    generation_config = model.generation_config
    decoder_start_token_id = generation_config.decoder_start_token_id
    if decoder_start_token_id is None:
        decoder_start_token_id = model.config.decoder_start_token_id

    return {
        "num_beams": beam_size,
        "num_return_sequences": beam_size,
        "max_length": generation_config.max_length or MAX_DECODE_STEPS,
        "min_length": generation_config.min_length or MIN_DECODE_STEPS,
        "length_penalty": generation_config.length_penalty or LENGTH_PENALTY,
        "no_repeat_ngram_size": generation_config.no_repeat_ngram_size or NO_REPEAT_NGRAM,
        "early_stopping": True if generation_config.early_stopping is None else generation_config.early_stopping,
        "use_cache": True,
        "return_dict_in_generate": True,
        "output_scores": True,
        "decoder_start_token_id": decoder_start_token_id,
        "forced_bos_token_id": generation_config.forced_bos_token_id,
        "forced_eos_token_id": generation_config.forced_eos_token_id,
        "bos_token_id": generation_config.bos_token_id,
        "eos_token_id": generation_config.eos_token_id,
        "pad_token_id": generation_config.pad_token_id,
    }


def batch_beam_search_decode(model, input_ids, attention_mask, beam_size):
    """Return beam candidates grouped by sample for a batched input tensor."""
    generation_kwargs = _resolve_generation_kwargs(model, beam_size)
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs,
        )

    sequence_scores = generated.sequences_scores
    if sequence_scores is None:
        sequence_scores = torch.zeros(generated.sequences.size(0), device=generated.sequences.device)

    flat_candidates = [
        (float(score), token_ids)
        for score, token_ids in zip(sequence_scores.tolist(), generated.sequences.tolist())
    ]

    batch_size = int(input_ids.size(0))
    if batch_size <= 1:
        return [flat_candidates]

    grouped_candidates = []
    for start in range(0, len(flat_candidates), beam_size):
        grouped_candidates.append(flat_candidates[start:start + beam_size])
    return grouped_candidates
