"""Beam-search helpers for PRIMERA-MultiNews (LED seq2seq generation)."""

from __future__ import annotations

import torch

from core.config import LENGTH_PENALTY, MAX_DECODE_STEPS, MIN_DECODE_STEPS, NO_REPEAT_NGRAM


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _resolve_generation_kwargs(model, beam_size: int) -> dict:
    generation_config = model.generation_config
    model_config = model.config
    task_params = getattr(model_config, "task_specific_params", {}) or {}
    summarization_params = task_params.get("summarization", {}) or {}

    decoder_start_token_id = _first_not_none(
        generation_config.decoder_start_token_id,
        model_config.decoder_start_token_id,
    )

    return {
        "num_beams": beam_size,
        "num_return_sequences": beam_size,
        "max_length": _first_not_none(
            generation_config.max_length,
            summarization_params.get("max_length"),
            MAX_DECODE_STEPS,
        ),
        "min_length": _first_not_none(
            generation_config.min_length,
            summarization_params.get("min_length"),
            MIN_DECODE_STEPS,
        ),
        "length_penalty": _first_not_none(
            generation_config.length_penalty,
            summarization_params.get("length_penalty"),
            LENGTH_PENALTY,
        ),
        "no_repeat_ngram_size": _first_not_none(
            generation_config.no_repeat_ngram_size,
            summarization_params.get("no_repeat_ngram_size"),
            NO_REPEAT_NGRAM,
        ),
        "early_stopping": _first_not_none(
            generation_config.early_stopping,
            summarization_params.get("early_stopping"),
            True,
        ),
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


def batch_beam_search_decode(model, input_ids, attention_mask, beam_size, **extra_generate_kwargs):
    """Return beam candidates grouped by sample for a batched input tensor."""
    generation_kwargs = _resolve_generation_kwargs(model, beam_size)
    generation_kwargs.update(extra_generate_kwargs)
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
