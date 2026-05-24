"""HF generation backend for the generate-then-optimize pipeline."""

from __future__ import annotations

import re

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
)


_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
CNN_DAILYMAIL_PROMPT_VERSION = "cnn_dailymail_concise_summary_only_v1"
CNN_DAILYMAIL_PROMPT_TEMPLATE = """Summarize the following news article in English in 3-5 concise sentences.

Output only the summary text.
Do not include a title, label, bullet points, markdown, explanation, preface, or closing note.

Article:
{article}

Summary:
"""
MULTI_NEWS_PROMPT_VERSION = "multi_news_summary_only_v4"
MULTI_NEWS_PROMPT_TEMPLATE = """Summarize the following collection of news articles in English in 8-12 sentences.

Synthesize the shared key facts across the documents and avoid repeating the same information.
Output only the summary text.
Do not include a title, label, bullet points, markdown, explanation, preface, or closing note.

Articles:
{article}

Summary:
"""

PROMPT_TEMPLATES = {
    "cnn_dailymail": CNN_DAILYMAIL_PROMPT_TEMPLATE,
    "multi_news": MULTI_NEWS_PROMPT_TEMPLATE,
}
PROMPT_VERSIONS = {
    "cnn_dailymail": CNN_DAILYMAIL_PROMPT_VERSION,
    "multi_news": MULTI_NEWS_PROMPT_VERSION,
}
_BASE_MODEL_STOP_STRINGS = (
    "Summarize the following news article",
    "Summarize the following collection",
    "Thinking Process:",
    "Analysis:",
    "Drafting:",
    "Step 1:",
    "Self-Correction:",
    "Review against Constraints:",
    "Draft 1:",
    "Draft 2:",
    "Refinement:",
    "Final Polish:",
    "Final Output:",
    "Internal Monologue:",
    "Evaluation:",
    "thought\n",
    "\nThought:",
)


class GeneratedTokenPresencePenalty(LogitsProcessor):
    """OpenAI-style presence penalty over tokens generated in the response."""

    def __init__(self, penalty: float, prompt_width: int):
        self.penalty = float(penalty)
        self.prompt_width = int(prompt_width)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.penalty == 0.0 or input_ids.size(1) <= self.prompt_width:
            return scores

        generated_ids = input_ids[:, self.prompt_width:]
        for row_index, row_tokens in enumerate(generated_ids):
            unique_tokens = torch.unique(row_tokens)
            if unique_tokens.numel() > 0:
                scores[row_index, unique_tokens] -= self.penalty
        return scores


class SentenceCountStoppingCriteria(StoppingCriteria):
    """Stop generation if the number of sentences exceeds a threshold."""

    def __init__(self, tokenizer, prompt_width: int, max_sentences: int = 10):
        self.tokenizer = tokenizer
        self.prompt_width = int(prompt_width)
        self.max_sentences = max_sentences
        self.sentence_end_pattern = re.compile(r"[.!?]\s")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        del scores, kwargs
        if input_ids.size(1) <= self.prompt_width:
            return False
        for row in input_ids:
            generated = self.tokenizer.decode(
                row[self.prompt_width:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            sentences = self.sentence_end_pattern.findall(generated)
            if len(sentences) >= self.max_sentences:
                return True
        return False


class StopOnSubstrings(StoppingCriteria):
    """Stop generation if a base model starts continuing into another prompt."""

    def __init__(self, tokenizer, prompt_width: int, stop_strings: tuple[str, ...]):
        self.tokenizer = tokenizer
        self.prompt_width = int(prompt_width)
        self.stop_strings = tuple(s for s in stop_strings if s)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        del scores, kwargs
        if not self.stop_strings or input_ids.size(1) <= self.prompt_width:
            return False
        for row in input_ids:
            generated = self.tokenizer.decode(
                row[self.prompt_width:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if any(stop in generated for stop in self.stop_strings):
                return True
        return False


def get_summary_prompt_version(dataset_name: str) -> str:
    return PROMPT_VERSIONS.get(dataset_name, CNN_DAILYMAIL_PROMPT_VERSION)


def clean_model_output(text: str) -> str:
    """Remove thinking blocks, answer labels, and prompt continuation."""
    text = _THINK_RE.sub("", text or "").strip()
    for prefix in ("Summary:", "Final summary:", "Answer:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    for stop in _BASE_MODEL_STOP_STRINGS:
        idx = text.find(stop)
        if idx >= 0:
            text = text[:idx].strip()
            break
    return text


class SummaryGenerator:
    """Generate direct summaries or summary candidates with Transformers."""

    def __init__(
        self,
        model_path: str,
        dataset_name: str,
        device: torch.device,
        compute_dtype,
        max_input_tokens: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        presence_penalty: float,
        repetition_penalty: float,
        enable_thinking: bool,
        do_sample: str,
        tensor_parallel_size: int = 1,
    ):
        del tensor_parallel_size
        self.model_path = model_path
        self.dataset_name = dataset_name
        self.device = device
        self.max_input_tokens = int(max_input_tokens)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.min_p = float(min_p)
        self.presence_penalty = float(presence_penalty)
        self.repetition_penalty = float(repetition_penalty)
        self.enable_thinking = bool(enable_thinking)
        self.do_sample = do_sample
        self.prompt_template = PROMPT_TEMPLATES.get(dataset_name, CNN_DAILYMAIL_PROMPT_TEMPLATE)
        self.prompt_version = get_summary_prompt_version(dataset_name)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {"trust_remote_code": True, "attn_implementation": "sdpa"}
        if device.type == "cuda" and compute_dtype is not None:
            model_kwargs["torch_dtype"] = compute_dtype

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).to(device)
            self.model_class = "AutoModelForCausalLM"
        except (ValueError, TypeError, AttributeError):
            self.model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs).to(device)
            self.model_class = "AutoModelForImageTextToText"
        self.model.eval()
        self._use_stop_strings = not getattr(self.tokenizer, "chat_template", None)

    def close(self) -> None:
        del self.model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _truncate_article_tokens(self, article_text: str, overhead: int = 256) -> str:
        """Pre-truncate article to fit within max_input_tokens, preserving prompt suffix."""
        max_article_tokens = self.max_input_tokens - overhead
        if max_article_tokens <= 0:
            return article_text
        ids = self.tokenizer.encode(article_text, add_special_tokens=False)
        if len(ids) <= max_article_tokens:
            return article_text
        truncated_ids = ids[:max_article_tokens]
        # Walk back to the last sentence-ending token to avoid a mid-sentence cut.
        decoded = self.tokenizer.decode(truncated_ids, skip_special_tokens=True)
        for punct in (".", "!", "?", "\n"):
            idx = decoded.rfind(punct)
            if idx > len(decoded) * 0.7:
                return decoded[: idx + 1].strip()
        return decoded.strip()

    def _prompt_text(self, article: str) -> str:
        if self.dataset_name == "multi_news":
            docs = [d.strip() for d in article.split("|||||") if d.strip()]
            formatted = "\n\n---\n\n".join(f"[Document {i + 1}]\n{d}" for i, d in enumerate(docs))
            formatted = self._truncate_article_tokens(formatted)
            return self.prompt_template.format(article=formatted)
        article = self._truncate_article_tokens(article)
        return self.prompt_template.format(article=article)

    def _messages(self, article: str, baseline_mode: bool) -> list[dict[str, str]]:
        del baseline_mode
        return [{"role": "user", "content": self._prompt_text(article)}]

    def _render_prompt(self, article: str, baseline_mode: bool) -> str:
        if not getattr(self.tokenizer, "chat_template", None):
            return self._prompt_text(article)
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self.enable_thinking is not None:
            kwargs["enable_thinking"] = self.enable_thinking
        try:
            return self.tokenizer.apply_chat_template(
                self._messages(article, baseline_mode),
                **kwargs,
            )
        except (TypeError, ValueError):
            kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(
                self._messages(article, baseline_mode),
                **kwargs,
            )

    def _resolve_do_sample(self, num_candidates: int, baseline_mode: bool) -> bool:
        if self.do_sample == "true":
            return True
        if self.do_sample == "false":
            return False
        return bool(baseline_mode)

    def _candidate_payload(self, raw_text: str, prompt_text: str) -> dict:
        postprocessed_summary = clean_model_output(raw_text)
        return {
            "raw_model_output": raw_text,
            "postprocessed_summary": postprocessed_summary,
            "summary": postprocessed_summary,
            "prompt": prompt_text,
            "prompt_version": self.prompt_version,
        }

    def generate_batch(
        self,
        articles: list[str],
        num_candidates: int,
        baseline_mode: bool = False,
    ) -> list[list[tuple[float, dict]]]:
        prompt_texts = [self._prompt_text(article) for article in articles]
        prompts = [self._render_prompt(article, baseline_mode) for article in articles]
        inputs = self.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        input_width = int(inputs["input_ids"].shape[1])
        requested_candidates = max(1, int(num_candidates))
        do_sample = self._resolve_do_sample(requested_candidates, baseline_mode)
        num_return_sequences = requested_candidates if (do_sample or requested_candidates > 1) else 1

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": self.repetition_penalty,
            "no_repeat_ngram_size": 3,
        }
        if num_return_sequences > 1:
            generation_kwargs["num_return_sequences"] = num_return_sequences
        if do_sample:
            logits_processors = LogitsProcessorList()
            if self.presence_penalty != 0.0:
                logits_processors.append(
                    GeneratedTokenPresencePenalty(
                        penalty=self.presence_penalty,
                        prompt_width=input_width,
                    )
                )
            generation_kwargs.update(
                {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "logits_processor": logits_processors,
                }
            )
            if self.min_p > 0.0:
                generation_kwargs["min_p"] = self.min_p
        elif requested_candidates > 1:
            generation_kwargs.update(
                {
                    "num_beams": requested_candidates,
                    "early_stopping": True,
                    "return_dict_in_generate": True,
                    "output_scores": True,
                }
            )

        generation_kwargs["stopping_criteria"] = StoppingCriteriaList([
            SentenceCountStoppingCriteria(self.tokenizer, input_width, max_sentences=50),
            StopOnSubstrings(self.tokenizer, input_width, _BASE_MODEL_STOP_STRINGS),
        ])

        with torch.inference_mode():
            generated = self.model.generate(**inputs, **generation_kwargs)

        sequence_scores = None
        if hasattr(generated, "sequences"):
            rows = generated.sequences.tolist()
            sequence_scores = getattr(generated, "sequences_scores", None)
            if sequence_scores is not None:
                sequence_scores = sequence_scores.detach().float().cpu().tolist()
        else:
            rows = generated.tolist()

        grouped: list[list[tuple[float, dict]]] = []
        for sample_index in range(len(articles)):
            sample_start = sample_index * num_return_sequences
            sample_rows = rows[sample_start:sample_start + num_return_sequences]
            candidates = []
            for candidate_offset, token_ids in enumerate(sample_rows):
                new_token_ids = token_ids[input_width:]
                raw_text = self.tokenizer.decode(
                    new_token_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
                flat_index = sample_start + candidate_offset
                score = (
                    float(sequence_scores[flat_index])
                    if sequence_scores is not None and flat_index < len(sequence_scores)
                    else -float(candidate_offset)
                )
                candidates.append((score, self._candidate_payload(raw_text, prompt_texts[sample_index])))
            grouped.append(candidates)
        return grouped
