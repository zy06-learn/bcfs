"""
FactCC 评估工具（用于最终评估阶段）。

遵循 FactCC 原研究的 document-sentence 设置：
1. 把生成的摘要拆成单句
2. 每个句子和原文配对，用 FactCC 判断是否一致
3. 对所有句子的 CORRECT 概率取平均 → 作为该摘要的 FactCC 分数
"""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


FACTCC_MODEL_NAME = "manueldeprada/FactCC"


def load_factcc_eval_model(device, model_name: str = FACTCC_MODEL_NAME):
    """
    加载 FactCC 评估模型。

    返回: (tokenizer, model, correct_id)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    correct_id = model.config.label2id.get("CORRECT")
    if correct_id is None:
        for idx, label in model.config.id2label.items():
            if label == "CORRECT":
                correct_id = int(idx)
                break
    if correct_id is None:
        correct_id = 0

    return tokenizer, model, int(correct_id)


def _split_summary_sentences(summary, segmenter):
    """用 pysbd 把摘要文本拆成单句。"""
    sentences = [sent.strip() for sent in segmenter.segment(summary) if sent.strip()]
    if not sentences and summary.strip():
        sentences = [summary.strip()]
    return sentences


def _safe_pair_tokenize(tokenizer, articles, sentences, max_length, device):
    """
    安全地对 (article, sentence) 对进行 tokenization。

    优先使用 truncation="only_first"（只截断 article）。如果因为 sentence 本身
    过长（或其他边界情况）触发 "Sequence to truncate too short" 异常，
    则降级为 truncation=True（两端都截断，长度较长的一侧优先截断）。
    """
    try:
        return tokenizer(
            articles, sentences,
            return_tensors="pt", padding="max_length",
            truncation="only_first", max_length=max_length,
        ).to(device)
    except ValueError:
        # Fallback: allow truncation on both segments (longest_first).
        return tokenizer(
            articles, sentences,
            return_tensors="pt", padding="max_length",
            truncation="longest_first", max_length=max_length,
        ).to(device)


def compute_factcc_sentence_scores(
    article, summary, tokenizer, model, device, correct_id,
    segmenter, batch_size: int = 16, max_length: int = 512,
):
    """对单条摘要的每个句子分别评分。返回每个句子的 P(CORRECT) 列表。"""
    sentences = _split_summary_sentences(summary, segmenter)
    if not sentences:
        return []

    # Reserve enough space for: (a) special tokens of a pair input, and
    # (b) at least a meaningful chunk of the article so `truncation="only_first"`
    # can work. Empirically we cap the sentence side at roughly half of
    # max_length. This avoids the HuggingFace fast-tokenizer error:
    # "Sequence to truncate too short to respect the provided max_length",
    # which is raised when the second segment + special tokens already exceed
    # max_length (leaving nothing to truncate on the first segment).
    try:
        num_special = int(tokenizer.num_special_tokens_to_add(pair=True))
    except Exception:
        num_special = 3  # BERT pair default: [CLS] A [SEP] B [SEP]

    # Keep at least half of max_length for the article side (minimum 64 tokens).
    min_article_budget = max(64, max_length // 2)
    max_sent_tokens = max(1, max_length - num_special - min_article_budget)

    truncated_sentences = []
    for sent in sentences:
        ids = tokenizer.encode(sent, add_special_tokens=False)
        if len(ids) > max_sent_tokens:
            sent = tokenizer.decode(ids[:max_sent_tokens], skip_special_tokens=True)
        truncated_sentences.append(sent)
    sentences = truncated_sentences

    sentence_scores = []
    for start in range(0, len(sentences), batch_size):
        batch_sentences = sentences[start:start + batch_size]
        batch_articles = [article] * len(batch_sentences)
        inputs = _safe_pair_tokenize(
            tokenizer, batch_articles, batch_sentences, max_length, device
        )
        with torch.no_grad():
            logits = model(**inputs).logits.float()
        probs = torch.softmax(logits, dim=-1)[:, correct_id]
        sentence_scores.extend(probs.detach().cpu().tolist())
    return sentence_scores


def compute_factcc_summary_scores(
    generated_summaries, articles, tokenizer, model, device, correct_id,
    segmenter, batch_size: int = 16, max_length: int = 512,
):
    """对所有生成摘要计算 FactCC 分数（句子级平均）。"""
    summary_scores = []
    for article, summary in zip(articles, generated_summaries):
        sentence_scores = compute_factcc_sentence_scores(
            article, summary, tokenizer, model, device, correct_id,
            segmenter, batch_size=batch_size, max_length=max_length,
        )
        if sentence_scores:
            summary_scores.append(sum(sentence_scores) / len(sentence_scores))
        else:
            summary_scores.append(0.0)
    return summary_scores
