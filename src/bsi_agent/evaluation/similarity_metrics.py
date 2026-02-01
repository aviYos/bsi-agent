"""
Similarity metrics for question generation evaluation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _validate_inputs(predictions: list[str], references: list[str]) -> None:
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")


def compute_sentence_similarity(
    predictions: list[str],
    references: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    device: Optional[str] = None,
) -> float:
    """
    Compute average cosine similarity between sentence embeddings.
    Requires sentence-transformers.
    """
    _validate_inputs(predictions, references)

    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    model = SentenceTransformer(model_name, device=device)
    pred_emb = model.encode(predictions, batch_size=batch_size, convert_to_numpy=True)
    ref_emb = model.encode(references, batch_size=batch_size, convert_to_numpy=True)

    sims = cosine_similarity(pred_emb, ref_emb).diagonal()
    return float(np.mean(sims)) if len(sims) else 0.0


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    """
    Compute corpus BLEU score (smoothed).
    Requires nltk.
    """
    _validate_inputs(predictions, references)

    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

    smoothing = SmoothingFunction().method1
    refs = [[ref.split()] for ref in references]
    preds = [pred.split() for pred in predictions]
    return float(corpus_bleu(refs, preds, smoothing_function=smoothing))


def compute_rouge_l(predictions: list[str], references: list[str]) -> float:
    """
    Compute average ROUGE-L F1 score.
    Requires rouge-score.
    """
    _validate_inputs(predictions, references)

    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)["rougeL"]
        scores.append(score.fmeasure)

    return float(np.mean(scores)) if scores else 0.0


def compute_bertscore_f1(
    predictions: list[str],
    references: list[str],
    lang: str = "en",
    model_type: Optional[str] = None,
    device: Optional[str] = None,
) -> float:
    """
    Compute average BERTScore F1.
    Requires bert-score.
    """
    _validate_inputs(predictions, references)

    from bert_score import score

    P, R, F1 = score(
        predictions,
        references,
        lang=lang,
        model_type=model_type,
        device=device,
    )
    return float(F1.mean().item()) if len(F1) else 0.0

