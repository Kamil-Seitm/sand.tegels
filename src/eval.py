from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from src.model import GenerationConfig, Model


Metric = Callable[[str, str], float]


def exact_match(prediction: str, reference: str) -> float:
    return float(prediction.strip() == reference.strip())


def _lcs(a: Sequence[str], b: Sequence[str]) -> int:
    lengths = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, token_a in enumerate(a, start=1):
        for j, token_b in enumerate(b, start=1):
            lengths[i][j] = (
                lengths[i - 1][j - 1] + 1
                if token_a == token_b
                else max(lengths[i - 1][j], lengths[i][j - 1])
            )
    return lengths[-1][-1]


def rouge_l(prediction: str, reference: str) -> float:
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    if not ref_tokens:
        return 0.0
    lcs = _lcs(pred_tokens, ref_tokens)
    return lcs / len(ref_tokens)


def bleu1(prediction: str, reference: str) -> float:
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    ref_counts: MutableMapping[str, int] = {}
    for tok in ref_tokens:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1

    match = 0
    for tok in pred_tokens:
        if ref_counts.get(tok, 0) > 0:
            match += 1
            ref_counts[tok] -= 1
    precision = match / len(pred_tokens)
    brevity_penalty = min(1.0, len(pred_tokens) / len(ref_tokens))
    return precision * brevity_penalty


@dataclass
class Example:
    prompt: str
    reference: str
    subset: str = "default"


def generate_predictions(
    model: Model,
    examples: Iterable[Example],
    *,
    config: GenerationConfig | None = None,
) -> List[str]:
    return [model.generate(example.prompt, config=config) for example in examples]


def evaluate_predictions(
    predictions: Sequence[str],
    references: Sequence[str],
    metrics: Mapping[str, Metric],
) -> Dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")

    results: Dict[str, float] = {name: 0.0 for name in metrics}
    count = len(predictions)
    for pred, ref in zip(predictions, references):
        for name, metric in metrics.items():
            results[name] += metric(pred, ref)
    return {name: score / count if count else 0.0 for name, score in results.items()}


def evaluate_on_subsets(
    model: Model,
    examples: Iterable[Example],
    *,
    metrics: Mapping[str, Metric] | None = None,
    subsets: Iterable[str] | None = None,
    config: GenerationConfig | None = None,
) -> Dict[str, Dict[str, float]]:
    metrics = metrics or {"exact_match": exact_match, "rouge_l": rouge_l, "bleu1": bleu1}
    subset_names = list(subsets) if subsets is not None else None

    buckets: Dict[str, List[Example]] = {}
    for example in examples:
        key = example.subset
        if subset_names is not None and key not in subset_names:
            continue
        buckets.setdefault(key, []).append(example)

    summary: Dict[str, Dict[str, float]] = {}
    for subset_name, subset_examples in buckets.items():
        preds = generate_predictions(model, subset_examples, config=config)
        refs = [ex.reference for ex in subset_examples]
        summary[subset_name] = evaluate_predictions(preds, refs, metrics)
    return summary


__all__ = [
    "Example",
    "Metric",
    "exact_match",
    "rouge_l",
    "bleu1",
    "generate_predictions",
    "evaluate_predictions",
    "evaluate_on_subsets",
]
