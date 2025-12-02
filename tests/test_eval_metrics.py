from src.eval import Example, bleu1, evaluate_on_subsets, exact_match, rouge_l
from src.model import GenerationConfig, Model


def test_metrics_are_computed_for_subsets(tmp_path):
    checkpoint = tmp_path / "checkpoints"
    checkpoint.mkdir()
    (checkpoint / "sample.json").write_text('{"prefix": "answer:"}', encoding="utf-8")

    model = Model.load_latest(checkpoint)
    examples = [
        Example(prompt="hello", reference="answer: hello", subset="a"),
        Example(prompt="world", reference="answer: world", subset="b"),
    ]

    results = evaluate_on_subsets(
        model,
        examples,
        metrics={"exact": exact_match, "rouge": rouge_l, "bleu": bleu1},
        subsets=["a", "b"],
        config=GenerationConfig(max_tokens=5),
    )

    assert set(results) == {"a", "b"}
    assert results["a"]["exact"] == 1.0
    assert results["b"]["exact"] == 1.0
    assert 0 <= results["a"]["rouge"] <= 1
    assert 0 <= results["a"]["bleu"] <= 1
