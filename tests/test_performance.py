import time

from src.model import GenerationConfig, Model


def test_generation_speed(tmp_path):
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "latest.json").write_text('{"prefix": "speed"}', encoding="utf-8")

    model = Model.load_latest(checkpoint_dir)
    config = GenerationConfig(max_tokens=8)

    prompts = [f"prompt {i}" for i in range(200)]
    start = time.perf_counter()
    for prompt in prompts:
        text = model.generate(prompt, config=config)
        assert text
    duration = time.perf_counter() - start

    assert duration < 1.0, f"Generation too slow: {duration} seconds"
