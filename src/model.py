from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class GenerationConfig:
    """Configuration for text generation.

    Parameters
    ----------
    max_tokens:
        Maximum number of tokens (space separated) to keep in the output.
    prefix:
        Optional prefix prepended to each generation.
    suffix:
        Optional suffix appended to each generation.
    temperature:
        Placeholder field to mirror common generation APIs.
    """

    max_tokens: int = 64
    prefix: str = ""
    suffix: str = ""
    temperature: float = 0.0


class Model:
    """A tiny deterministic text generator backed by a checkpoint file.

    The goal of this implementation is to provide a predictable interface for
    unit tests, rather than to train an actual language model.
    """

    def __init__(self, *, prefix: str = "", suffix: str = "") -> None:
        self.prefix = prefix.strip()
        self.suffix = suffix.strip()

    @classmethod
    def load_latest(cls, checkpoint_dir: Path) -> "Model":
        """Load the most recently modified checkpoint from a directory.

        A checkpoint is a JSON file containing optional ``prefix`` and ``suffix``
        keys. The newest file (by modification time) with a ``.json`` extension
        is used.
        """

        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

        checkpoints = sorted(
            checkpoint_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

        latest = checkpoints[0]
        with latest.open("r", encoding="utf-8") as fp:
            data = json.load(fp)

        prefix = str(data.get("prefix", ""))
        suffix = str(data.get("suffix", ""))
        return cls(prefix=prefix, suffix=suffix)

    def generate(self, prompt: str, *, config: Optional[GenerationConfig] = None) -> str:
        """Generate deterministic text given a prompt and optional config."""

        config = config or GenerationConfig()
        prompt = prompt or ""
        raw_output = " ".join(part for part in [self.prefix, prompt, self.suffix] if part)
        tokens = raw_output.split()
        tokens = tokens[: max(1, config.max_tokens)]
        return " ".join(tokens)


__all__ = ["GenerationConfig", "Model"]
