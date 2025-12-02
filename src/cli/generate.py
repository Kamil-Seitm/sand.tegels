from __future__ import annotations

import argparse
from pathlib import Path

from src.inference.server import generate_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI text generation")
    parser.add_argument("prompt", help="Prompt to generate from")
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=Path("checkpoints"),
        help="Directory containing checkpoints",
    )
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum tokens to return")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = generate_from_checkpoint(args.prompt, args.checkpoints, max_tokens=args.max_tokens)
    print(output)


if __name__ == "__main__":  # pragma: no cover
    main()
