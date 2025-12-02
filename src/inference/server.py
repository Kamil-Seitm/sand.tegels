from __future__ import annotations

import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Tuple

from src.model import GenerationConfig, Model


logger = logging.getLogger(__name__)


def load_model(checkpoint_dir: Path) -> Model:
    return Model.load_latest(checkpoint_dir)


def generate_from_checkpoint(
    prompt: str,
    checkpoint_dir: Path,
    *,
    max_tokens: int = 128,
) -> str:
    model = load_model(checkpoint_dir)
    return model.generate(prompt, config=GenerationConfig(max_tokens=max_tokens))


def _handler_factory(model: Model, checkpoint_dir: Path) -> Callable:
    class GenerationHandler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:  # pragma: no cover
            logger.info("%s - %s", self.address_string(), format % args)

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") != "/generate":
                self._send_json(404, {"error": "not_found"})
                return

            content_length = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(content_length) if content_length else b"{}"
            try:
                payload = json.loads(data.decode("utf-8"))
            except json.JSONDecodeError:
                self._send_json(400, {"error": "invalid_json"})
                return

            prompt = str(payload.get("prompt", ""))
            max_tokens = int(payload.get("max_tokens", 128))

            output = model.generate(prompt, config=GenerationConfig(max_tokens=max_tokens))
            response = {
                "prompt": prompt,
                "output": output,
                "max_tokens": max_tokens,
                "checkpoint_dir": str(checkpoint_dir),
            }
            self._send_json(200, response)

    return GenerationHandler


def build_server(host: str, port: int, checkpoint_dir: Path) -> ThreadingHTTPServer:
    model = load_model(checkpoint_dir)
    handler = _handler_factory(model, checkpoint_dir)
    return ThreadingHTTPServer((host, port), handler)


def run_server(host: str, port: int, checkpoint_dir: Path) -> None:
    server = build_server(host, port, checkpoint_dir)
    logger.info("Starting inference server on %s:%s", host, port)
    try:
        server.serve_forever()
    finally:  # pragma: no cover - clean shutdown during normal execution
        server.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple REST inference server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=Path("checkpoints"),
        help="Directory containing checkpoint files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_server(args.host, args.port, args.checkpoints)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    main()
