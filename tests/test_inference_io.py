import http.client
import json
import threading
import time

from src.inference import server
from src.model import Model


def start_temporary_server(tmp_path, port=8765):
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "latest.json").write_text('{"prefix": "demo"}', encoding="utf-8")

    httpd = server.build_server("127.0.0.1", port, checkpoint_dir)

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)
    return httpd


def test_rest_format_and_payload(tmp_path):
    httpd = start_temporary_server(tmp_path)

    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port)
    payload = json.dumps({"prompt": "hello", "max_tokens": 4})
    conn.request("POST", "/generate", body=payload, headers={"Content-Type": "application/json"})
    response = conn.getresponse()
    body = response.read().decode()
    conn.close()

    httpd.shutdown()
    httpd.server_close()

    assert response.status == 200
    parsed = json.loads(body)
    assert parsed["prompt"] == "hello"
    assert "output" in parsed and parsed["output"].startswith("demo")
    assert parsed["max_tokens"] == 4
    assert parsed["checkpoint_dir"].endswith("ckpt")


def test_cli_generation_format(tmp_path):
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "latest.json").write_text('{"prefix": "demo"}', encoding="utf-8")

    output = server.generate_from_checkpoint("test", checkpoint_dir, max_tokens=3)
    assert output.split() == ["demo", "test"]
