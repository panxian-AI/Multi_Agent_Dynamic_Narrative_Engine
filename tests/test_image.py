import json
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from madne.config import Settings
from madne.engine.orchestrator import Orchestrator
from madne.exceptions import ImageGenError
from madne.image import ImageRequest, build_scene_prompt, create_image_client
from madne.image.comfyui import ComfyUIImageGenerator
from madne.image.factory import FallbackImageGenerator
from madne.image.hunyuan import HunyuanImageGenerator
from madne.image.mock import render_placeholder_png
from madne.llm import MockLLMClient
from madne.models import NPCProfile, PlayerInput, SceneOutline

CYBERPUNK_INPUT = "我想要一个赛博朋克世界里，关于记忆买卖的侦探故事"


def make_settings(tmp_path: Path, **overrides) -> Settings:
    defaults = {
        "use_mock": True,
        "data_dir": tmp_path,
        "memory_compress_every": 10,
        "review_max_rewrites": 2,
        "img_enabled": True,
        "img_backend": "mock",
        "img_width": 256,
        "img_height": 192,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def test_mock_png_is_valid_and_uses_seed():
    first = render_placeholder_png(64, 48, 123)
    second = render_placeholder_png(64, 48, 456)

    assert first.startswith(b"\x89PNG\r\n\x1a\n")
    assert struct.unpack(">II", first[16:24]) == (64, 48)
    assert first != second


def test_scene_prompt_contains_character_appearance():
    scene = SceneOutline(
        scene_id="scene_prompt",
        summary="侦探在雨夜咖啡馆查看芯片",
        setting="霓虹灯下的旧城咖啡馆",
        involved_npcs=["npc_a"],
        visual_prompt="rain-soaked neon cafe, wide shot",
    )
    npc = NPCProfile(
        npc_id="npc_a",
        name="林默",
        role="侦探",
        appearance="charcoal trench coat, cybernetic left eye",
    )

    prompt = build_scene_prompt(scene, None, [npc], style="cinematic concept art")

    assert "cinematic concept art" in prompt
    assert "rain-soaked neon cafe" in prompt
    assert "charcoal trench coat" in prompt
    assert "霓虹灯下的旧城咖啡馆" in prompt

    compact = build_scene_prompt(
        scene, None, [npc], style="cinematic", max_chars=256
    )
    assert len(compact) <= 256
    assert "林默" in compact




def test_hunyuan_factory_uses_api_config():
    client = create_image_client(
        Settings(
            use_mock=True,
            img_backend="hunyuan",
            img_api_key="sk-image-test",
            img_base_url="https://image.example/v1",
            img_api_model="hunyuan-image-lite",
        )
    )

    health = client.health()
    assert health["backend"] == "hunyuan"
    assert health["available"] is True
def test_unknown_backend_falls_back_to_mock():
    client = create_image_client(
        Settings(use_mock=True, img_backend="not-a-backend")
    )

    assert client.health()["backend"] == "mock"


def test_fallback_generator_degrades_when_primary_fails(tmp_path: Path):
    class BrokenClient:
        backend_name = "broken"

        def health(self):
            return {"backend": "broken", "available": False}

        def generate(self, request):  # noqa: ANN001
            raise ImageGenError("primary down")

    client = FallbackImageGenerator(BrokenClient())
    target = tmp_path / "broken.png"
    result = client.generate(
        ImageRequest(
            prompt="test",
            output_path=target,
            width=64,
            height=64,
            seed=7,
        )
    )

    assert result.backend == "mock"
    assert target.exists()
    assert any("已自动降级 mock" in warning for warning in result.warnings)


def test_orchestrator_generates_illustration(tmp_path: Path):
    orchestrator = Orchestrator(
        llm=MockLLMClient(),
        settings=make_settings(tmp_path),
    )

    output = orchestrator.run_cycle(
        PlayerInput(text=CYBERPUNK_INPUT, session_id="img")
    )

    illustration = output.illustration
    assert illustration is not None
    assert illustration.backend == "mock"
    assert illustration.scene_id == output.scene.scene_id
    assert illustration.relative_path == (
        "sessions/img/assets/cycle_001.png"
    )
    assert illustration.url == "/assets/img/cycle_001.png"
    assert (tmp_path / illustration.relative_path).exists()

    payload = json.loads(
        (
            tmp_path / "sessions" / "img" / "cycles" / "cycle_001" / "output.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["illustration"]["backend"] == "mock"
    assert payload["illustration"]["relative_path"] == illustration.relative_path


def _comfyui_handler(png: bytes):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # noqa: ANN002
            return

        def _send_json(self, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/system_stats":
                self._send_json({"system": {"comfyui_version": "test"}})
                return
            if self.path.startswith("/history/"):
                self._send_json(
                    {
                        "pid": {
                            "status": {"completed": True},
                            "outputs": {
                                "7": {
                                    "images": [
                                        {
                                            "filename": "madne_00001_.png",
                                            "subfolder": "",
                                            "type": "output",
                                        }
                                    ]
                                }
                            },
                        }
                    }
                )
                return
            if self.path.startswith("/view"):
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(png)))
                self.end_headers()
                self.wfile.write(png)
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            if self.path == "/prompt":
                self._send_json({"prompt_id": "pid"})
                return
            self.send_response(404)
            self.end_headers()

    return Handler


def test_comfyui_client_roundtrip(tmp_path: Path):
    png = render_placeholder_png(64, 48, 9)
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), _comfyui_handler(png)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = ComfyUIImageGenerator(
            f"http://127.0.0.1:{server.server_address[1]}",
            checkpoint="dummy.safetensors",
            timeout=10,
        )
        assert client.health()["available"] is True

        target = tmp_path / "comfyui.png"
        result = client.generate(
            ImageRequest(
                prompt="test prompt",
                negative_prompt="blur",
                output_path=target,
                width=64,
                height=48,
                seed=42,
            )
        )

        assert result.backend == "comfyui"
        assert (result.width, result.height) == (64, 48)
        assert target.read_bytes() == png
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _hunyuan_handler(png: bytes):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # noqa: ANN002
            return

        def _send_json(self, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if self.path != "/v1/images/generations":
                self.send_response(404)
                self.end_headers()
                return
            assert payload["model"] == "hunyuan-image-lite"
            assert payload["size"] == "128x72"
            port = self.server.server_address[1]
            self._send_json(
                {
                    "data": [
                        {"url": f"http://127.0.0.1:{port}/image.png"}
                    ]
                }
            )

        def do_GET(self):  # noqa: N802
            if self.path == "/image.png":
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(png)))
                self.end_headers()
                self.wfile.write(png)
                return
            self.send_response(404)
            self.end_headers()

    return Handler


def test_hunyuan_client_downloads_generated_image(tmp_path: Path):
    png = render_placeholder_png(128, 72, 11)
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), _hunyuan_handler(png)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = HunyuanImageGenerator(
            api_key="sk-image-test",
            base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
            model="hunyuan-image-lite",
            timeout=10,
        )
        target = tmp_path / "hunyuan.png"
        result = client.generate(
            ImageRequest(
                prompt="雨夜咖啡馆里的侦探",
                negative_prompt="模糊，水印",
                output_path=target,
                width=128,
                height=72,
                seed=3,
            )
        )

        assert result.backend == "hunyuan"
        assert (result.width, result.height) == (128, 72)
        assert target.read_bytes() == png
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_hunyuan_client_requires_api_key(tmp_path: Path):
    client = HunyuanImageGenerator(api_key="")

    with pytest.raises(ImageGenError):
        client.generate(
            ImageRequest(
                prompt="test",
                output_path=tmp_path / "missing.png",
                width=128,
                height=72,
            )
        )
