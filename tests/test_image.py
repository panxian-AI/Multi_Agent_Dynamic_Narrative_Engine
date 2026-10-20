import json
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from madne.config import Settings
from madne.engine.context_store import ContextStore
from madne.engine.orchestrator import Orchestrator
from madne.exceptions import ImageGenError, ImageTimeoutError
from madne.image import ImageRequest, build_scene_prompt, create_image_client
from madne.image.base import ImageResult
from madne.image.comfyui import ComfyUIImageGenerator
from madne.image.diffusers import DiffusersImageGenerator
from madne.image.factory import FallbackImageGenerator
from madne.image.hunyuan import HunyuanImageGenerator, OpenAICompatImageGenerator
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




def test_default_backend_is_local_diffusers():
    settings = Settings()

    assert settings.img_backend == "diffusers"
    assert settings.img_timeout == 60


class _FakeDevice:
    type = "cpu"


class _FakeGenerator:
    def __init__(self, device=None):  # noqa: ANN001
        self.device = device

    def manual_seed(self, seed):  # noqa: ANN001
        return self


class _FakeTorch:
    Generator = _FakeGenerator


class _SavedImage:
    def save(self, path):  # noqa: ANN001
        Path(path).write_bytes(b"fake-png")


class _FakePipe:
    def __init__(self, delay: float, steps: int = 3) -> None:
        self.delay = delay
        self.steps = steps

    def __call__(self, **kwargs):  # noqa: ANN003
        callback = kwargs.get("callback_on_step_end")
        for step in range(self.steps):
            time.sleep(self.delay)
            if callback is not None:
                callback(self, step, None, {})
        return type("Output", (), {"images": [_SavedImage()]})()


def test_diffusers_generation_finishes_within_timeout(
    tmp_path: Path, monkeypatch
):
    generator = DiffusersImageGenerator(timeout=5)
    monkeypatch.setattr(
        generator,
        "_pipeline_or_raise",
        lambda *_args, **_kwargs: (
            _FakePipe(0.01),
            _FakeDevice(),
            _FakeTorch(),
        ),
    )

    progress: list[tuple[int, int]] = []
    result = generator.generate(
        ImageRequest(
            prompt="雨夜侦探",
            output_path=tmp_path / "sd.png",
            width=64,
            height=48,
            seed=1,
            steps=3,
            on_progress=lambda current, total: progress.append(
                (current, total)
            ),
        )
    )

    assert result.backend == "diffusers"
    assert (tmp_path / "sd.png").exists()
    assert progress == [(0, 3), (1, 3), (2, 3), (3, 3)]


def test_diffusers_aborts_after_timeout_budget(
    tmp_path: Path, monkeypatch
):
    generator = DiffusersImageGenerator(timeout=1)
    monkeypatch.setattr(
        generator,
        "_pipeline_or_raise",
        lambda *_args, **_kwargs: (
            _FakePipe(0.6),
            _FakeDevice(),
            _FakeTorch(),
        ),
    )

    with pytest.raises(ImageTimeoutError, match="超时"):
        generator.generate(
            ImageRequest(
                prompt="超时测试",
                output_path=tmp_path / "timeout.png",
                width=64,
                height=48,
                seed=1,
            )
        )


def test_seedream_factory_uses_api_config():
    client = create_image_client(
        Settings(
            use_mock=True,
            img_backend="seedream",
            img_api_key="sk-image-test",
            img_base_url="https://image.example/v1",
            img_api_model="Seedream-Image-v5.0-lite",
            img_api_size="2K",
        )
    )

    health = client.health()
    assert health["backend"] == "seedream"
    assert health["available"] is True


def test_hunyuan_alias_still_supported():
    client = create_image_client(
        Settings(
            use_mock=True,
            img_backend="hunyuan",
            img_api_key="sk-image-test",
            img_base_url="https://image.example/v1",
            img_api_model="hunyuan-image-lite",
        )
    )

    assert client.health()["backend"] == "hunyuan"
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


def test_fallback_generator_does_not_mock_on_timeout(tmp_path: Path):
    class TimeoutClient:
        backend_name = "slow"

        def health(self):
            return {"backend": "slow", "available": True}

        def generate(self, request):  # noqa: ANN001
            raise ImageTimeoutError("本地 SD 生成超时（>60s）")

    client = FallbackImageGenerator(TimeoutClient())

    with pytest.raises(ImageTimeoutError):
        client.generate(
            ImageRequest(
                prompt="test",
                output_path=tmp_path / "timeout.png",
                width=64,
                height=64,
                seed=7,
            )
        )


def test_orchestrator_skips_illustration_on_timeout(tmp_path: Path):
    class TimeoutClient:
        backend_name = "slow"

        def health(self):
            return {"backend": "slow", "available": True}

        def generate(self, request):  # noqa: ANN001
            raise ImageTimeoutError("本地 SD 生成超时（>60s）")

    orchestrator = Orchestrator(
        llm=MockLLMClient(),
        settings=make_settings(tmp_path),
    )
    orchestrator.image_client = TimeoutClient()

    output = orchestrator.run_cycle(
        PlayerInput(text=CYBERPUNK_INPUT, session_id="timeout-img")
    )

    assert output.illustration is None
    entries = orchestrator.database.load_run_log("timeout-img")
    assert any(
        entry["step_id"] == "05b_illustration"
        and entry["status"] == "timeout"
        for entry in entries
    )
    snapshot = orchestrator.database.load_latest_context("timeout-img")
    warnings = ContextStore.from_snapshot(snapshot).warnings
    assert any("本回合不生成插图" in warning for warning in warnings)


def test_orchestrator_logs_illustration_progress(tmp_path: Path):
    class ProgressClient:
        backend_name = "fake"

        def health(self):
            return {"backend": "fake", "available": True}

        def generate(self, request):  # noqa: ANN001
            for current in range(1, 5):
                assert request.on_progress is not None
                request.on_progress(current, 4)
            request.output_path.parent.mkdir(parents=True, exist_ok=True)
            request.output_path.write_bytes(
                render_placeholder_png(64, 48, request.seed)
            )
            return ImageResult(
                path=request.output_path,
                backend="fake",
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                seed=request.seed,
                width=64,
                height=48,
                consistency_mode=request.consistency_mode,
            )

    orchestrator = Orchestrator(
        llm=MockLLMClient(),
        settings=make_settings(tmp_path),
    )
    orchestrator.image_client = ProgressClient()

    output = orchestrator.run_cycle(
        PlayerInput(text=CYBERPUNK_INPUT, session_id="progress-img")
    )

    assert output.illustration is not None
    assert output.illustration.backend == "fake"
    entries = orchestrator.database.load_run_log("progress-img")
    progress = [
        entry
        for entry in entries
        if entry["status"] == "running"
        and "正在生成本回合插图" in (entry["detail"] or "")
    ]
    assert progress
    assert any("100%" in (entry["detail"] or "") for entry in progress)


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
            backend_name="hunyuan",
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


def test_ark_factory_applies_volcengine_defaults():
    client = create_image_client(
        Settings(
            use_mock=True,
            img_backend="ark",
            img_api_key="sk-image-test",
        )
    )

    health = client.health()
    assert health["backend"] == "ark"
    assert health["available"] is True
    assert "ark.cn-beijing.volces.com" in health["detail"]
    assert "doubao-seedream-4-0-250828" in health["detail"]


def _ark_handler(png: bytes):
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
            assert self.path == "/api/v3/images/generations"
            assert self.headers.get("Authorization") == "Bearer sk-image-test"
            assert payload["model"] == "doubao-seedream-4-0-250828"
            assert payload["size"] == "2K"
            assert payload["response_format"] == "url"
            assert payload["sequential_image_generation"] == "disabled"
            assert payload["stream"] is False
            assert payload["watermark"] is True
            assert "n" not in payload
            port = self.server.server_address[1]
            self._send_json(
                {
                    "data": [
                        {"url": f"http://127.0.0.1:{port}/ark.png"}
                    ]
                }
            )

        def do_GET(self):  # noqa: N802
            if self.path == "/ark.png":
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(png)))
                self.end_headers()
                self.wfile.write(png)
                return
            self.send_response(404)
            self.end_headers()

    return Handler


def test_ark_client_sends_seedream_rest_fields(tmp_path: Path):
    png = render_placeholder_png(128, 72, 13)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ark_handler(png))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = OpenAICompatImageGenerator(
            api_key="sk-image-test",
            base_url=f"http://127.0.0.1:{server.server_address[1]}/api/v3",
            model="doubao-seedream-4-0-250828",
            size="2K",
            timeout=10,
            backend_name="ark",
            watermark=True,
        )
        target = tmp_path / "ark.png"
        result = client.generate(
            ImageRequest(
                prompt="黑洞中冲出的复古列车",
                negative_prompt="模糊，水印",
                output_path=target,
                width=1280,
                height=720,
            )
        )

        assert result.backend == "ark"
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
