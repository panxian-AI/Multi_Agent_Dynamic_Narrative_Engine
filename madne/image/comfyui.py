"""ComfyUI 图像后端：通过 HTTP API 提交工作流并下载生成结果。

默认工作流只使用 ComfyUI 内置节点（CheckpointLoaderSimple / KSampler /
SaveImage）。部署方若安装了 InstantID 自定义节点，可通过 ``COMFYUI_WORKFLOW``
指向自定义工作流 JSON，并使用以下占位符：

``{{PROMPT}}`` ``{{NEGATIVE_PROMPT}}`` ``{{SEED}}`` ``{{WIDTH}}``
``{{HEIGHT}}`` ``{{STEPS}}`` ``{{GUIDANCE}}`` ``{{CHECKPOINT}}``
``{{REFERENCE_IMAGE}}``
"""

from __future__ import annotations

import copy
import json
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from ..exceptions import ImageGenError
from .base import ImageRequest, ImageResult

DEFAULT_WORKFLOW: dict[str, Any] = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "{{CHECKPOINT}}"},
    },
    "2": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "{{PROMPT}}", "clip": ["1", 1]},
    },
    "3": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "{{NEGATIVE_PROMPT}}", "clip": ["1", 1]},
    },
    "4": {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "width": "{{WIDTH}}",
            "height": "{{HEIGHT}}",
            "batch_size": 1,
        },
    },
    "5": {
        "class_type": "KSampler",
        "inputs": {
            "seed": "{{SEED}}",
            "steps": "{{STEPS}}",
            "cfg": "{{GUIDANCE}}",
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["1", 0],
            "positive": ["2", 0],
            "negative": ["3", 0],
            "latent_image": ["4", 0],
        },
    },
    "6": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
    },
    "7": {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "madne", "images": ["6", 0]},
    },
}


def _substitute(value: Any, mapping: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: _substitute(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [_substitute(item, mapping) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{{") and stripped.endswith("}}"):
            key = stripped[2:-2].strip()
            if key in mapping:
                return mapping[key]
        for key, replacement in mapping.items():
            value = value.replace("{{" + key + "}}", str(replacement))
    return value


def _png_size(data: bytes) -> tuple[int, int] | None:
    if len(data) < 24 or not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    width, height = struct.unpack(">II", data[16:24])
    return int(width), int(height)


class ComfyUIImageGenerator:
    """调用部署方的 ComfyUI 实例；地址由 ``COMFYUI_URL`` 配置。"""

    backend_name = "comfyui"

    def __init__(
        self,
        base_url: str,
        checkpoint: str = "sd_xl_base_1.0.safetensors",
        workflow_path: str | Path | None = None,
        timeout: int = 180,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.checkpoint = checkpoint
        self.workflow_path = (
            Path(workflow_path) if workflow_path is not None else None
        )
        self.timeout = max(10, int(timeout))

    # ------------------------------------------------------------------ 健康检查
    def health(self) -> dict[str, Any]:
        try:
            self._request_json("GET", "/system_stats", timeout=3)
            return {
                "backend": self.backend_name,
                "available": True,
                "detail": self.base_url,
            }
        except Exception as exc:  # 网络错误统一转为不可用
            return {
                "backend": self.backend_name,
                "available": False,
                "detail": f"{type(exc).__name__}: {exc}",
            }

    # ------------------------------------------------------------------ 生成入口
    def generate(self, request: ImageRequest) -> ImageResult:
        status = self.health()
        if not status["available"]:
            raise ImageGenError(
                f"ComfyUI 不可用（{self.base_url}）：{status['detail']}"
            )

        raw_workflow, raw_text = self._load_workflow()
        warnings: list[str] = []

        reference_name = ""
        if request.reference_images:
            if "{{REFERENCE_IMAGE}}" in raw_text:
                reference_name = self._upload_image(request.reference_images[0])
            else:
                warnings.append(
                    "当前 ComfyUI 工作流未包含 {{REFERENCE_IMAGE}} 占位符，"
                    "参考图未参与生成；InstantID 注入需要自定义工作流。"
                )
        elif request.consistency_mode == "id_injection":
            warnings.append(
                "id_injection 模式需要角色定妆照，但本回合没有可用参考图，"
                "已按 self_attn（固定 seed + 外观提示）生成。"
            )

        mapping: dict[str, Any] = {
            "PROMPT": request.prompt,
            "NEGATIVE_PROMPT": request.negative_prompt,
            "SEED": int(request.seed),
            "WIDTH": int(request.width),
            "HEIGHT": int(request.height),
            "STEPS": int(request.steps),
            "GUIDANCE": float(request.guidance_scale),
            "CHECKPOINT": self.checkpoint,
            "REFERENCE_IMAGE": reference_name,
        }
        workflow = _substitute(raw_workflow, mapping)

        prompt_id = self._submit(workflow)
        image_info = self._wait_for_image(prompt_id)
        data = self._download(image_info)

        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        request.output_path.write_bytes(data)
        size = _png_size(data) or (request.width, request.height)
        return ImageResult(
            path=request.output_path,
            backend=self.backend_name,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            seed=request.seed,
            width=size[0],
            height=size[1],
            consistency_mode=request.consistency_mode,
            reference_images=list(request.reference_images),
            warnings=warnings,
        )

    # ------------------------------------------------------------------ 内部实现
    def _load_workflow(self) -> tuple[dict[str, Any], str]:
        if self.workflow_path is not None:
            if not self.workflow_path.exists():
                raise ImageGenError(
                    f"COMFYUI_WORKFLOW 文件不存在: {self.workflow_path}"
                )
            try:
                text = self.workflow_path.read_text(encoding="utf-8")
                workflow = json.loads(text)
            except (OSError, json.JSONDecodeError) as exc:
                raise ImageGenError(
                    f"无法读取 ComfyUI 工作流 {self.workflow_path}: {exc}"
                ) from exc
            if not isinstance(workflow, dict):
                raise ImageGenError("ComfyUI 工作流 JSON 顶层必须是对象")
            return workflow, text

        workflow = copy.deepcopy(DEFAULT_WORKFLOW)
        return workflow, json.dumps(workflow, ensure_ascii=False)

    def _submit(self, workflow: dict[str, Any]) -> str:
        payload = {
            "prompt": workflow,
            "client_id": uuid.uuid4().hex,
        }
        response = self._request_json("POST", "/prompt", payload, timeout=30)
        prompt_id = response.get("prompt_id")
        if not prompt_id:
            raise ImageGenError(f"ComfyUI 未返回 prompt_id: {response}")
        return str(prompt_id)

    def _wait_for_image(self, prompt_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            history = self._request_json(
                "GET",
                f"/history/{urllib.parse.quote(prompt_id)}",
                timeout=10,
            )
            entry = history.get(prompt_id)
            if isinstance(entry, dict):
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    raise ImageGenError(
                        f"ComfyUI 任务失败: {status.get('messages', status)}"
                    )
                outputs = entry.get("outputs") or {}
                for node_output in outputs.values():
                    images = node_output.get("images") or []
                    if images:
                        return images[0]
                if status.get("completed"):
                    raise ImageGenError(
                        "ComfyUI 任务已完成，但工作流没有输出图片（请检查 SaveImage 节点）"
                    )
            time.sleep(1.0)
        raise ImageGenError(f"ComfyUI 生成超时（>{self.timeout}s）")

    def _download(self, image_info: dict[str, Any]) -> bytes:
        query = urllib.parse.urlencode(
            {
                "filename": image_info.get("filename", ""),
                "subfolder": image_info.get("subfolder", ""),
                "type": image_info.get("type", "output"),
            }
        )
        return self._request_bytes(f"/view?{query}", timeout=60)

    def _upload_image(self, path: Path) -> str:
        path = Path(path)
        if not path.exists():
            raise ImageGenError(f"参考图不存在: {path}")
        boundary = f"----madne{uuid.uuid4().hex}"
        content = path.read_bytes()
        body = bytearray()
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            (
                f'Content-Disposition: form-data; name="image"; '
                f'filename="{path.name}"\r\n'
            ).encode("utf-8")
        )
        body.extend(b"Content-Type: application/octet-stream\r\n\r\n")
        body.extend(content)
        body.extend(f"\r\n--{boundary}\r\n".encode("utf-8"))
        body.extend(
            b'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue'
        )
        body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))

        request = urllib.request.Request(
            f"{self.base_url}/upload/image",
            data=bytes(body),
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8") or "{}")
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
            raise ImageGenError(f"上传参考图到 ComfyUI 失败: {exc}") from exc

        name = str(payload.get("name") or "")
        subfolder = str(payload.get("subfolder") or "")
        if not name:
            raise ImageGenError(f"ComfyUI 上传参考图未返回文件名: {payload}")
        return f"{subfolder}/{name}" if subfolder else name

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        timeout: int | float = 30,
    ) -> Any:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except OSError:
                pass
            raise ImageGenError(
                f"ComfyUI HTTP {exc.code} {path}: {detail[:500]}"
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise ImageGenError(f"ComfyUI 请求失败 {path}: {exc}") from exc
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ImageGenError(f"ComfyUI 返回非 JSON: {raw[:200]!r}") from exc

    def _request_bytes(self, path: str, timeout: int | float = 60) -> bytes:
        try:
            with urllib.request.urlopen(
                f"{self.base_url}{path}", timeout=timeout
            ) as response:
                return response.read()
        except (urllib.error.URLError, OSError) as exc:
            raise ImageGenError(f"下载 ComfyUI 图片失败 {path}: {exc}") from exc
