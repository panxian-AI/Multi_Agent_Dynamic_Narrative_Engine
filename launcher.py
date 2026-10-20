"""MADNE Windows 桌面启动器。

双击运行即可打开可视化界面；也可用 `--selftest` 做无界面自检。
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
import webbrowser
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText
import tkinter as tk

from madne.config import Settings
from madne.engine.orchestrator import Orchestrator
from madne.engine.run_log import RunLogEntry
from madne.exceptions import ConfigError
from madne.image import ImageRequest, create_image_client
from madne.image.hunyuan import OpenAICompatImageGenerator
from madne.llm import MockLLMClient, OpenAICompatClient, create_llm_client
from madne.models import Branch, GameOutput, PlayerInput
from madne.storage import SessionDatabase
from madne.user_config import (
    ApiConfig,
    DEFAULT_BASE_URL,
    DEFAULT_IMG_BASE_URL,
    DEFAULT_IMG_HEIGHT,
    DEFAULT_IMG_MODEL,
    DEFAULT_IMG_WIDTH,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    clear_api_config,
    config_dir,
    config_path,
    load_api_config,
    load_app_settings,
    save_api_config,
)

APP_TITLE = "MADNE · 多 Agent 动态叙事引擎"
IS_FROZEN = bool(getattr(sys, "frozen", False))
BASE_DIR = (
    Path(sys.executable).resolve().parent
    if IS_FROZEN
    else Path(__file__).resolve().parent
)
DATA_DIR = BASE_DIR / "runtime"
DEFAULT_PROMPT = "我想要一个赛博朋克世界里，关于记忆买卖的侦探故事"
MODES = {"真实模型": False, "Mock 演示": True}
ADMIN_INVITE_CODE = "admin"
PLAYER_TITLE = "MADNE · 互动故事"

STATUS_LABELS = {
    "success": "成功",
    "review_rejected": "审查驳回",
    "retry": "重试",
    "failed": "失败",
    "degraded": "降级",
    "skipped": "跳过",
    "running": "进行中",
    "timeout": "超时",
}


def _new_session_id() -> str:
    return f"desktop-{time.strftime('%Y%m%d-%H%M%S')}"


def _database() -> SessionDatabase:
    return SessionDatabase(DATA_DIR / "madne.db")


def _unique_session_id() -> str:
    base = _new_session_id()
    candidate = base
    index = 1
    database = _database()
    while database.session_exists(candidate):
        index += 1
        candidate = f"{base}-{index}"
    return candidate


def _read_spin(var: tk.IntVar, default: int, minimum: int) -> int:
    try:
        value = int(var.get())
    except (tk.TclError, TypeError, ValueError):
        value = default
    return max(minimum, value)


class ApiConfigDialog(tk.Toplevel):
    """独立的 API 配置窗口；密钥只保存在本机用户目录。"""

    def __init__(
        self,
        parent: tk.Misc,
        on_saved: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.title("API 配置 · 使用你自己的 Key")
        self.geometry("720x780")
        self.resizable(False, False)
        self.transient(parent)
        try:
            self.grab_set()
        except tk.TclError:
            self.after(120, self._grab)

        self._on_saved = on_saved
        self._queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._config = load_api_config()

        self._key_var = tk.StringVar(value=self._config.api_key)
        self._base_var = tk.StringVar(value=self._config.base_url)
        self._model_var = tk.StringVar(value=self._config.model)
        self._max_tokens_var = tk.IntVar(value=self._config.max_tokens)
        self._admin_code_var = tk.StringVar(value="")
        self._show_key_var = tk.BooleanVar(value=False)

        self._img_enabled_var = tk.BooleanVar(value=self._config.img_enabled)
        self._img_backend_var = tk.StringVar(
            value=(
                "local"
                if self._config.img_backend
                in {"diffusers", "comfyui", "mock"}
                else "api"
            )
        )
        self._img_timeout_var = tk.IntVar(value=self._config.img_timeout)
        self._img_key_var = tk.StringVar(value=self._config.img_api_key)
        self._img_base_var = tk.StringVar(value=self._config.img_base_url)
        self._img_model_var = tk.StringVar(value=self._config.img_model)
        self._img_size_var = tk.StringVar(value=self._config.img_size)
        self._img_width_var = tk.IntVar(value=self._config.img_width)
        self._img_height_var = tk.IntVar(value=self._config.img_height)
        self._show_img_key_var = tk.BooleanVar(value=False)
        self._img_hint_var = tk.StringVar(value="")
        self._img_base_var.trace_add(
            "write", lambda *_: self._update_img_hint()
        )
        self._img_model_var.trace_add(
            "write", lambda *_: self._update_img_hint()
        )
        self._img_timeout_var.trace_add(
            "write", lambda *_: self._update_img_hint()
        )

        self._status_var = tk.StringVar(
            value="填写 API Key 后点击“测试连接”或“保存”；管理员可填写邀请码。"
        )

        self._build_ui()
        self._on_backend_changed()
        self.after(100, self._poll_queue)
        self.after(50, self._center)
        self.after(80, self._bring_to_front)
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.bind("<Escape>", lambda _event: self.destroy())

    def _build_ui(self) -> None:
        frame = ttk.Frame(self, padding=16)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="配置你自己的 API", style="Title.TLabel").pack(
            anchor="w"
        )
        ttk.Label(
            frame,
            text="密钥只保存在本机配置文件，不会写入 exe、日志或分发包。",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 12))

        form = ttk.Frame(frame)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="API Key").grid(row=0, column=0, sticky="w")
        key_frame = ttk.Frame(form)
        key_frame.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        key_frame.columnconfigure(0, weight=1)
        self._key_entry = ttk.Entry(
            key_frame, textvariable=self._key_var, show="•"
        )
        self._key_entry.grid(row=0, column=0, sticky="ew")
        ttk.Checkbutton(
            key_frame,
            text="显示",
            variable=self._show_key_var,
            command=self._toggle_key_visibility,
        ).grid(row=0, column=1, padx=(6, 0))

        ttk.Label(form, text="Base URL").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        self._base_entry = ttk.Entry(form, textvariable=self._base_var)
        self._base_entry.grid(
            row=1, column=1, sticky="ew", padx=(8, 0), pady=(8, 0)
        )

        ttk.Label(form, text="模型").grid(
            row=2, column=0, sticky="w", pady=(8, 0)
        )
        self._model_entry = ttk.Entry(form, textvariable=self._model_var)
        self._model_entry.grid(
            row=2, column=1, sticky="ew", padx=(8, 0), pady=(8, 0)
        )

        ttk.Label(form, text="最大输出 tokens").grid(
            row=3, column=0, sticky="w", pady=(8, 0)
        )
        self._max_spin = ttk.Spinbox(
            form,
            from_=256,
            to=32768,
            textvariable=self._max_tokens_var,
            width=10,
        )
        self._max_spin.grid(
            row=3, column=1, sticky="w", padx=(8, 0), pady=(8, 0)
        )

        ttk.Label(form, text="管理员邀请码（可选）").grid(
            row=4, column=0, sticky="w", pady=(8, 0)
        )
        self._admin_code_entry = ttk.Entry(
            form, textvariable=self._admin_code_var, show="•"
        )
        self._admin_code_entry.grid(
            row=4, column=1, sticky="ew", padx=(8, 0), pady=(8, 0)
        )
        ttk.Label(
            form,
            text="仅管理员需要填写；普通玩家留空。",
            style="Hint.TLabel",
        ).grid(row=5, column=1, sticky="w", padx=(8, 0), pady=(4, 0))

        img_box = ttk.LabelFrame(
            frame,
            text="场景插图（本地 SD 默认，在线 API 可选）",
            padding=10,
        )
        img_box.pack(fill="x", pady=(12, 0))
        img_box.columnconfigure(1, weight=1)

        ttk.Checkbutton(
            img_box,
            text="为本回合场景生成 AI 插图",
            variable=self._img_enabled_var,
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        ttk.Label(img_box, text="生图方式").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        backend_frame = ttk.Frame(img_box)
        backend_frame.grid(
            row=1, column=1, sticky="w", padx=(8, 0), pady=(8, 0)
        )
        self._img_local_radio = ttk.Radiobutton(
            backend_frame,
            text="本地 SD（推荐）",
            variable=self._img_backend_var,
            value="local",
            command=self._on_backend_changed,
        )
        self._img_local_radio.pack(side="left")
        self._img_api_radio = ttk.Radiobutton(
            backend_frame,
            text="在线 API（可选）",
            variable=self._img_backend_var,
            value="api",
            command=self._on_backend_changed,
        )
        self._img_api_radio.pack(side="left", padx=(12, 0))

        ttk.Label(img_box, text="单张时间上限").grid(
            row=2, column=0, sticky="w", pady=(8, 0)
        )
        timeout_frame = ttk.Frame(img_box)
        timeout_frame.grid(
            row=2, column=1, sticky="w", padx=(8, 0), pady=(8, 0)
        )
        self._img_timeout_spin = ttk.Spinbox(
            timeout_frame,
            from_=10,
            to=600,
            increment=10,
            textvariable=self._img_timeout_var,
            width=8,
        )
        self._img_timeout_spin.pack(side="left")
        ttk.Label(timeout_frame, text=" 秒（默认 60）").pack(side="left")

        ttk.Label(img_box, text="生图 API Key").grid(
            row=3, column=0, sticky="w", pady=(8, 0)
        )
        img_key_frame = ttk.Frame(img_box)
        img_key_frame.grid(
            row=3, column=1, sticky="ew", padx=(8, 0), pady=(8, 0)
        )
        img_key_frame.columnconfigure(0, weight=1)
        self._img_key_entry = ttk.Entry(
            img_key_frame, textvariable=self._img_key_var, show="•"
        )
        self._img_key_entry.grid(row=0, column=0, sticky="ew")
        ttk.Checkbutton(
            img_key_frame,
            text="显示",
            variable=self._show_img_key_var,
            command=self._toggle_img_key_visibility,
        ).grid(row=0, column=1, padx=(6, 0))

        ttk.Label(img_box, text="Base URL").grid(
            row=4, column=0, sticky="w", pady=(8, 0)
        )
        self._img_base_entry = ttk.Entry(
            img_box, textvariable=self._img_base_var
        )
        self._img_base_entry.grid(
            row=4, column=1, sticky="ew", padx=(8, 0), pady=(8, 0)
        )

        ttk.Label(img_box, text="模型").grid(
            row=5, column=0, sticky="w", pady=(8, 0)
        )
        self._img_model_entry = ttk.Entry(
            img_box, textvariable=self._img_model_var
        )
        self._img_model_entry.grid(
            row=5, column=1, sticky="ew", padx=(8, 0), pady=(8, 0)
        )

        ttk.Label(img_box, text="分辨率").grid(
            row=6, column=0, sticky="w", pady=(8, 0)
        )
        size_frame = ttk.Frame(img_box)
        size_frame.grid(
            row=6, column=1, sticky="w", padx=(8, 0), pady=(8, 0)
        )
        self._img_width_spin = ttk.Spinbox(
            size_frame,
            from_=256,
            to=4096,
            increment=64,
            textvariable=self._img_width_var,
            width=8,
        )
        self._img_width_spin.pack(side="left")
        ttk.Label(size_frame, text=" × ").pack(side="left")
        self._img_height_spin = ttk.Spinbox(
            size_frame,
            from_=256,
            to=4096,
            increment=64,
            textvariable=self._img_height_var,
            width=8,
        )
        self._img_height_spin.pack(side="left")

        ttk.Label(img_box, text="尺寸档位（可选）").grid(
            row=7, column=0, sticky="w", pady=(8, 0)
        )
        self._img_size_entry = ttk.Entry(
            img_box, textvariable=self._img_size_var
        )
        self._img_size_entry.grid(
            row=7, column=1, sticky="ew", padx=(8, 0), pady=(8, 0)
        )

        ttk.Label(
            img_box,
            textvariable=self._img_hint_var,
            style="Hint.TLabel",
            wraplength=620,
        ).grid(row=8, column=0, columnspan=2, sticky="w", pady=(6, 0))

        self._test_img_btn = ttk.Button(
            img_box,
            text="测试生图",
            command=self._test_image_connection,
        )
        self._test_img_btn.grid(
            row=9, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=(14, 0))
        self._test_btn = ttk.Button(
            buttons, text="测试连接", command=self._test_connection
        )
        self._test_btn.pack(side="left")
        self._save_btn = ttk.Button(
            buttons,
            text="保存",
            style="Primary.TButton",
            command=self._save,
        )
        self._save_btn.pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="取消", command=self.destroy).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(
            buttons,
            text="LLM 控制台",
            command=self._open_console,
        ).pack(side="left", padx=(18, 0))
        ttk.Button(
            buttons,
            text="生图控制台",
            command=self._open_image_console,
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            buttons, text="清除本机配置", command=self._clear
        ).pack(side="left", padx=(8, 0))

        ttk.Label(
            frame,
            textvariable=self._status_var,
            wraplength=610,
        ).pack(anchor="w", pady=(12, 0))
        ttk.Label(
            frame,
            text=f"配置文件：{config_path()}",
            style="Hint.TLabel",
            wraplength=610,
        ).pack(anchor="w", pady=(4, 0))

    def _toggle_img_key_visibility(self) -> None:
        self._img_key_entry.configure(
            show="" if self._show_img_key_var.get() else "•"
        )

    def _on_backend_changed(self) -> None:
        local = self._img_backend_var.get() == "local"
        state = "disabled" if local else "normal"
        for widget in (
            self._img_key_entry,
            self._img_base_entry,
            self._img_model_entry,
            self._img_size_entry,
            self._img_width_spin,
            self._img_height_spin,
        ):
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass
        self._test_img_btn.configure(
            text=(
                "测试本地 SD（首次可能下载模型）"
                if local
                else "测试在线 API（消耗一次调用）"
            )
        )
        self._update_img_hint()

    def _preview_endpoint(self) -> str:
        try:
            generator = OpenAICompatImageGenerator(
                api_key="preview",
                base_url=(
                    self._img_base_var.get().strip() or DEFAULT_IMG_BASE_URL
                ),
                model=(
                    self._img_model_var.get().strip() or DEFAULT_IMG_MODEL
                ),
            )
            return generator.endpoint()
        except Exception:
            return self._img_base_var.get().strip()

    def _update_img_hint(self) -> None:
        try:
            timeout = max(10, int(self._img_timeout_var.get()))
        except (tk.TclError, TypeError, ValueError):
            timeout = 60
        if self._img_backend_var.get() == "local":
            model = os.getenv("IMG_MODEL", "stabilityai/sd-turbo")
            self._img_hint_var.set(
                f"当前：本地 SD，单张上限 {timeout} 秒。\n"
                f"模型：{model}（源码运行可在 .env 用 IMG_MODEL 修改）\n"
                "分辨率/步数由 IMG_WIDTH / IMG_HEIGHT / IMG_STEPS 控制。\n"
                "首次生图会自动下载权重（约 2.4GB）；若项目/exe 目录已有 "
                "sd_turbo.safetensors，会优先直接加载本地单文件。\n"
                "官方站点失败会自动切到 hf-mirror.com，下载支持断点续传、"
                "进度写入运行日志。\n"
                "超时或失败时本回合不生成插图，文本剧情继续。"
            )
            return

        endpoint = self._preview_endpoint()
        endpoint_text = endpoint or "（请先填写 Base URL）"
        self._img_hint_var.set(
            f"当前：在线 API，单张上限 {timeout} 秒。\n"
            f"实际请求：{endpoint_text}\n"
            "方舟 API：https://ark.cn-beijing.volces.com/api/v3，"
            "模型 doubao-seedream-4-0-250828。\n"
            "尺寸可填 2K / 4K；留空则使用上面的宽×高。"
        )

    def _test_image_connection(self) -> None:
        config = self._read_config()
        if config is None:
            return

        local = config.img_backend in {"diffusers", "comfyui", "mock"}
        if local:
            settings = replace(
                Settings.from_env(),
                img_enabled=True,
                img_backend=config.img_backend,
                img_timeout=config.img_timeout,
            )
            target = config_dir() / f"test_image_{config.img_backend}.png"
            if not messagebox.askyesno(
                "测试本地 SD",
                f"将使用本地模型：\n{settings.img_model}\n\n"
                f"尺寸 {settings.img_width}×{settings.img_height}，"
                f"{settings.img_steps} 步，上限 {config.img_timeout} 秒。\n"
                "首次运行可能需要下载模型。确定继续吗？",
                parent=self,
            ):
                return
        else:
            if not config.img_api_key.strip():
                messagebox.showwarning(
                    "缺少生图 Key",
                    "选择在线 API 时请先填写生图 API Key，"
                    "或切换为本地 SD。",
                    parent=self,
                )
                return
            if not config.img_base_url.strip():
                messagebox.showwarning(
                    "缺少 Base URL",
                    "请填写服务商控制台提供的 OpenAI 兼容 Base URL，"
                    "例如火山方舟的 https://ark.cn-beijing.volces.com/api/v3。",
                    parent=self,
                )
                return

            endpoint = self._preview_endpoint()
            if (
                "seedream" in config.img_model.lower()
                and "hunyuan.cloud.tencent.com" in endpoint.lower()
            ):
                messagebox.showwarning(
                    "Base URL 可能不正确",
                    f"当前会请求：\n{endpoint}\n\n"
                    "混元默认地址不提供 Seedream 图像接口，通常会返回 404。\n"
                    "请填写腾讯云/服务商控制台提供的 OpenAI 兼容 Base URL。",
                    parent=self,
                )

            if not messagebox.askyesno(
                "测试在线 API",
                f"将请求：\n{endpoint}\n\n"
                f"上限 {config.img_timeout} 秒，这会消耗一次生图调用，确定继续吗？",
                parent=self,
            ):
                return

            settings = replace(
                Settings.from_env(),
                img_enabled=True,
                img_backend=config.img_backend,
                img_timeout=config.img_timeout,
                img_api_key=config.img_api_key,
                img_base_url=config.img_base_url,
                img_api_model=config.img_model,
                img_api_size=config.img_size,
                img_width=config.img_width,
                img_height=config.img_height,
            )
            target = config_dir() / "test_image_api.png"

        self._set_busy(True)
        self._status_var.set("正在测试生图…")

        def progress(current: int, total: int) -> None:
            if total > 0:
                self._queue.put(
                    ("status", f"本地 SD 采样 {current}/{total}…")
                )

        def worker() -> None:
            try:
                client = create_image_client(settings)
                result = client.generate(
                    ImageRequest(
                        prompt=(
                            "一只橘猫坐在木质窗台上，窗外雨后竹林，"
                            "柔和的午后侧光，写实摄影，高清细节，不含文字"
                        ),
                        output_path=target,
                        width=settings.img_width,
                        height=settings.img_height,
                        seed=1,
                        steps=settings.img_steps,
                        guidance_scale=settings.img_guidance,
                        on_progress=progress if local else None,
                    )
                )
                if result.backend == "mock":
                    reason = "；".join(result.warnings) or "已回退 Mock"
                    self._queue.put(
                        ("error", f"生图测试失败：{reason}")
                    )
                else:
                    self._queue.put(
                        (
                            "ok",
                            f"生图成功：{result.backend} "
                            f"{result.width}x{result.height}\n"
                            f"测试图：{target}",
                        )
                    )
            except Exception as exc:
                self._queue.put(
                    (
                        "error",
                        f"生图测试失败：{type(exc).__name__}: {exc}",
                    )
                )

        threading.Thread(target=worker, daemon=True).start()

    def _toggle_key_visibility(self) -> None:
        self._key_entry.configure(
            show="" if self._show_key_var.get() else "•"
        )

    def _read_config(self, *, check_admin: bool = False) -> ApiConfig | None:
        key = self._key_var.get().strip()
        if not key:
            self._status_var.set("请先填写 API Key。")
            self._key_entry.focus_set()
            return None
        base_url = self._base_var.get().strip() or DEFAULT_BASE_URL
        model = self._model_var.get().strip() or DEFAULT_MODEL
        try:
            max_tokens = max(256, int(self._max_tokens_var.get()))
        except (tk.TclError, TypeError, ValueError):
            max_tokens = DEFAULT_MAX_TOKENS

        try:
            img_width = max(256, int(self._img_width_var.get()))
        except (tk.TclError, TypeError, ValueError):
            img_width = DEFAULT_IMG_WIDTH
        try:
            img_height = max(256, int(self._img_height_var.get()))
        except (tk.TclError, TypeError, ValueError):
            img_height = DEFAULT_IMG_HEIGHT
        try:
            img_timeout = max(10, min(600, int(self._img_timeout_var.get())))
        except (tk.TclError, TypeError, ValueError):
            img_timeout = 60

        img_base_url = (
            self._img_base_var.get().strip() or DEFAULT_IMG_BASE_URL
        )
        if self._img_backend_var.get() == "local":
            img_backend = "diffusers"
        else:
            img_backend = (
                "ark"
                if "ark.cn-beijing.volces.com" in img_base_url.lower()
                else "seedream"
            )

        admin_unlocked = self._config.admin_unlocked
        invite_code = self._admin_code_var.get().strip()
        if invite_code:
            if invite_code.lower() == ADMIN_INVITE_CODE:
                admin_unlocked = True
            elif check_admin:
                self._status_var.set(
                    "管理员邀请码无效；普通玩家请留空后保存。"
                )
                self._admin_code_entry.focus_set()
                return None

        return ApiConfig(
            api_key=key,
            base_url=base_url,
            model=model,
            max_tokens=max_tokens,
            admin_unlocked=admin_unlocked,
            img_enabled=bool(self._img_enabled_var.get()),
            img_backend=img_backend,
            img_timeout=img_timeout,
            img_api_key=self._img_key_var.get().strip(),
            img_base_url=img_base_url,
            img_model=(
                self._img_model_var.get().strip() or DEFAULT_IMG_MODEL
            ),
            img_size=self._img_size_var.get().strip(),
            img_width=img_width,
            img_height=img_height,
        )

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        for widget in (
            self._key_entry,
            self._base_entry,
            self._model_entry,
            self._max_spin,
            self._admin_code_entry,
            self._img_key_entry,
            self._img_base_entry,
            self._img_model_entry,
            self._img_size_entry,
            self._img_width_spin,
            self._img_height_spin,
            self._img_timeout_spin,
            self._img_local_radio,
            self._img_api_radio,
            self._test_btn,
            self._test_img_btn,
            self._save_btn,
        ):
            widget.configure(state=state)
        if not busy:
            self._on_backend_changed()

    def _test_connection(self) -> None:
        config = self._read_config()
        if config is None:
            return
        self._set_busy(True)
        self._status_var.set("正在测试连接…")

        def worker() -> None:
            try:
                client = OpenAICompatClient(
                    api_key=config.api_key,
                    base_url=config.base_url,
                    model=config.model,
                    max_tokens=config.max_tokens,
                    timeout=60,
                )
                response = client.complete(
                    '请只输出 JSON：{"ok": true}',
                    system="你是连通性测试助手，只输出合法 JSON。",
                    json_schema={"type": "object"},
                )
                self._queue.put(("ok", f"连接成功：{response[:180]}"))
            except Exception as exc:
                self._queue.put(
                    (
                        "error",
                        f"连接失败：{type(exc).__name__}: {exc}",
                    )
                )

        threading.Thread(target=worker, daemon=True).start()

    def _poll_queue(self) -> None:
        try:
            while True:
                _kind, message = self._queue.get_nowait()
                self._set_busy(False)
                self._status_var.set(message)
        except queue.Empty:
            pass
        try:
            if self.winfo_exists():
                self.after(100, self._poll_queue)
        except tk.TclError:
            pass

    def _save(self) -> None:
        config = self._read_config(check_admin=True)
        if config is None:
            return
        if (
            config.img_enabled
            and config.img_backend not in {"diffusers", "comfyui", "mock"}
            and not config.img_api_key.strip()
        ):
            messagebox.showwarning(
                "缺少生图 Key",
                "当前选择“在线 API”，请填写生图 API Key；\n"
                "或切换为“本地 SD（推荐）”。",
                parent=self,
            )
            return
        path = save_api_config(config)
        if self._on_saved is not None:
            self._on_saved()
        extra = "\n\n管理员界面已解锁。" if config.admin_unlocked else ""
        messagebox.showinfo(
            "已保存",
            f"API 配置已保存到：\n{path}\n\n"
            f"该文件只在本机，不会进入 exe 或日志。{extra}",
            parent=self,
        )
        self.destroy()

    def _clear(self) -> None:
        if not messagebox.askyesno(
            "清除配置",
            "确定删除本机保存的 API Key 吗？",
            parent=self,
        ):
            return
        clear_api_config()
        self._config = load_api_config()
        self._key_var.set("")
        self._admin_code_var.set("")
        self._img_enabled_var.set(self._config.img_enabled)
        self._img_backend_var.set(
            (
                "local"
                if self._config.img_backend
                in {"diffusers", "comfyui", "mock"}
                else "api"
            )
        )
        self._img_timeout_var.set(self._config.img_timeout)
        self._img_key_var.set("")
        self._img_base_var.set(self._config.img_base_url)
        self._img_size_var.set(self._config.img_size)
        self._img_model_var.set(self._config.img_model)
        self._img_width_var.set(self._config.img_width)
        self._img_height_var.set(self._config.img_height)
        self._status_var.set("已清除本机 API 配置。")
        if self._on_saved is not None:
            self._on_saved()

    def _open_console(self) -> None:
        webbrowser.open("https://platform.deepseek.com/api_keys")

    def _open_image_console(self) -> None:
        webbrowser.open("https://console.cloud.tencent.com/hunyuan")

    def _grab(self) -> None:
        try:
            self.grab_set()
        except tk.TclError:
            pass

    def _bring_to_front(self) -> None:
        try:
            self.deiconify()
            self.lift()
            self.focus_force()
            self.attributes("-topmost", True)
            self.after(500, self._drop_topmost)
            self._grab()
        except tk.TclError:
            pass

    def _drop_topmost(self) -> None:
        try:
            self.attributes("-topmost", False)
        except tk.TclError:
            pass

    def _center(self) -> None:
        try:
            self.update_idletasks()
            self.deiconify()
            parent = self.master
            if parent is None:
                return
            px = parent.winfo_rootx()
            py = parent.winfo_rooty()
            pw = parent.winfo_width()
            ph = parent.winfo_height()
            width = self.winfo_width()
            height = self.winfo_height()
            x = px + max(0, (pw - width) // 2)
            y = py + max(0, (ph - height) // 2)
            self.geometry(f"+{x}+{y}")
            self.lift()
        except tk.TclError:
            pass


class MadneApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1180x820")
        self.minsize(980, 700)

        self._queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._busy = False
        self._orchestrator: Orchestrator | None = None
        self._session = ""
        self._turn = 0
        self._last_output: GameOutput | None = None
        self._image_timeout_notice: str | None = None

        self._api_config = load_api_config()
        self._api_dialog: ApiConfigDialog | None = None
        self._ui_mode = "admin" if self._api_config.admin_unlocked else "player"
        initial_settings = load_app_settings()
        self._prompt_var = tk.StringVar(value=DEFAULT_PROMPT)
        self._session_var = tk.StringVar(value=_unique_session_id())
        self._player_setup_visible = True
        self._mode_var = tk.StringVar(
            value=(
                "真实模型"
                if (self._api_config.configured or initial_settings.llm_api_key)
                else "Mock 演示"
            )
        )
        self._compress_var = tk.IntVar(value=initial_settings.memory_compress_every)
        self._rewrites_var = tk.IntVar(value=initial_settings.review_max_rewrites)
        self._prefetch_var = tk.BooleanVar(value=initial_settings.prefetch_choices)
        self._api_status_var = tk.StringVar(value="")
        self._status_var = tk.StringVar(value="就绪：输入需求后点击“开始生成”")
        self._player_prompt_var = tk.StringVar(value=DEFAULT_PROMPT)
        self._player_status_var = tk.StringVar(value="")

        self._build_style()
        self._build_ui()
        self._refresh_api_status()
        self.bind_all("<F12>", lambda _event: self._toggle_ui())
        self.bind_all("<Control-Shift-A>", lambda _event: self._toggle_ui())
        self.bind_all("<F11>", lambda _event: self._open_api_config())
        self.bind_all(
            "<Control-Shift-K>", lambda _event: self._open_api_config()
        )
        self.after(100, self._poll_queue)
        self.after(350, self._startup_config_check)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ UI 构建
    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 18, "bold"))
        style.configure("Subtitle.TLabel", foreground="#666666")
        style.configure("Hint.TLabel", foreground="#777777")
        style.configure("Primary.TButton", font=("Microsoft YaHei UI", 10, "bold"))
        style.configure(
            "PlayerTitle.TLabel", font=("Microsoft YaHei UI", 24, "bold")
        )
        style.configure(
            "PlayerSubtitle.TLabel",
            foreground="#555555",
            font=("Microsoft YaHei UI", 11),
        )
        style.configure(
            "PlayerChoice.TLabel",
            font=("Microsoft YaHei UI", 12, "bold"),
        )

    def _build_ui(self) -> None:
        self._admin_frame = ttk.Frame(self)
        container = ttk.Frame(self._admin_frame, padding=12)
        container.pack(fill="both", expand=True)

        header = ttk.Frame(container)
        header.pack(fill="x")
        ttk.Label(header, text=APP_TITLE, style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="一句话生成结构化叙事；点击分支持续推进。RunLog、记忆与输出自动写入数据库。",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 8))

        settings = ttk.LabelFrame(container, text="创作设置", padding=10)
        settings.pack(fill="x", pady=(0, 8))
        settings.columnconfigure(1, weight=1)

        ttk.Label(settings, text="故事需求").grid(row=0, column=0, sticky="w")
        self._prompt_entry = ttk.Entry(settings, textvariable=self._prompt_var)
        self._prompt_entry.grid(
            row=0, column=1, columnspan=7, sticky="ew", padx=(8, 0)
        )

        ttk.Label(settings, text="会话 ID").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        self._session_entry = ttk.Entry(settings, textvariable=self._session_var)
        self._session_entry.grid(
            row=1, column=1, sticky="ew", padx=(8, 12), pady=(8, 0)
        )

        ttk.Label(settings, text="LLM 模式").grid(
            row=1, column=2, sticky="w", pady=(8, 0)
        )
        self._mode_combo = ttk.Combobox(
            settings,
            textvariable=self._mode_var,
            values=list(MODES),
            state="readonly",
            width=12,
        )
        self._mode_combo.grid(row=1, column=3, padx=(8, 12), pady=(8, 0))

        ttk.Label(settings, text="压缩间隔").grid(
            row=1, column=4, sticky="w", pady=(8, 0)
        )
        self._compress_spin = ttk.Spinbox(
            settings, from_=1, to=50, textvariable=self._compress_var, width=5
        )
        self._compress_spin.grid(row=1, column=5, padx=(8, 12), pady=(8, 0))

        ttk.Label(settings, text="最大重写").grid(
            row=1, column=6, sticky="w", pady=(8, 0)
        )
        self._rewrites_spin = ttk.Spinbox(
            settings, from_=0, to=5, textvariable=self._rewrites_var, width=5
        )
        self._rewrites_spin.grid(row=1, column=7, padx=(8, 0), pady=(8, 0))

        actions = ttk.Frame(settings)
        actions.grid(row=2, column=0, columnspan=8, sticky="ew", pady=(10, 0))
        self._prefetch_check = ttk.Checkbutton(
            actions,
            text="预取两层选项",
            variable=self._prefetch_var,
        )
        self._prefetch_check.pack(side="left", padx=(0, 14))
        self._start_btn = ttk.Button(
            actions,
            text="开始生成",
            style="Primary.TButton",
            command=self._start_generation,
        )
        self._start_btn.pack(side="left")
        self._resume_btn = ttk.Button(
            actions, text="恢复会话", command=self._resume_session
        )
        self._resume_btn.pack(side="left", padx=(8, 0))
        self._new_btn = ttk.Button(
            actions, text="新会话", command=self._new_session
        )
        self._new_btn.pack(side="left", padx=(8, 0))
        self._api_btn = ttk.Button(
            actions, text="API 配置", command=self._open_api_config
        )
        self._api_btn.pack(side="left", padx=(8, 0))
        self._switch_player_btn = ttk.Button(
            actions,
            text="玩家界面",
            command=lambda: self._switch_ui("player"),
        )
        self._switch_player_btn.pack(side="left", padx=(8, 0))
        self._open_btn = ttk.Button(
            actions, text="打开输出目录", command=self._open_output_dir
        )
        self._open_btn.pack(side="left", padx=(20, 0))
        self._json_btn = ttk.Button(
            actions, text="打开最新 JSON", command=self._open_latest_json
        )
        self._json_btn.pack(side="left", padx=(8, 0))

        ttk.Label(
            settings,
            text=f"数据目录：{DATA_DIR}",
            style="Hint.TLabel",
        ).grid(row=3, column=0, columnspan=8, sticky="w", pady=(8, 0))
        ttk.Label(
            settings,
            textvariable=self._api_status_var,
            style="Hint.TLabel",
        ).grid(row=4, column=0, columnspan=8, sticky="w", pady=(2, 0))

        paned = ttk.Panedwindow(container, orient="horizontal")
        paned.pack(fill="both", expand=True)

        log_frame = ttk.LabelFrame(paned, text="运行日志", padding=6)
        self._log_text = ScrolledText(
            log_frame,
            wrap="word",
            font=("Consolas", 9),
            state="disabled",
        )
        self._log_text.pack(fill="both", expand=True)
        paned.add(log_frame)

        notebook = ttk.Notebook(paned)
        result_tab = ttk.Frame(notebook, padding=6)
        notebook.add(result_tab, text="场景 / 选项")
        self._choices_frame = ttk.LabelFrame(
            result_tab, text="下一步：点击选项继续", padding=8
        )
        self._choices_frame.pack(fill="x", pady=(0, 6))
        self._choices_hint = ttk.Label(
            self._choices_frame,
            text="生成完成后，点击一个分支继续推进剧情。",
            style="Hint.TLabel",
        )
        self._choices_hint.pack(anchor="w")

        self._result_text = ScrolledText(
            result_tab,
            wrap="word",
            font=("Microsoft YaHei UI", 10),
            state="disabled",
        )
        self._result_text.pack(fill="both", expand=True)

        json_tab = ttk.Frame(notebook, padding=6)
        notebook.add(json_tab, text="完整 JSON")
        self._json_text = ScrolledText(
            json_tab,
            wrap="none",
            font=("Consolas", 9),
            state="disabled",
        )
        self._json_text.pack(fill="both", expand=True)

        cache_tab = ttk.Frame(notebook, padding=6)
        notebook.add(cache_tab, text="预取选项")
        self._cache_text = ScrolledText(
            cache_tab,
            wrap="word",
            font=("Microsoft YaHei UI", 9),
            state="disabled",
        )
        self._cache_text.pack(fill="both", expand=True)
        paned.add(notebook)
        self.after(120, lambda: paned.sashpos(0, 410))

        status_bar = ttk.Frame(container)
        status_bar.pack(fill="x", pady=(8, 0))
        ttk.Label(status_bar, textvariable=self._status_var).pack(side="left")
        ttk.Label(
            status_bar,
            text="管理员界面 · F12 切换玩家界面",
            style="Hint.TLabel",
        ).pack(side="right", padx=(0, 14))
        ttk.Label(
            status_bar,
            text="MADNE 0.1 · DeepSeek / OpenAI 兼容",
            style="Hint.TLabel",
        ).pack(side="right")

        self._build_player_ui()
        self._apply_ui_mode()

    def _build_player_ui(self) -> None:
        """玩家界面只保留故事、选项与开始操作。"""
        self._player_frame = ttk.Frame(self, padding=24)

        header = ttk.Frame(self._player_frame)
        header.pack(fill="x")
        ttk.Label(
            header, text=PLAYER_TITLE, style="PlayerTitle.TLabel"
        ).pack(anchor="w")
        ttk.Label(
            header,
            text="你的选择会改变故事走向。",
            style="PlayerSubtitle.TLabel",
        ).pack(anchor="w", pady=(2, 16))

        self._player_setup_frame = ttk.Frame(self._player_frame)
        setup_box = ttk.Frame(self._player_setup_frame)
        setup_box.pack(expand=True, fill="x", padx=70)
        ttk.Label(
            setup_box,
            text="你想体验什么故事？",
            style="PlayerTitle.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            setup_box,
            text="例如：赛博朋克城市里，一名侦探追查被买卖的记忆。",
            style="PlayerSubtitle.TLabel",
        ).pack(anchor="w", pady=(4, 10))
        self._player_prompt_entry = ttk.Entry(
            setup_box,
            textvariable=self._player_prompt_var,
            font=("Microsoft YaHei UI", 12),
        )
        self._player_prompt_entry.pack(fill="x", ipady=6)
        self._player_start_btn = ttk.Button(
            setup_box,
            text="开始故事",
            style="Primary.TButton",
            command=self._player_start,
        )
        self._player_start_btn.pack(anchor="w", pady=(14, 0))
        self._player_api_btn = ttk.Button(
            setup_box,
            text="API 设置",
            command=self._open_api_config,
        )
        self._player_api_btn.pack(anchor="w", pady=(12, 0))

        self._player_story_frame = ttk.Frame(self._player_frame)
        self._player_story_frame.columnconfigure(0, weight=1)
        self._player_story_frame.rowconfigure(3, weight=1)
        story_header = ttk.Frame(self._player_story_frame)
        story_header.grid(row=0, column=0, sticky="ew")
        ttk.Label(
            story_header, text="故事", style="PlayerTitle.TLabel"
        ).pack(side="left")
        self._player_new_btn = ttk.Button(
            story_header,
            text="新的故事",
            command=self._player_new_story,
        )
        self._player_new_btn.pack(side="right")
        self._player_status_label = ttk.Label(
            self._player_story_frame,
            textvariable=self._player_status_var,
            style="PlayerSubtitle.TLabel",
            wraplength=1000,
        )
        self._player_status_label.grid(
            row=1, column=0, sticky="w", pady=(2, 8)
        )
        self._player_image_label = ttk.Label(
            self._player_story_frame, anchor="center"
        )
        self._player_image: tk.PhotoImage | None = None
        self._player_story_text = ScrolledText(
            self._player_story_frame,
            wrap="word",
            font=("Microsoft YaHei UI", 11),
            state="disabled",
            bg="#fbfbfb",
            relief="flat",
        )
        self._player_story_text.grid(row=3, column=0, sticky="nsew")
        # 选项固定在第 4 行，故事文本伸缩，避免选项被挤出窗口。
        self._player_choices_frame = ttk.Frame(self._player_story_frame)
        self._player_choices_frame.grid(
            row=4, column=0, sticky="ew", pady=(12, 0)
        )

    # ------------------------------------------------------------------ 界面切换
    def _apply_ui_mode(self) -> None:
        if self._ui_mode == "admin" and self._api_config.admin_unlocked:
            self._player_frame.pack_forget()
            self._admin_frame.pack(fill="both", expand=True)
            self.title(APP_TITLE)
        else:
            self._ui_mode = "player"
            self._admin_frame.pack_forget()
            self._player_frame.pack(fill="both", expand=True)
            self.title(PLAYER_TITLE)
            self._sync_player_view()

    def _switch_ui(self, mode: str) -> None:
        if mode == "admin" and not self._api_config.admin_unlocked:
            return
        self._ui_mode = mode
        if mode == "admin":
            self._player_frame.pack_forget()
            self._admin_frame.pack(fill="both", expand=True)
            self.title(APP_TITLE)
            self._refresh_api_status()
        else:
            self._admin_frame.pack_forget()
            self._player_frame.pack(fill="both", expand=True)
            self.title(PLAYER_TITLE)
            self._sync_player_view()

    def _toggle_ui(self) -> None:
        if not self._api_config.admin_unlocked:
            return
        self._switch_ui("admin" if self._ui_mode == "player" else "player")

    def _sync_player_view(self) -> None:
        if self._busy:
            if (
                self._last_output is not None
                and not self._player_story_text.get("1.0", "end").strip()
            ):
                self._render_player_output(self._last_output)
            self._show_player_story()
            self._player_status_var.set("故事生成中…")
        elif self._last_output is not None and not self._player_setup_visible:
            if not self._player_story_text.get("1.0", "end").strip():
                self._render_player_output(self._last_output)
            self._show_player_story()
        else:
            self._show_player_setup()

    def _show_player_setup(self) -> None:
        self._player_story_frame.pack_forget()
        self._player_setup_frame.pack(fill="both", expand=True)

    def _show_player_story(self) -> None:
        self._player_setup_frame.pack_forget()
        self._player_story_frame.pack(fill="both", expand=True)

    def _player_start(self) -> None:
        if self._busy:
            return
        prompt = self._player_prompt_var.get().strip()
        if not prompt:
            messagebox.showwarning("缺少故事设定", "请先写下你想体验的故事。")
            self._player_prompt_entry.focus_set()
            return
        self._prompt_var.set(prompt)
        self._session_var.set(_unique_session_id())
        self._start_generation()

    def _player_new_story(self) -> None:
        if self._busy:
            return
        self._clear_player_result()
        self._player_status_var.set("")
        self._player_setup_visible = True
        self._show_player_setup()
        self._player_prompt_entry.focus_set()

    def _clear_player_result(self) -> None:
        self._set_text(self._player_story_text, "")
        self._player_image_label.grid_remove()
        self._player_image = None
        for child in self._player_choices_frame.winfo_children():
            child.destroy()

    def _render_player_illustration(self, output: GameOutput) -> None:
        illustration = output.illustration
        if illustration is None:
            self._player_image_label.grid_remove()
            self._player_image = None
            return
        path = DATA_DIR / illustration.relative_path
        if not path.exists():
            self._player_image_label.grid_remove()
            self._player_image = None
            return
        try:
            photo = tk.PhotoImage(master=self, file=str(path))
        except tk.TclError as exc:
            self._player_image_label.grid_remove()
            self._player_image = None
            self._append_log_text(f"插图加载失败：{exc}")
            return

        # Tk 只支持整数倍缩放；同时限制宽高，避免插图把故事与选项挤出窗口。
        max_width = 720
        max_height = 260
        factor = max(
            1,
            (photo.width() + max_width - 1) // max_width,
            (photo.height() + max_height - 1) // max_height,
        )
        if factor > 1:
            photo = photo.subsample(factor, factor)
        self._player_image = photo
        self._player_image_label.configure(image=photo)
        self._player_image_label.image = photo
        self._player_image_label.grid(
            row=2, column=0, sticky="ew", pady=(0, 10)
        )

        failure = next(
            (
                warning
                for warning in illustration.warnings
                if "失败" in warning or "降级" in warning
            ),
            None,
        )
        if illustration.backend == "mock" and failure:
            reason = failure.replace("已自动降级 mock：", "")[:110]
            self._player_status_var.set(f"AI 生图失败，已显示占位图：{reason}")

    def _render_player_output(self, output: GameOutput) -> None:
        self._player_setup_visible = False
        scene = output.scene
        script = output.script
        lines: list[str] = [scene.summary, "", f"【环境】{scene.setting}", ""]
        if scene.actions:
            lines.append("【动作】")
            lines.extend(
                f"{index}. {action}"
                for index, action in enumerate(scene.actions, 1)
            )
            lines.append("")
        if scene.events:
            lines.append("【事件】")
            lines.extend(f"· {event}" for event in scene.events)
            lines.append("")
        lines.append("【对话】")
        for line in script.lines:
            lines.append(f"{line.speaker}：{line.text}")
            if line.inner_monologue:
                lines.append(f"（内心：{line.inner_monologue}）")
        lines.append("")

        self._set_text(self._player_story_text, "\n".join(lines))
        self._player_status_var.set("请选择你的行动")
        if self._image_timeout_notice:
            self._player_status_var.set(self._image_timeout_notice)
            self._image_timeout_notice = None
        self._show_player_story()
        self._render_player_illustration(output)
        self._render_player_choices(output)

    def _render_player_choices(self, output: GameOutput) -> None:
        for child in self._player_choices_frame.winfo_children():
            child.destroy()
        ttk.Label(
            self._player_choices_frame,
            text="你的选择",
            style="PlayerChoice.TLabel",
        ).pack(anchor="w", pady=(0, 6))

        buttons: list[tk.Button] = []
        for index, branch in enumerate(output.choices, 1):
            button = tk.Button(
                self._player_choices_frame,
                text=f"{index}. {branch.description}",
                wraplength=900,
                justify="left",
                anchor="w",
                relief="groove",
                bg="#eef4ff",
                activebackground="#d8e6ff",
                activeforeground="#111111",
                cursor="hand2",
                padx=12,
                pady=8,
                font=("Microsoft YaHei UI", 11),
                state="disabled" if self._busy else "normal",
                command=lambda selected=branch: self._choose_branch(selected),
            )
            button.pack(fill="x", pady=4)
            buttons.append(button)

        self.update_idletasks()
        wrap = max(520, self._player_choices_frame.winfo_width() - 40)
        for button in buttons:
            button.configure(wraplength=wrap)

    # ------------------------------------------------------------------ 运行控制
    def _start_generation(self) -> None:
        if self._busy:
            return
        prompt = self._prompt_var.get().strip()
        if not prompt:
            messagebox.showwarning("缺少需求", "请先输入故事需求。")
            return

        session = self._session_var.get().strip() or _unique_session_id()
        session_dir = DATA_DIR / "sessions" / session
        if _database().session_exists(session) or session_dir.exists():
            if messagebox.askyesno(
                "会话已存在",
                f"会话 {session} 已有运行记录。\n是否从断点恢复？\n\n"
                "选择“否”将自动创建新会话。",
            ):
                self._session_var.set(session)
                self._resume_session()
                return
            session = f"{session}-{time.strftime('%H%M%S')}"
            self._session_var.set(session)

        orchestrator = self._prepare_orchestrator(session)
        if orchestrator is None:
            return

        self._orchestrator = orchestrator
        self._session = session
        self._turn = 1
        self._last_output = None
        self._image_timeout_notice = None
        self._clear_result()
        self._clear_player_result()
        self._player_setup_visible = False
        self._show_player_story()
        self._player_status_var.set("故事生成中…")
        self._set_text(self._log_text, "")
        self._append_log_text(f"=== 新会话 {session} ===")
        self._append_log_text(f"需求：{prompt}")
        self._status_var.set("第 1 回合生成中…")
        self._start_worker(
            lambda: orchestrator.run_cycle(
                PlayerInput(text=prompt, session_id=session)
            )
        )

    def _resume_session(self) -> None:
        if self._busy:
            return
        session = self._session_var.get().strip()
        if not session:
            messagebox.showwarning("缺少会话", "请先填写会话 ID。")
            return

        orchestrator = self._prepare_orchestrator(session)
        if orchestrator is None:
            return

        self._orchestrator = orchestrator
        self._session = session
        self._turn = max(1, self._turn)
        self._append_log_text(f"=== 恢复会话 {session} ===")
        self._status_var.set("正在恢复…")
        self._start_worker(lambda: orchestrator.resume(session))

    def _prepare_orchestrator(self, session_id: str) -> Orchestrator | None:
        settings = self._make_settings()
        mock = MODES.get(self._mode_var.get(), False)
        if not mock and not settings.llm_api_key:
            if self._ui_mode == "player":
                messagebox.showinfo(
                    "需要配置 API",
                    "请先配置 API Key，然后再开始故事。",
                )
            else:
                messagebox.showinfo(
                    "需要配置 API",
                    "请先点击“API 配置”，输入你自己的 API Key。",
                )
            self._open_api_config()
            return None

        try:
            llm = create_llm_client(settings, mock=mock)
            return Orchestrator(
                llm=llm,
                settings=settings,
                on_event=self._on_event,
            )
        except ConfigError as exc:
            self._append_log_text(f"!!! API 配置错误：{exc}")
            if self._ui_mode == "player":
                messagebox.showerror(
                    "API 配置不完整",
                    "请先在配置窗口补全 API 信息。",
                )
            else:
                messagebox.showerror("API 配置不完整", str(exc))
            self._open_api_config()
            return None
        except Exception as exc:
            self._append_log_text(
                f"!!! 初始化失败：{type(exc).__name__}: {exc}"
            )
            if self._ui_mode == "player":
                messagebox.showerror(
                    "故事暂时无法开始",
                    "请稍后重试，或联系管理员。",
                )
            else:
                messagebox.showerror(
                    "初始化失败",
                    f"{type(exc).__name__}: {exc}",
                )
            return None

    def _make_settings(self) -> Settings:
        base = load_app_settings()
        return replace(
            base,
            data_dir=DATA_DIR,
            model_cache_dir=BASE_DIR / "models",
            memory_compress_every=_read_spin(self._compress_var, 2, 1),
            review_max_rewrites=_read_spin(self._rewrites_var, 2, 0),
            prefetch_choices=bool(self._prefetch_var.get()),
        )

    def _refresh_api_status(self) -> None:
        config = load_api_config()
        local = config.img_backend in {"diffusers", "comfyui", "mock"}
        if not config.img_enabled:
            img_status = "生图：已关闭"
        elif local:
            img_status = (
                f"生图：本地 {config.img_backend}"
                f"（≤{config.img_timeout}s）"
            )
        elif config.img_configured:
            img_status = f"生图：在线 API（≤{config.img_timeout}s）"
        else:
            img_status = "生图：在线 API 未配置"
        if config.configured:
            self._api_status_var.set(
                f"API：已配置 {config.model}（本机配置）；{img_status}"
            )
            return
        base = load_app_settings()
        if base.llm_api_key:
            self._api_status_var.set(
                f"API：使用本机/环境配置 {base.llm_model}；{img_status}"
            )
        else:
            self._api_status_var.set(
                f"API：未配置（可先使用 Mock 演示）；{img_status}"
            )

    def _startup_config_check(self) -> None:
        config = load_api_config()
        if config.configured or load_app_settings().llm_api_key:
            return
        self._mode_var.set("Mock 演示")
        self._status_var.set(
            "首次使用：请在“API 配置”中输入你自己的 Key，或先用 Mock 演示"
        )
        self._open_api_config()

    def _open_api_config(self) -> None:
        if self._api_dialog is not None:
            try:
                if self._api_dialog.winfo_exists():
                    self._api_dialog.lift()
                    self._api_dialog.focus_force()
                    return
            except tk.TclError:
                pass
        self._api_dialog = ApiConfigDialog(self, on_saved=self._on_api_saved)
        try:
            self._api_dialog.lift()
            self._api_dialog.focus_force()
        except tk.TclError:
            pass

    def _on_api_saved(self) -> None:
        was_admin = self._ui_mode == "admin"
        self._api_config = load_api_config()
        self._refresh_api_status()

        # 让当前会话立即使用新保存的生图配置（无需重开故事）
        if self._orchestrator is not None:
            updated = self._make_settings()
            self._orchestrator.settings = replace(
                self._orchestrator.settings,
                img_enabled=updated.img_enabled,
                img_backend=updated.img_backend,
                img_api_key=updated.img_api_key,
                img_base_url=updated.img_base_url,
                img_api_size=updated.img_api_size,
                img_api_model=updated.img_api_model,
                img_width=updated.img_width,
                img_height=updated.img_height,
                img_style=updated.img_style,
            )
            self._orchestrator.image_client = (
                create_image_client(self._orchestrator.settings)
                if self._orchestrator.settings.img_enabled
                else None
            )

        if self._api_config.configured:
            self._mode_var.set("真实模型")
            self._status_var.set("API 配置已保存，可以开始生成")
        elif load_app_settings().llm_api_key:
            self._mode_var.set("真实模型")
            self._status_var.set("本机配置已清除，将使用 .env 中的 API 配置")
        else:
            self._mode_var.set("Mock 演示")
            self._status_var.set("本机 API 配置已清除，将使用 Mock 演示")

        if self._api_config.admin_unlocked and not was_admin:
            self._switch_ui("admin")
        elif not self._api_config.admin_unlocked and was_admin:
            self._switch_ui("player")

    def _choose_branch(self, branch: Branch) -> None:
        if self._busy:
            self._status_var.set("正在生成，请稍候…")
            self._player_status_var.set("正在生成，请稍候…")
            return
        if self._orchestrator is None or not self._session:
            self._status_var.set("请先生成一次故事，再选择选项")
            self._player_status_var.set("请先开始一个故事")
            return
        choice = f"{branch.branch_id}. {branch.description}"
        self._turn += 1
        self._image_timeout_notice = None
        orchestrator = self._orchestrator
        session = self._session
        self._status_var.set(f"第 {self._turn} 回合生成中…")
        self._player_status_var.set("故事继续中…")
        self._start_worker(
            lambda: orchestrator.run_cycle(
                PlayerInput(
                    text=f"我选择：{choice}。请继续推进剧情。",
                    session_id=session,
                ),
                player_choice=choice,
                player_branch_id=branch.branch_id,
            )
        )

    def _start_worker(self, fn: Callable[[], GameOutput]) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._set_busy(True)

        def runner() -> None:
            try:
                output = fn()
                self._queue.put(("output", output))
            except Exception as exc:  # 真实 LLM 错误需要反馈到 UI
                self._queue.put(("error", f"{type(exc).__name__}: {exc}"))
            finally:
                self._queue.put(("idle", None))

        self._worker = threading.Thread(target=runner, daemon=True)
        self._worker.start()

    def _on_event(self, entry: RunLogEntry) -> None:
        self._queue.put(("progress", entry))

    # ------------------------------------------------------------------ 队列与渲染
    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "progress":
                    entry = payload  # type: ignore[assignment]
                    if entry.status == "running":
                        self._status_var.set(
                            entry.detail or f"正在执行 {entry.step_id}…"
                        )
                        if self._ui_mode == "player":
                            self._player_status_var.set(
                                entry.detail or "故事生成中…"
                            )
                    elif entry.status == "timeout":
                        self._image_timeout_notice = (
                            entry.detail
                            or "插图生成超时，本回合不生成插图"
                        )
                        self._status_var.set(self._image_timeout_notice)
                        if self._ui_mode == "player":
                            self._player_status_var.set(
                                self._image_timeout_notice
                            )
                    self._append_log(entry)
                elif kind == "status":
                    self._status_var.set(str(payload))
                elif kind == "output":
                    self._last_output = payload  # type: ignore[assignment]
                    self._render_output(self._last_output)
                    self._status_var.set(f"第 {self._turn} 回合生成完成")
                elif kind == "error":
                    self._status_var.set("运行失败")
                    self._append_log_text(f"!!! {payload}")
                    if self._ui_mode == "player":
                        self._player_status_var.set("故事生成失败，请重试")
                        messagebox.showerror(
                            "故事暂时无法继续",
                            "请稍后重试，或联系管理员。",
                        )
                    else:
                        messagebox.showerror("运行失败", str(payload))
                elif kind == "idle":
                    self._set_busy(False)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _append_log(self, entry: RunLogEntry) -> None:
        label = STATUS_LABELS.get(entry.status, entry.status)
        line = (
            f"[回合 {entry.cycle}] {entry.step_id:<16} {label:<6} "
            f"attempt={entry.attempt} {entry.duration_ms / 1000:.1f}s"
        )
        if entry.error:
            line += f"\n    ↳ {entry.error[:300]}"
        if entry.detail:
            line += f"\n    ↳ {entry.detail[:300]}"
        self._append_log_text(line)

    def _append_log_text(self, text: str) -> None:
        self._log_text.configure(state="normal")
        self._log_text.insert("end", text + "\n")
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    def _render_output(self, output: GameOutput) -> None:
        scene = output.scene
        script = output.script
        lines: list[str] = [
            f"【场景 {scene.scene_id}】",
            scene.summary,
            "",
            f"【环境】{scene.setting}",
            "",
        ]
        if scene.actions:
            lines.append("【动作】")
            lines.extend(f"{index}. {action}" for index, action in enumerate(scene.actions, 1))
            lines.append("")
        if scene.events:
            lines.append("【事件】")
            lines.extend(f"· {event}" for event in scene.events)
            lines.append("")
        lines.append("【台本】")
        for line in script.lines:
            lines.append(f"{line.speaker}（{line.emotion}）：{line.text}")
            if line.inner_monologue:
                lines.append(f"    内心：{line.inner_monologue}")
        lines.append("")

        self._set_text(self._result_text, "\n".join(lines))
        self._set_text(
            self._json_text,
            json.dumps(output.to_engine_dict(), ensure_ascii=False, indent=2),
        )
        if output.illustration is not None:
            self._append_log_text(
                f"插图：{output.illustration.relative_path}"
                f"（{output.illustration.backend}）"
            )
        self._render_choice_cache(output)
        self._render_choices(output)
        self._render_player_output(output)
        self._start_btn.configure(state="disabled")

    def _render_choice_cache(self, output: GameOutput) -> None:
        if not self._prefetch_var.get():
            self._set_text(self._cache_text, "当前设置已关闭选项预取。")
            return
        if self._orchestrator is None or not self._session:
            self._set_text(self._cache_text, "当前会话尚未预取选项。")
            return
        cache = self._orchestrator.choice_cache_for(self._session)
        if cache is None:
            self._set_text(
                self._cache_text,
                "尚未预取。请确认“预取两层选项”已开启，并完成一次生成。",
            )
            return

        lines: list[str] = []
        for branch in output.choices:
            nexts = cache.next_choices(branch.branch_id)
            lines.append(f"【{branch.branch_id}. {branch.description}】")
            if not nexts:
                lines.append("  未缓存下一层选项")
            for next_choice in nexts:
                lines.append(
                    f"  {next_choice.path[-1]}. {next_choice.description}"
                    f"（{next_choice.risk_level}）"
                )
                for grandchild in cache.grandchildren(
                    branch.branch_id, next_choice.path[-1]
                ):
                    lines.append(
                        f"      ↳ {grandchild.path[-1]}. {grandchild.description}"
                        f"（{grandchild.risk_level}）"
                    )
            lines.append("")
        self._set_text(self._cache_text, "\n".join(lines).strip())

    def _render_choices(self, output: GameOutput) -> None:
        for child in self._choices_frame.winfo_children():
            child.destroy()
        ttk.Label(
            self._choices_frame,
            text="点击一个选项继续（生成期间按钮会暂时禁用）：",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(0, 4))

        buttons: list[tk.Button] = []
        for branch in output.choices:
            button = tk.Button(
                self._choices_frame,
                text=(
                    f"▶ {branch.branch_id}. {branch.description}"
                    f"（风险：{branch.risk_level}）"
                ),
                wraplength=720,
                justify="left",
                anchor="w",
                relief="groove",
                bg="#e8f0fe",
                activebackground="#cfe2ff",
                activeforeground="#111111",
                cursor="hand2",
                padx=10,
                pady=6,
                font=("Microsoft YaHei UI", 10),
                command=lambda selected=branch: self._choose_branch(selected),
            )
            button.pack(fill="x", pady=3)
            buttons.append(button)

        self.update_idletasks()
        wrap = max(320, self._choices_frame.winfo_width() - 60)
        for button in buttons:
            button.configure(wraplength=wrap)

    @staticmethod
    def _set_text(widget: ScrolledText, content: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.configure(state="disabled")

    def _clear_result(self) -> None:
        self._set_text(self._result_text, "")
        self._set_text(self._json_text, "")
        self._set_text(self._cache_text, "")
        for child in self._choices_frame.winfo_children():
            child.destroy()
        ttk.Label(
            self._choices_frame,
            text="生成完成后，点击一个分支继续推进剧情。",
            style="Hint.TLabel",
        ).pack(anchor="w")

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        for widget in (
            self._start_btn,
            self._resume_btn,
            self._new_btn,
            self._open_btn,
            self._json_btn,
            self._prompt_entry,
            self._session_entry,
            self._compress_spin,
            self._rewrites_spin,
            self._prefetch_check,
            self._api_btn,
            self._player_prompt_entry,
            self._player_start_btn,
            self._player_api_btn,
            self._player_new_btn,
        ):
            widget.configure(state=state)
        self._mode_combo.configure(state="disabled" if busy else "readonly")
        for frame in (self._choices_frame, self._player_choices_frame):
            for child in frame.winfo_children():
                if isinstance(child, tk.Button):
                    child.configure(state="disabled" if busy else "normal")
        if not busy:
            self._start_btn.configure(
                state="normal" if self._last_output is None else "disabled"
            )

    # ------------------------------------------------------------------ 辅助操作
    def _new_session(self) -> None:
        if self._busy:
            return
        self._session_var.set(_unique_session_id())
        self._orchestrator = None
        self._session = ""
        self._turn = 0
        self._last_output = None
        self._image_timeout_notice = None
        self._clear_result()
        self._clear_player_result()
        self._player_setup_visible = True
        self._player_status_var.set("")
        self._show_player_setup()
        self._set_text(self._log_text, "")
        self._status_var.set("已创建新会话：输入需求后点击“开始生成”")

    def _open_output_dir(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(DATA_DIR))  # type: ignore[attr-defined]
        except Exception as exc:
            messagebox.showerror("无法打开目录", str(exc))

    def _open_latest_json(self) -> None:
        session = self._session or self._session_var.get().strip()
        candidates: list[Path] = []
        if session:
            cycles = DATA_DIR / "sessions" / session / "cycles"
            candidates = sorted(cycles.glob("cycle_*/output.json"))
        if not candidates:
            messagebox.showinfo("暂无输出", "当前会话还没有生成过 JSON。")
            return
        try:
            os.startfile(str(candidates[-1]))  # type: ignore[attr-defined]
        except Exception as exc:
            messagebox.showerror("无法打开文件", str(exc))

    def _on_close(self) -> None:
        if self._busy and not messagebox.askyesno(
            "退出", "任务仍在运行，确定退出吗？"
        ):
            return
        self.destroy()


def run_selftest() -> int:
    """无界面自检：验证打包后的模块、数据文件与 Mock 流程。"""
    result_path = BASE_DIR / "selftest_result.txt"
    try:
        settings = replace(
            load_app_settings(),
            data_dir=BASE_DIR / "runtime_selftest",
            memory_compress_every=2,
            review_max_rewrites=1,
            # 自检禁止调用付费生图 API，只验证 Mock 全流程
            img_backend="mock",
            img_api_key=None,
        )
        orchestrator = Orchestrator(
            llm=MockLLMClient(),
            settings=settings,
        )
        output = orchestrator.run_cycle(
            PlayerInput(text="自检：赛博朋克记忆侦探故事", session_id="selftest")
        )
        lines = [
            "MADNE selftest OK",
            f"frozen={IS_FROZEN}",
            f"base_dir={BASE_DIR}",
            f"key_configured={bool(settings.llm_api_key)}",
            f"base_url={settings.llm_base_url}",
            f"model={settings.llm_model}",
            f"choices={len(output.choices)}",
            f"scene={output.scene.scene_id}",
            f"output={orchestrator.output_path_for('selftest')}",
        ]
        result_path.write_text("\n".join(lines), encoding="utf-8")
        return 0
    except Exception as exc:
        result_path.write_text(
            f"MADNE selftest FAILED\n{type(exc).__name__}: {exc}",
            encoding="utf-8",
        )
        return 1


def run_real_selftest() -> int:
    """真实 LLM 连通性自检：只发一次最小请求，验证打包后的网络与配置。"""
    result_path = BASE_DIR / "selftest_real_result.txt"
    try:
        settings = load_app_settings()
        llm = create_llm_client(settings, mock=False)
        response = llm.complete(
            '请只输出 JSON：{"ok": true}',
            system="你是连通性测试助手，只输出合法 JSON。",
            json_schema={"type": "object"},
        )
        result_path.write_text(
            "\n".join(
                [
                    "MADNE real selftest OK",
                    f"frozen={IS_FROZEN}",
                    f"base_url={settings.llm_base_url}",
                    f"model={settings.llm_model}",
                    f"key_configured={bool(settings.llm_api_key)}",
                    f"response={response[:300]}",
                ]
            ),
            encoding="utf-8",
        )
        return 0
    except Exception as exc:
        result_path.write_text(
            f"MADNE real selftest FAILED\n{type(exc).__name__}: {exc}",
            encoding="utf-8",
        )
        return 1


def main() -> None:
    if "--selftest-real" in sys.argv:
        raise SystemExit(run_real_selftest())
    if "--selftest" in sys.argv:
        raise SystemExit(run_selftest())
    app = MadneApp()
    app.mainloop()


if __name__ == "__main__":
    main()
