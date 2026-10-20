"""Hugging Face 模型预下载：精简文件、落盘进度、镜像自动回退。

默认模型仓库同时包含 fp16 / fp32 / ``.bin`` / ``.onnx`` 等多份权重，
``from_pretrained`` 一次拉全量往往 5GB 以上。这里先读取仓库文件列表，
每个组件只挑一份最合适的权重（优先 ``*.fp16.safetensors``），
把实际下载量压到最小，并在网络失败时自动切换到 ``hf-mirror.com``。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..exceptions import ImageGenError
from ..hf_env import (
    DEFAULT_HF_MIRROR,
    configure_hf_env,
    endpoint_candidates,
    resolve_endpoint,
)

LogFn = Callable[[str], None]

#: 权重文件后缀：每个组件目录只保留其中一个
_WEIGHT_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".ckpt",
    ".onnx",
    ".msgpack",
    ".h5",
    ".gguf",
    ".pt",
    ".pth",
)
#: 与加载无关的文件，避免占用带宽
_SKIP_SUFFIXES = (
    ".md",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".webp",
    ".yaml",
    ".yml",
    ".gitattributes",
)


@dataclass(slots=True)
class ModelDownload:
    """预下载结果。"""

    path: Path
    endpoint: str
    file_count: int
    total_bytes: int

    def summary(self) -> str:
        if not self.file_count and not self.total_bytes:
            return f"{self.path}（已缓存）"
        size = (
            f"{self.total_bytes / 1024 ** 3:.2f} GB"
            if self.total_bytes
            else "大小未知"
        )
        return f"{self.path}（{self.file_count} 个文件，{size}，来源 {self.endpoint}）"


@dataclass(slots=True)
class DownloadPlan:
    """仓库下载计划。"""

    allow_patterns: list[str] | None
    file_count: int
    total_bytes: int
    fp16: bool
    has_model_index: bool


# --------------------------------------------------------------------------- 计划
def _is_weight(name: str) -> bool:
    return name.lower().endswith(_WEIGHT_SUFFIXES)


def _is_skipped(name: str) -> bool:
    lower = name.lower()
    return lower.endswith(_SKIP_SUFFIXES) or lower.startswith(".git")


def _pick_weight(candidates: list[str]) -> str | None:
    """同一组件目录内只挑一份权重：fp16 safetensors > safetensors > bin。"""
    ordered = sorted(candidates, key=lambda item: item.lower())
    for suffix in (".fp16.safetensors", ".safetensors", ".bin"):
        for name in ordered:
            if name.lower().endswith(suffix):
                return name
    return ordered[0] if ordered else None


def select_files(repo_files: list[str]) -> list[str] | None:
    """从仓库文件列表生成最小下载集合。

    返回 ``None`` 表示无法安全裁剪（交给 hub 全量下载）。
    """
    files = [name for name in repo_files if name]
    if not files:
        return None

    has_model_index = "model_index.json" in files
    selected = [
        name
        for name in files
        if not _is_weight(name) and not _is_skipped(name)
    ]

    grouped: dict[str, list[str]] = {}
    for name in files:
        # 只取组件目录（unet/、vae/、text_encoder/ …）内的权重；
        # 仓库根目录的单文件 checkpoint 动辄 7GB，diffusers 加载并不需要。
        if "/" not in name or not _is_weight(name) or _is_skipped(name):
            continue
        folder = name.rsplit("/", 1)[0]
        grouped.setdefault(folder, []).append(name)

    for candidates in grouped.values():
        chosen = _pick_weight(candidates)
        if chosen is not None:
            selected.append(chosen)

    fp16 = any(name.lower().endswith(".fp16.safetensors") for name in selected)
    # 没有 model_index.json 说明不是 diffusers 目录结构：不做裁剪。
    if not has_model_index:
        return None
    return sorted(set(selected))


def _list_repo_files(model: str, endpoint: str) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi(endpoint=endpoint) if endpoint else HfApi()
    return list(api.list_repo_files(model))


def _measure_files(model: str, endpoint: str, names: list[str]) -> int:
    """查询选中文件的总字节数；失败时返回 0（进度退化为「已下载 xx MB」）。"""
    try:
        from huggingface_hub import HfApi

        api = HfApi(endpoint=endpoint) if endpoint else HfApi()
        infos: Any = api.get_paths_info(model, paths=names)
        return sum(int(getattr(info, "size", 0) or 0) for info in infos)
    except Exception:
        return 0


def plan_download(model: str, endpoint: str) -> DownloadPlan:
    """读取远端文件列表，得到最小下载集合与总大小。"""
    files = _list_repo_files(model, endpoint)
    patterns = select_files(files)
    if patterns is None:
        file_count = len(files)
        total = _measure_files(model, endpoint, files) if files else 0
        return DownloadPlan(None, file_count, total, False, False)
    total = _measure_files(model, endpoint, patterns)
    return DownloadPlan(
        allow_patterns=patterns,
        file_count=len(patterns),
        total_bytes=total,
        fp16=any(name.lower().endswith(".fp16.safetensors") for name in patterns),
        has_model_index="model_index.json" in patterns,
    )


# --------------------------------------------------------------------------- 进度
class _DiskProgress:
    """按缓存目录实际落盘字节报告进度。

    不用 hub 的 tqdm 回调：快照下载会同时更新 transfer/reconstruct 两个
    进度条，重试时还会重复计数，导致「显示 80% 但磁盘只有 10%」。
    """

    def __init__(
        self,
        model: str,
        cache_dir: Path,
        total_bytes: int,
        log: LogFn | None,
        interval: float = 2.0,
    ) -> None:
        self.model = model
        self.folder = Path(cache_dir) / (
            "models--" + model.replace("/", "--")
        )
        self.total_bytes = max(0, total_bytes)
        self.log = log
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_percent = -1
        self._bytes = 0

    def start(self, file_count: int) -> None:
        self._emit(
            f"开始下载 {self.model}（{file_count} 个文件，"
            f"{self._size_text(self.total_bytes)}）；支持断点续传"
        )
        existing = self._disk_bytes()
        if existing > 0:
            self._bytes = existing
            self._emit(
                f"检测到已有缓存 {self._size_text(existing)}，从断点继续"
            )
            self._report()
        self._thread = threading.Thread(
            target=self._run,
            name="madne-model-progress",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._report()

    def _report(self) -> None:
        size = max(self._bytes, self._disk_bytes())
        self._bytes = size
        if self.total_bytes <= 0:
            self._emit(f"正在下载模型权重：已写入 {self._size_text(size)}")
            return
        # 留 1% 给最后的落盘/校验，避免提前显示 100%。
        percent = min(99, int(size * 100 / self.total_bytes))
        if percent <= self._last_percent:
            return
        self._last_percent = percent
        self._emit(
            f"正在下载模型权重：{percent}%"
            f"（{self._size_text(size)} / {self._size_text(self.total_bytes)}）"
        )

    def _disk_bytes(self) -> int:
        try:
            # 只统计 blobs：snapshots 里是链接，重复统计会成倍虚高。
            blobs = self.folder / "blobs"
            if not blobs.is_dir():
                return 0
            total = 0
            # 同一 blob 的每次重试都会新建一个 .incomplete；只算最新写入的
            # 那个（旧尝试不会被 hub 续传），否则进度会长期卡在高位。
            partials: dict[str, tuple[float, int]] = {}
            for path in blobs.rglob("*"):
                if not path.is_file():
                    continue
                stat = path.stat()
                if path.name.endswith(".incomplete"):
                    key = path.name.split(".", 1)[0]
                    current = partials.get(key)
                    if current is None or stat.st_mtime >= current[0]:
                        partials[key] = (stat.st_mtime, stat.st_size)
                else:
                    total += stat.st_size
            return total + sum(size for _, size in partials.values())
        except OSError:
            return 0

    def finish(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._bytes = max(self._bytes, self._disk_bytes())
        if self.total_bytes:
            self._emit(
                f"模型下载完成：{self._size_text(self._bytes)} / "
                f"{self._size_text(self.total_bytes)}"
            )
        else:
            self._emit(f"模型下载完成：{self._size_text(self._bytes)}")

    @staticmethod
    def _size_text(num_bytes: int) -> str:
        if num_bytes <= 0:
            return "大小未知"
        if num_bytes >= 1024 ** 3:
            return f"{num_bytes / 1024 ** 3:.2f} GB"
        return f"{num_bytes / 1024 ** 2:.0f} MB"

    def _emit(self, message: str) -> None:
        _say(self.log, message)


def _silent_tqdm() -> type:
    """让 hub 不要往 stderr 画进度条（GUI/exe 无控制台）。"""
    from tqdm.auto import tqdm

    class _SilentTqdm(tqdm):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.pop("name", None)
            kwargs.pop("log_level", None)
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

    return _SilentTqdm


# --------------------------------------------------------------------------- 下载
def _short_error(exc: Exception) -> str:
    message = str(exc).strip().replace("\n", " ")
    return message[:200] if message else type(exc).__name__


def _is_auth_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in ("401", "403", "gated", "access to model", "unauthorized")
    )


def _is_network_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "connection",
            "timed out",
            "timeout",
            "max retries",
            "temporary failure",
            "ssl",
            "proxy",
            "read timed out",
            "cannot connect",
            "getaddrinfo",
            "name or service not known",
            "502",
            "503",
            "504",
            "server error",
        )
    ) or isinstance(exc, (TimeoutError, ConnectionError, OSError))


def _friendly_error(model: str, exc: Exception, tried: list[str]) -> ImageGenError:
    if _is_auth_error(exc):
        return ImageGenError(
            f"下载 {model} 被拒绝（鉴权/受限仓库）：{_short_error(exc)}。"
            "若该仓库需要接受许可，请先在 Hugging Face 页面同意条款并设置 "
            "HF_TOKEN；也可以换用不需要授权的轻量模型 stabilityai/sd-turbo。"
        )
    if _is_network_error(exc):
        return ImageGenError(
            f"下载 {model} 网络失败：{_short_error(exc)}。"
            f"已尝试：{', '.join(tried)}。请检查网络/代理，或设置 "
            "HF_ENDPOINT=https://hf-mirror.com、HF_HUB_DISABLE_XET=1 后重试，"
            "也可以手动把权重放入 MODEL_CACHE_DIR。"
        )
    return ImageGenError(f"下载 {model} 失败：{_short_error(exc)}")


def _snapshot(
    model: str,
    cache_dir: Path,
    plan: DownloadPlan,
    retries: int,
    max_workers: int,
) -> Path:
    from huggingface_hub import snapshot_download

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            kwargs: dict[str, Any] = {
                "repo_id": model,
                "cache_dir": str(cache_dir),
                "max_workers": max_workers,
                "tqdm_class": _silent_tqdm(),
            }
            if plan.allow_patterns is not None:
                kwargs["allow_patterns"] = plan.allow_patterns
            return Path(snapshot_download(**kwargs))
        except Exception as exc:  # noqa: BLE001 - 统一转为友好错误
            last_error = exc
            if attempt < retries:
                time.sleep(min(4.0, 0.8 * attempt))
    assert last_error is not None
    raise last_error


def _clean_stale_incomplete(
    cache_dir: Path, model: str, log: LogFn | None, min_age: float = 900.0
) -> None:
    """清理上次中断遗留的临时分片（hub 不会跨进程续传这些文件）。"""
    folder = cache_dir / ("models--" + model.replace("/", "--"))
    if not folder.is_dir():
        return
    cutoff = time.time() - min_age
    removed = 0
    freed = 0
    try:
        for path in folder.rglob("*.incomplete"):
            try:
                stat = path.stat()
                if stat.st_mtime > cutoff:
                    continue
                path.unlink()
                removed += 1
                freed += stat.st_size
            except OSError:
                continue
    except OSError:
        return
    if removed:
        _say(
            log,
            f"已清理 {removed} 个过期临时分片"
            f"（释放 {freed / 1024 ** 2:.0f} MB）",
        )


def download_model(
    model: str,
    cache_dir: str | Path,
    *,
    endpoint: str | None = None,
    log: LogFn | None = None,
    retries: int = 2,
    max_workers: int = 4,
) -> ModelDownload:
    """确保模型完整落到 ``cache_dir``，返回本地快照路径。

    网络错误时按 ``endpoint_candidates`` 依次尝试（官方 → hf-mirror）。
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    _clean_stale_incomplete(cache_path, model, log)

    cached = _cached_snapshot(model, cache_path)
    if cached is not None:
        _say(log, f"{model} 已缓存：{cached}")
        return ModelDownload(
            path=cached,
            endpoint=resolve_endpoint(endpoint),
            file_count=0,
            total_bytes=0,
        )

    candidates = endpoint_candidates(endpoint)
    tried: list[str] = []
    last_error: Exception | None = None

    for candidate in candidates:
        configure_hf_env(candidate)
        tried.append(candidate)
        try:
            plan = plan_download(model, candidate)
        except Exception as exc:  # noqa: BLE001 - 列目录失败可能是网络/鉴权
            last_error = exc
            if candidate != candidates[-1] and _is_network_error(exc):
                _say(
                    log,
                    f"无法从 {candidate} 获取文件列表（{_short_error(exc)}），"
                    f"切换到镜像 {DEFAULT_HF_MIRROR} …",
                )
                continue
            raise _friendly_error(model, exc, tried) from exc

        reporter = _DiskProgress(
            model, cache_path, plan.total_bytes, log
        )
        reporter.start(plan.file_count)
        try:
            path = _snapshot(
                model, cache_path, plan, retries, max_workers
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if candidate != candidates[-1] and _is_network_error(exc):
                _say(
                    log,
                    f"从 {candidate} 下载失败（{_short_error(exc)}），"
                    f"改用镜像 {DEFAULT_HF_MIRROR} 重试 …",
                )
                continue
            raise _friendly_error(model, exc, tried) from exc

        reporter.finish()
        return ModelDownload(
            path=path,
            endpoint=candidate,
            file_count=plan.file_count,
            total_bytes=plan.total_bytes,
        )

    assert last_error is not None
    raise _friendly_error(model, last_error, tried)


def _cached_snapshot(model: str, cache_dir: str | Path) -> Path | None:
    """返回可直接加载的本地快照目录；不联网。"""
    folder = Path(cache_dir) / ("models--" + model.replace("/", "--"))
    snapshots = folder / "snapshots"
    if not snapshots.is_dir():
        return None
    try:
        candidates = [path for path in snapshots.iterdir() if path.is_dir()]
    except OSError:
        return None
    for snapshot in candidates:
        if _snapshot_ready(snapshot):
            return snapshot
    return None


def _snapshot_ready(snapshot: Path) -> bool:
    """检查 diffusers 目录结构是否完整（只读本地文件）。"""
    index = snapshot / "model_index.json"
    if not index.is_file():
        return False
    try:
        import json

        data = json.loads(index.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    for component, value in data.items():
        # model_index.json 里 _class_name / _diffusers_version 是字符串；
        # 可选组件（如 safety_checker=None）形如 [null, null]。
        if not isinstance(value, list) or not value or value[0] is None:
            continue
        folder = snapshot / component
        if not folder.is_dir():
            return False
        # scheduler / tokenizer / feature_extractor 只有配置没有权重。
        if not (folder / "config.json").is_file():
            continue
        if not any(
            path.suffix.lower()
            in {".safetensors", ".bin", ".ckpt", ".pt", ".pth"}
            for path in folder.rglob("*")
            if path.is_file()
        ):
            return False
    return True


def is_model_cached(model: str, cache_dir: str | Path) -> bool:
    """只查本地缓存，不触发网络。"""
    return _cached_snapshot(model, cache_dir) is not None


def _say(log: LogFn | None, message: str) -> None:
    if log is None:
        return
    try:
        log(message)
    except Exception:
        pass
