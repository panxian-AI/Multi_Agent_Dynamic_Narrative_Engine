"""本地人物一致性度量：CLIP-I 与 DINO（按需懒加载，缺失依赖时跳过）。"""

from __future__ import annotations

from pathlib import Path

from ..models import ImageMetrics

_CLIP: tuple[object, object, object] | None = None
_DINO: tuple[object, object] | None = None


def _device(torch_module: object) -> str:
    if torch_module.cuda.is_available():  # type: ignore[attr-defined]
        return "cuda"
    return "cpu"


def _load_clip() -> tuple[object, object, object]:
    global _CLIP
    if _CLIP is not None:
        return _CLIP
    import open_clip
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    device = _device(torch)
    model = model.to(device).eval()
    _CLIP = (model, preprocess, torch)
    return _CLIP


def _load_dino() -> tuple[object, object]:
    global _DINO
    if _DINO is not None:
        return _DINO
    import torch

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    device = _device(torch)
    model = model.to(device).eval()
    _DINO = (model, torch)
    return _DINO


def _normalize(tensor: object) -> object:
    return tensor / tensor.norm(dim=-1, keepdim=True)  # type: ignore[attr-defined]


def _cosine(left: object, right: object) -> float:
    value = float((_normalize(left) * _normalize(right)).sum(dim=-1).item())
    return max(0.0, min(1.0, value))


def _clip_embedding(path: Path) -> object:
    model, preprocess, torch = _load_clip()
    from PIL import Image

    image = Image.open(path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(_device(torch))
    with torch.no_grad():
        return _normalize(model.encode_image(tensor)).float().cpu()


def _dino_embedding(path: Path) -> object:
    model, torch = _load_dino()
    import numpy as np
    from PIL import Image

    image = Image.open(path).convert("RGB").resize((224, 224))
    array = np.asarray(image).astype("float32") / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = ((tensor - mean) / std).unsqueeze(0).to(_device(torch))
    with torch.no_grad():
        return model(tensor).float().cpu()


def compute_metrics(
    generated: Path,
    references: list[Path],
) -> tuple[ImageMetrics | None, list[str]]:
    """计算生成图与角色定妆照的 CLIP-I / DINO 余弦相似度。

    任一依赖缺失或计算失败时返回 ``None`` 与告警，不影响主流程。
    """
    valid = [path for path in references if Path(path).exists()]
    if not valid:
        return None, []

    try:
        import torch  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError as exc:
        return None, [f"一致性指标不可用：缺少依赖 {exc.name}"]

    warnings: list[str] = []
    clip_i: float | None = None
    dino: float | None = None

    try:
        generated_clip = _clip_embedding(generated)
        scores = [
            _cosine(generated_clip, _clip_embedding(path)) for path in valid
        ]
        clip_i = sum(scores) / len(scores)
    except Exception as exc:
        warnings.append(f"CLIP-I 计算失败：{exc}")

    try:
        generated_dino = _dino_embedding(generated)
        scores = [
            _cosine(generated_dino, _dino_embedding(path)) for path in valid
        ]
        dino = sum(scores) / len(scores)
    except Exception as exc:
        warnings.append(f"DINO 计算失败：{exc}")

    if clip_i is None and dino is None:
        return None, warnings
    return (
        ImageMetrics(
            clip_i=clip_i,
            dino=dino,
            reference_count=len(valid),
            warnings=list(warnings),
        ),
        warnings,
    )
