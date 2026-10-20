"""MADNE 命令行 Demo 入口。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import typer

from .config import Settings
from .engine.context_store import ContextStore
from .engine.orchestrator import Orchestrator
from .engine.run_log import RunLogEntry
from .hf_env import DEFAULT_HF_MIRROR
from .image import (
    ImageRequest,
    build_negative_prompt,
    build_scene_prompt,
    create_image_client,
)
from .image.downloads import download_model, is_model_cached
from .image.prompts import select_involved_npcs
from .llm import create_llm_client
from .models import (
    Branch,
    GameOutput,
    Illustration,
    ImageCharacter,
    NPCProfile,
    PlayerInput,
)
from .storage import SessionDatabase

app = typer.Typer(
    add_completion=False,
    help="MADNE - 多 Agent 动态叙事引擎 Demo",
)
models_app = typer.Typer(
    add_completion=False,
    help="本地 SD 模型管理：预下载、缓存状态",
)
app.add_typer(models_app, name="models")


def _build_settings(
    data_dir: str,
    compress_every: int | None = None,
    prefetch: bool | None = None,
    illustrate: bool | None = None,
    img_backend: str | None = None,
) -> Settings:
    settings = Settings.from_env()
    overrides = {"data_dir": Path(data_dir)}
    if compress_every is not None:
        overrides["memory_compress_every"] = max(1, compress_every)
    if prefetch is not None:
        overrides["prefetch_choices"] = prefetch
    if illustrate is not None:
        overrides["img_enabled"] = illustrate
    if img_backend:
        overrides["img_backend"] = img_backend
    return replace(settings, **overrides)


def _display_path(path: Path, data_dir: Path) -> str:
    try:
        return path.resolve().relative_to(data_dir.resolve()).as_posix()
    except (ValueError, OSError):
        return path.as_posix()


def _echo_choice_cache(orchestrator: Orchestrator, session: str, output: GameOutput) -> None:
    if not orchestrator.settings.prefetch_choices:
        return
    cache = orchestrator.choice_cache_for(session)
    if cache is None:
        return
    previews = []
    for branch in output.choices:
        nexts = cache.next_choices(branch.branch_id)
        if nexts:
            previews.append(
                f"{branch.branch_id}→" + "/".join(
                    choice.path[-1] for choice in nexts[:3]
                )
            )
    if previews:
        typer.echo("预取下一层：" + " | ".join(previews))


def _pick_demo_choice(output: GameOutput) -> Branch:
    """Demo 中的自动玩家：优先选择“寻求盟友”，否则选第二个选项。"""
    for branch in output.choices:
        if "盟友" in branch.description:
            return branch
    return output.choices[min(1, len(output.choices) - 1)]


def _print_progress(entry: RunLogEntry) -> None:
    typer.echo(
        f"  [cycle {entry.cycle}] {entry.step_id} {entry.status} "
        f"(attempt {entry.attempt}, {entry.duration_ms / 1000:.1f}s)"
    )


def _echo_output(turn: int, output: GameOutput, path: Path | None) -> None:
    typer.echo(f"[回合 {turn}] 场景：{output.scene.summary}")
    if path is not None:
        typer.echo(f"输出文件：{path}")
    if output.illustration is not None:
        illustration = output.illustration
        typer.echo(
            f"插图：{illustration.relative_path} "
            f"（{illustration.backend} · {illustration.width}x{illustration.height}）"
        )
        for warning in illustration.warnings:
            typer.echo(f"  提示：{warning}")
    choices = " | ".join(
        f"{branch.branch_id}. {branch.description}" for branch in output.choices
    )
    typer.echo(f"玩家选项：{choices}")


@app.command()
def run(
    input: str = typer.Option(
        ...,
        "--input",
        "-i",
        help="玩家的一句话需求",
    ),
    session: str = typer.Option("demo", "--session", help="会话 ID"),
    turns: int = typer.Option(1, "--turns", min=1, help="连续生成的回合数"),
    mock: bool | None = typer.Option(
        None,
        "--mock/--real",
        help="不指定时遵循 .env / 环境变量中的 MADNE_MOCK",
    ),
    data_dir: str = typer.Option("runtime", "--data-dir", help="运行数据目录"),
    compress_every: int = typer.Option(
        10,
        "--compress-every",
        min=1,
        help="每 N 个记忆事件触发压缩",
    ),
    prefetch: bool | None = typer.Option(
        None,
        "--prefetch/--no-prefetch",
        help="是否预取未来两层选项；不指定时遵循 .env",
    ),
    illustrate: bool | None = typer.Option(
        None,
        "--illustrate/--no-illustrate",
        help="是否生成场景插图；不指定时遵循 IMG_ENABLED",
    ),
    img_backend: str | None = typer.Option(
        None,
        "--img-backend",
        help="临时覆盖图像后端：mock / ark / seedream / comfyui / diffusers",
    ),
    print_json: bool = typer.Option(False, "--print-json", help="打印完整 GameOutput JSON"),
) -> None:
    """运行一次或连续多回合的叙事生成 Demo。"""
    settings = _build_settings(
        data_dir,
        compress_every,
        prefetch,
        illustrate=illustrate,
        img_backend=img_backend,
    )
    orchestrator = Orchestrator(
        llm=create_llm_client(settings, mock=mock),
        settings=settings,
        on_event=_print_progress,
    )

    current_input = input
    player_choice: str | None = None
    player_branch_id: str | None = None

    for turn in range(1, turns + 1):
        output = orchestrator.run_cycle(
            PlayerInput(text=current_input, session_id=session),
            player_choice=player_choice,
            player_branch_id=player_branch_id,
        )
        path = orchestrator.output_path_for(session)
        _echo_output(turn, output, path)
        _echo_choice_cache(orchestrator, session, output)
        if print_json:
            typer.echo(
                json.dumps(output.to_engine_dict(), ensure_ascii=False, indent=2)
            )
        if turn < turns:
            branch = _pick_demo_choice(output)
            player_branch_id = branch.branch_id
            player_choice = f"{branch.branch_id}. {branch.description}"
            current_input = f"我选择：{player_choice}。请继续推进剧情。"

    typer.echo(
        f"Session: {session} | Database: "
        f"{settings.db_path or settings.data_dir / 'madne.db'}"
    )


@app.command()
def resume(
    session: str = typer.Option(..., "--session", help="需要恢复的会话 ID"),
    mock: bool | None = typer.Option(
        None,
        "--mock/--real",
        help="不指定时遵循 .env / 环境变量中的 MADNE_MOCK",
    ),
    data_dir: str = typer.Option("runtime", "--data-dir", help="运行数据目录"),
    prefetch: bool | None = typer.Option(
        None,
        "--prefetch/--no-prefetch",
        help="是否预取未来两层选项；不指定时遵循 .env",
    ),
    illustrate: bool | None = typer.Option(
        None,
        "--illustrate/--no-illustrate",
        help="是否生成场景插图；不指定时遵循 IMG_ENABLED",
    ),
    img_backend: str | None = typer.Option(
        None,
        "--img-backend",
        help="临时覆盖图像后端：mock / ark / seedream / comfyui / diffusers",
    ),
) -> None:
    """从最近一次 ContextStore 断点继续执行。"""
    settings = _build_settings(
        data_dir,
        prefetch=prefetch,
        illustrate=illustrate,
        img_backend=img_backend,
    )
    orchestrator = Orchestrator(
        llm=create_llm_client(settings, mock=mock),
        on_event=_print_progress,
        settings=settings,
    )
    output = orchestrator.resume(session)
    typer.echo(f"已从会话 {session} 的断点恢复。")
    _echo_output(0, output, orchestrator.output_path_for(session))


@app.command()
def image(
    session: str = typer.Option(..., "--session", help="需要重绘插图的会话 ID"),
    data_dir: str = typer.Option("runtime", "--data-dir", help="运行数据目录"),
    img_backend: str | None = typer.Option(
        None,
        "--img-backend",
        help="临时覆盖图像后端：mock / ark / seedream / comfyui / diffusers",
    ),
) -> None:
    """为会话最近一回合重新生成插图（不调用叙事 LLM）。"""
    settings = _build_settings(data_dir, img_backend=img_backend)
    session_dir = settings.data_dir / "sessions" / session
    database = SessionDatabase(
        settings.db_path or settings.data_dir / "madne.db"
    )
    database.import_legacy_session(session, session_dir)
    ctx: ContextStore | None = None
    for cycle in reversed(database.context_cycles(session)):
        snapshot = database.load_context(session, cycle)
        if snapshot is None:
            continue
        candidate = ContextStore.from_snapshot(snapshot)
        if candidate.has("scene") and candidate.has("world"):
            ctx = candidate
            break
    if ctx is None:
        typer.echo("上下文中缺少 scene/world，无法生成插图")
        raise typer.Exit(code=1)

    scene = ctx.require("scene")
    world = ctx.require("world")
    raw_npcs = ctx.get("npcs")
    npcs = (
        [item for item in raw_npcs if isinstance(item, NPCProfile)]
        if isinstance(raw_npcs, list)
        else []
    )
    prompt = build_scene_prompt(scene, world, npcs, settings.img_style)
    prompt_limit = (
        256
        if settings.img_backend
        in {"seedream", "hunyuan", "openai", "openai_compat"}
        else None
    )
    prompt = build_scene_prompt(
        scene,
        world,
        npcs,
        settings.img_style,
        max_chars=prompt_limit,
    )
    seed = settings.img_seed
    if seed is None:
        seed = int(
            hashlib.sha256(
                f"{session}|madne-image".encode("utf-8")
            ).hexdigest()[:8],
            16,
        )
    involved = select_involved_npcs(scene, npcs)
    references: list[Path] = []
    for npc in involved:
        raw = (npc.reference_image or "").strip()
        if not raw:
            continue
        reference = Path(raw)
        if not reference.is_absolute():
            reference = settings.data_dir / reference
        if reference.exists():
            references.append(reference)

    mode = settings.img_consistency_mode
    if mode not in {"self_attn", "id_injection"}:
        mode = "self_attn"
    if mode == "id_injection" and not references:
        mode = "self_attn"

    target = session_dir / "assets" / f"cycle_{ctx.cycle:03d}.png"
    client = create_image_client(settings)
    result = client.generate(
        ImageRequest(
            prompt=prompt,
            negative_prompt=build_negative_prompt(max_chars=prompt_limit),
            output_path=target,
            width=settings.img_width,
            height=settings.img_height,
            seed=seed,
            steps=settings.img_steps,
            guidance_scale=settings.img_guidance,
            reference_images=references,
            consistency_mode=mode,
        )
    )
    if settings.img_consistency_mode == "id_injection" and not references:
        result.warnings.insert(
            0,
            "未找到角色定妆照，id_injection 已回退 self_attn",
        )

    relative_path = (
        f"sessions/{session}/assets/cycle_{ctx.cycle:03d}.png"
    )
    illustration = Illustration(
        scene_id=scene.scene_id,
        backend=result.backend,
        prompt=result.prompt,
        negative_prompt=result.negative_prompt,
        seed=result.seed,
        width=result.width,
        height=result.height,
        consistency_mode=mode,
        relative_path=relative_path,
        url=f"/assets/{session}/cycle_{ctx.cycle:03d}.png",
        characters=[
            ImageCharacter(
                npc_id=npc.npc_id,
                name=npc.name,
                appearance=npc.appearance,
            )
            for npc in involved
        ],
        reference_images=[
            _display_path(path, settings.data_dir) for path in references
        ],
        warnings=list(result.warnings),
    )
    output_path = (
        session_dir
        / "cycles"
        / f"cycle_{ctx.cycle:03d}"
        / "output.json"
    )
    payload = database.load_output(session, ctx.cycle)
    if payload is not None:
        payload["illustration"] = illustration.model_dump(mode="json")
        database.save_output(session, ctx.cycle, payload)
        # 同步导出文件（若仍存在），保持与旧版可视化路径兼容。
        if output_path.exists():
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    typer.echo(
        f"已生成插图：{target}（{result.backend} · "
        f"{result.width}x{result.height}）"
    )
    for warning in result.warnings:
        typer.echo(f"提示：{warning}")


@app.command()
def health(
    data_dir: str = typer.Option("runtime", "--data-dir", help="运行数据目录"),
    img_backend: str | None = typer.Option(
        None,
        "--img-backend",
        help="临时覆盖图像后端：mock / ark / seedream / comfyui / diffusers",
    ),
) -> None:
    """检查 LLM / 图像后端 / 存储的配置状态。"""
    settings = _build_settings(data_dir, img_backend=img_backend)
    client = create_image_client(settings)
    payload = {
        "llm": {
            "mock": settings.use_mock,
            "configured": bool(settings.llm_api_key) and not settings.use_mock,
            "base_url": settings.llm_base_url,
            "model": settings.llm_model,
        },
        "image": {
            "enabled": settings.img_enabled,
            "timeout": settings.img_timeout,
            **client.health(),
        },
        "storage": {
            "backend": "sqlite",
            "data_dir": str(settings.data_dir),
            "database": str(
                settings.db_path or settings.data_dir / "madne.db"
            ),
            "exists": settings.data_dir.exists(),
            "sessions": len(
                SessionDatabase(
                    settings.db_path or settings.data_dir / "madne.db"
                ).list_sessions()
            ),
        },
    }
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@models_app.command("download")
def models_download(
    model: str | None = typer.Option(
        None, "--model", help="Hugging Face 仓库名（默认 IMG_MODEL）"
    ),
    cache_dir: str | None = typer.Option(
        None, "--cache-dir", help="权重缓存目录（默认 MODEL_CACHE_DIR）"
    ),
    mirror: bool = typer.Option(
        False, "--mirror", help="强制使用 hf-mirror.com"
    ),
    endpoint: str | None = typer.Option(
        None, "--endpoint", help="自定义 Hugging Face 端点 URL"
    ),
) -> None:
    """预下载本地 SD 权重（只拉必要文件，支持断点续传与镜像回退）。"""
    settings = Settings.from_env()
    target_model = (model or settings.img_model).strip()
    target_dir = Path(cache_dir) if cache_dir else settings.model_cache_dir
    if mirror:
        chosen = DEFAULT_HF_MIRROR
    elif endpoint:
        chosen = endpoint
    else:
        chosen = settings.hf_endpoint

    typer.echo(f"模型：{target_model}")
    typer.echo(f"目录：{Path(target_dir).resolve()}")
    typer.echo(
        "端点："
        + (
            chosen
            if chosen
            else "huggingface.co（网络失败自动回退 hf-mirror.com）"
        )
    )
    try:
        result = download_model(
            target_model,
            target_dir,
            endpoint=chosen or None,
            log=typer.echo,
        )
    except Exception as exc:  # noqa: BLE001 - CLI 统一友好输出
        typer.echo(f"下载失败：{exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"完成：{result.summary()}")


@models_app.command("status")
def models_status(
    model: str | None = typer.Option(
        None, "--model", help="Hugging Face 仓库名（默认 IMG_MODEL）"
    ),
    cache_dir: str | None = typer.Option(
        None, "--cache-dir", help="权重缓存目录（默认 MODEL_CACHE_DIR）"
    ),
) -> None:
    """检查本地模型缓存（不联网）。"""
    settings = Settings.from_env()
    target_model = (model or settings.img_model).strip()
    target_dir = Path(cache_dir) if cache_dir else settings.model_cache_dir
    cached = is_model_cached(target_model, target_dir)
    typer.echo(
        json.dumps(
            {
                "model": target_model,
                "cache_dir": str(Path(target_dir).resolve()),
                "cached": cached,
                "hf_endpoint": settings.hf_endpoint or "https://huggingface.co",
                "note": (
                    "已缓存，可直接生图。"
                    if cached
                    else "未缓存：运行 madne models download 预下载。"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
