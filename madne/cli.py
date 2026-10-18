"""MADNE 命令行 Demo 入口。"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import typer

from .config import Settings
from .engine.orchestrator import Orchestrator
from .llm import create_llm_client
from .models import Branch, GameOutput, PlayerInput

app = typer.Typer(
    add_completion=False,
    help="MADNE - 多 Agent 动态叙事引擎 Demo",
)


def _build_settings(data_dir: str, compress_every: int | None = None) -> Settings:
    settings = Settings.from_env()
    overrides = {"data_dir": Path(data_dir)}
    if compress_every is not None:
        overrides["memory_compress_every"] = max(1, compress_every)
    return replace(settings, **overrides)


def _pick_demo_choice(output: GameOutput) -> Branch:
    """Demo 中的自动玩家：优先选择“寻求盟友”，否则选第二个选项。"""
    for branch in output.choices:
        if "盟友" in branch.description:
            return branch
    return output.choices[min(1, len(output.choices) - 1)]


def _echo_output(turn: int, output: GameOutput, path: Path | None) -> None:
    typer.echo(f"[回合 {turn}] 场景：{output.scene.summary}")
    if path is not None:
        typer.echo(f"输出文件：{path}")
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
    print_json: bool = typer.Option(False, "--print-json", help="打印完整 GameOutput JSON"),
) -> None:
    """运行一次或连续多回合的叙事生成 Demo。"""
    settings = _build_settings(data_dir, compress_every)
    orchestrator = Orchestrator(
        llm=create_llm_client(settings, mock=mock),
        settings=settings,
    )

    current_input = input
    player_choice: str | None = None

    for turn in range(1, turns + 1):
        output = orchestrator.run_cycle(
            PlayerInput(text=current_input, session_id=session),
            player_choice=player_choice,
        )
        path = orchestrator.output_path_for(session)
        _echo_output(turn, output, path)
        if print_json:
            typer.echo(
                json.dumps(output.to_engine_dict(), ensure_ascii=False, indent=2)
            )
        if turn < turns:
            branch = _pick_demo_choice(output)
            player_choice = f"{branch.branch_id}. {branch.description}"
            current_input = f"我选择：{player_choice}。请继续推进剧情。"

    typer.echo(
        f"Session: {session} | RunLog: "
        f"{settings.data_dir / 'sessions' / session / 'runlog.jsonl'}"
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
) -> None:
    """从最近一次 ContextStore 断点继续执行。"""
    settings = _build_settings(data_dir)
    orchestrator = Orchestrator(
        llm=create_llm_client(settings, mock=mock),
        settings=settings,
    )
    output = orchestrator.resume(session)
    typer.echo(f"已从会话 {session} 的断点恢复。")
    _echo_output(0, output, orchestrator.output_path_for(session))


if __name__ == "__main__":
    app()
