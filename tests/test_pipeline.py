import json
from pathlib import Path

from madne.config import Settings
from madne.engine.orchestrator import Orchestrator
from madne.llm import MockLLMClient
from madne.models import GameOutput, PlayerInput


CYBERPUNK_INPUT = "我想要一个赛博朋克世界里，关于记忆买卖的侦探故事"


def make_settings(tmp_path: Path, compress_every: int = 2) -> Settings:
    return Settings(
        use_mock=True,
        data_dir=tmp_path,
        memory_compress_every=compress_every,
        review_max_rewrites=2,
    )


def read_runlog(tmp_path: Path, session: str) -> list[dict]:
    path = tmp_path / "sessions" / session / "runlog.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_mock_pipeline_single_cycle(tmp_path: Path):
    llm = MockLLMClient()
    orchestrator = Orchestrator(llm=llm, settings=make_settings(tmp_path))

    output = orchestrator.run_cycle(
        PlayerInput(text=CYBERPUNK_INPUT, session_id="demo")
    )

    assert isinstance(output, GameOutput)
    assert len(output.choices) == 3
    assert len(output.world_hash) == 64
    assert output.scene.scene_id == "scene_001"
    assert output.script.scene_id == "scene_001"
    assert output.script.lines

    output_path = orchestrator.output_path_for("demo")
    assert output_path is not None and output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(payload["choices"]) == 3

    for agent in (
        "intent",
        "planner",
        "world",
        "framework",
        "filler",
        "script",
        "guardian_world",
        "guardian_framework",
        "guardian_scene",
    ):
        assert llm.call_count(agent) > 0, f"{agent} 未被调用"

    entries = read_runlog(tmp_path, "demo")
    assert any(entry["status"] == "success" for entry in entries)
    assert {entry["agent"] for entry in entries} >= {
        "world",
        "framework",
        "npc",
        "filler",
        "script",
    }


def test_multi_turn_continuation_and_memory_injection(tmp_path: Path):
    llm = MockLLMClient()
    orchestrator = Orchestrator(
        llm=llm,
        settings=make_settings(tmp_path, compress_every=2),
    )

    first = orchestrator.run_cycle(
        PlayerInput(text=CYBERPUNK_INPUT, session_id="multi")
    )
    assert first.choices

    choice = "B. 寻求盟友阿凯的帮助，交换情报并结成临时同盟"
    orchestrator.run_cycle(
        PlayerInput(text=f"我选择：{choice}", session_id="multi"),
        player_choice=choice,
    )
    third = orchestrator.run_cycle(
        PlayerInput(text=f"我选择：{choice}", session_id="multi"),
        player_choice=choice,
    )

    compressed_path = (
        tmp_path / "sessions" / "multi" / "memory" / "compressed.json"
    )
    assert compressed_path.exists()
    compressed = json.loads(compressed_path.read_text(encoding="utf-8"))
    assert any("盟友" in item for item in compressed["player_preferences"])

    planner_calls = llm.calls_for("planner")
    assert len(planner_calls) == 3
    assert "偏好通过寻求盟友推进调查" in planner_calls[-1].prompt

    entries = read_runlog(tmp_path, "multi")
    assert any(entry["agent"] == "npc_update" for entry in entries)

    npc_payload = json.loads(
        (tmp_path / "sessions" / "multi" / "npcs.json").read_text(
            encoding="utf-8"
        )
    )
    kai = next(item for item in npc_payload if item["npc_id"] == "npc_kai")
    assert kai["traits"]["loyalty"] > 85

    assert third.script.lines


def test_guardian_rejection_triggers_rewrite(tmp_path: Path):
    llm = MockLLMClient()
    bad_script = json.dumps(
        {
            "scene_id": "scene_001",
            "lines": [
                {
                    "speaker": "苏离",
                    "text": "我会无条件把一切都告诉你。",
                    "emotion": "坦诚",
                }
            ],
        },
        ensure_ascii=False,
    )
    # 第一次台本生成使用违规内容，重写时回退到默认合格台本。
    llm.push_responses("script", bad_script, front=True)

    passed = json.dumps(
        {"passed": True, "issues": [], "suggested_fix": None},
        ensure_ascii=False,
    )
    failed = json.dumps(
        {
            "passed": False,
            "issues": [
                {
                    "category": "character",
                    "detail": "苏离的台词违背其高 selfishness 权重",
                    "severity": "high",
                }
            ],
            "suggested_fix": "重写苏离台词，体现其自私与算计",
        },
        ensure_ascii=False,
    )
    # 第一轮四维审查：前三维通过，角色维度驳回；第二轮全部通过。
    llm.push_responses(
        "guardian_scene",
        passed,
        passed,
        passed,
        failed,
        passed,
        passed,
        passed,
        passed,
        front=True,
    )

    orchestrator = Orchestrator(
        llm=llm,
        settings=make_settings(tmp_path, compress_every=50),
    )
    output = orchestrator.run_cycle(
        PlayerInput(text=CYBERPUNK_INPUT, session_id="guardian")
    )

    assert llm.call_count("script") == 2
    entries = read_runlog(tmp_path, "guardian")
    assert any(entry["status"] == "review_rejected" for entry in entries)
    assert len(output.script.lines) == 4

    rewrite_prompt = llm.calls_for("script")[1].prompt
    assert "重写苏离台词" in rewrite_prompt


def test_resume_from_partial_context(tmp_path: Path):
    session = "resume"
    first_orchestrator = Orchestrator(
        llm=MockLLMClient(),
        settings=make_settings(tmp_path, compress_every=50),
    )
    first_orchestrator.run_cycle(
        PlayerInput(text=CYBERPUNK_INPUT, session_id=session)
    )

    cycle_dir = tmp_path / "sessions" / session / "cycles" / "cycle_001"
    context_path = cycle_dir / "context.json"
    payload = json.loads(context_path.read_text(encoding="utf-8"))
    # 模拟在 World / Framework 完成后、NPC 初始化之前中断。
    payload["completed_steps"] = ["01_world", "02_framework"]
    for key in ("npcs", "scene", "script", "game_output"):
        payload["data"].pop(key, None)
    context_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    output_path = cycle_dir / "output.json"
    output_path.unlink(missing_ok=True)

    resumed_orchestrator = Orchestrator(
        llm=MockLLMClient(),
        settings=make_settings(tmp_path, compress_every=50),
    )
    output = resumed_orchestrator.resume(session)

    assert isinstance(output, GameOutput)
    assert output_path.exists()

    entries = read_runlog(tmp_path, session)
    filler_success = [
        entry
        for entry in entries
        if entry["step_id"] == "04_filler" and entry["status"] == "success"
    ]
    framework_success = [
        entry
        for entry in entries
        if entry["step_id"] == "02_framework" and entry["status"] == "success"
    ]
    assert len(filler_success) == 2, "恢复后应重新执行未完成的 Filler"
    assert len(framework_success) == 1, "已完成的 Framework 不应重复执行"
