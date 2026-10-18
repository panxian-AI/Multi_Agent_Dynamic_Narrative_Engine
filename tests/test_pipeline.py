import json
from pathlib import Path

from madne.config import Settings
from madne.engine.context_store import ContextStore
from madne.engine.orchestrator import Orchestrator
from madne.llm import MockLLMClient
from madne.models import ChoiceCache, GameOutput, PlayerInput, StoryFramework


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
    assert llm.call_count("npc") == 1
    assert llm.call_count("choice_cache") == 1
    assert orchestrator.choice_cache_for("demo") is not None


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

    first_choice = next(
        branch for branch in first.choices if branch.branch_id == "B"
    )
    first_text = f"{first_choice.branch_id}. {first_choice.description}"
    second = orchestrator.run_cycle(
        PlayerInput(text=f"我选择：{first_text}", session_id="multi"),
        player_choice=first_text,
        player_branch_id=first_choice.branch_id,
    )
    assert [branch.branch_id for branch in second.choices] == [
        "B1",
        "B2",
        "B3",
    ]

    second_choice = second.choices[0]
    second_text = f"{second_choice.branch_id}. {second_choice.description}"
    third = orchestrator.run_cycle(
        PlayerInput(text=f"我选择：{second_text}", session_id="multi"),
        player_choice=second_text,
        player_branch_id=second_choice.branch_id,
    )
    assert [branch.branch_id for branch in third.choices] == [
        "B1a",
        "B1b",
        "B1c",
    ]

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


def test_third_turn_framework_reflects_compressed_memory(tmp_path: Path):
    llm = MockLLMClient()
    orchestrator = Orchestrator(
        llm=llm,
        settings=make_settings(tmp_path, compress_every=2),
    )
    first = orchestrator.run_cycle(
        PlayerInput(text=CYBERPUNK_INPUT, session_id="memory-loop")
    )
    first_choice = next(
        branch for branch in first.choices if branch.branch_id == "B"
    )
    first_text = f"{first_choice.branch_id}. {first_choice.description}"
    second = orchestrator.run_cycle(
        PlayerInput(text=f"我选择：{first_text}", session_id="memory-loop"),
        player_choice=first_text,
        player_branch_id="B",
    )
    second_choice = second.choices[0]
    second_text = f"{second_choice.branch_id}. {second_choice.description}"
    third = orchestrator.run_cycle(
        PlayerInput(text=f"我选择：{second_text}", session_id="memory-loop"),
        player_choice=second_text,
        player_branch_id=second_choice.branch_id,
    )

    assert third.scene.scene_id == "scene_002"
    assert [branch.branch_id for branch in third.choices] == [
        "B1a",
        "B1b",
        "B1c",
    ]
    context = ContextStore.load(
        tmp_path
        / "sessions"
        / "memory-loop"
        / "cycles"
        / "cycle_003"
        / "context.json"
    )
    framework = context.get("framework")
    assert isinstance(framework, StoryFramework)
    assert "盟友" in framework.main_plot or "同盟" in framework.main_plot
    assert any("盟友" in beat.summary for beat in framework.key_beats)
    assert "偏好通过寻求盟友推进调查" in llm.calls_for("planner")[-1].prompt


def test_choice_cache_prefetches_two_layers_and_consumes_them(
    tmp_path: Path,
):
    llm = MockLLMClient()
    orchestrator = Orchestrator(
        llm=llm,
        settings=make_settings(tmp_path, compress_every=50),
    )
    first = orchestrator.run_cycle(
        PlayerInput(text=CYBERPUNK_INPUT, session_id="cache")
    )

    cache = orchestrator.choice_cache_for("cache")
    assert isinstance(cache, ChoiceCache)
    for branch in first.choices:
        nexts = cache.next_choices(branch.branch_id)
        assert len(nexts) >= 3
        for next_choice in nexts:
            assert len(
                cache.grandchildren(branch.branch_id, next_choice.path[-1])
            ) >= 3
    assert llm.call_count("choice_cache") == 1

    first_choice = next(
        branch for branch in first.choices if branch.branch_id == "B"
    )
    first_text = f"{first_choice.branch_id}. {first_choice.description}"
    second = orchestrator.run_cycle(
        PlayerInput(text=f"我选择：{first_text}", session_id="cache"),
        player_choice=first_text,
        player_branch_id="B",
    )
    assert [branch.branch_id for branch in second.choices] == [
        "B1",
        "B2",
        "B3",
    ]
    # 第二轮直接使用上一层缓存，不应再次调用预取 AI。
    assert llm.call_count("choice_cache") == 1

    derived = orchestrator.choice_cache_for("cache")
    assert derived is not None
    assert all(option.depth == 1 for option in derived.options)
    assert [branch.branch_id for branch in derived.as_branches("B1")] == [
        "B1a",
        "B1b",
        "B1c",
    ]

    second_choice = second.choices[0]
    second_text = f"{second_choice.branch_id}. {second_choice.description}"
    third = orchestrator.run_cycle(
        PlayerInput(text=f"我选择：{second_text}", session_id="cache"),
        player_choice=second_text,
        player_branch_id=second_choice.branch_id,
    )
    assert [branch.branch_id for branch in third.choices] == [
        "B1a",
        "B1b",
        "B1c",
    ]
    # 第二层用完后，新一轮两层预取才会触发。
    assert llm.call_count("choice_cache") == 2


def test_prefetch_can_be_disabled(tmp_path: Path):
    llm = MockLLMClient()
    settings = Settings(
        use_mock=True,
        data_dir=tmp_path,
        memory_compress_every=50,
        review_max_rewrites=2,
        prefetch_choices=False,
    )
    orchestrator = Orchestrator(llm=llm, settings=settings)
    orchestrator.run_cycle(
        PlayerInput(text=CYBERPUNK_INPUT, session_id="no-prefetch")
    )

    assert llm.call_count("choice_cache") == 0
    assert orchestrator.choice_cache_for("no-prefetch") is None


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


def test_medium_severity_issue_does_not_block_delivery(tmp_path: Path):
    llm = MockLLMClient()
    passed = json.dumps(
        {"passed": True, "issues": [], "suggested_fix": None},
        ensure_ascii=False,
    )
    medium_failure = json.dumps(
        {
            "passed": False,
            "issues": [
                {
                    "category": "literary",
                    "detail": "可优化环境描写",
                    "severity": "medium",
                }
            ],
            "suggested_fix": "补充一段环境细节",
        },
        ensure_ascii=False,
    )
    # 四维审查中只有文学维度给出 medium 建议，不应阻断交付。
    llm.push_responses(
        "guardian_scene",
        passed,
        passed,
        passed,
        medium_failure,
        front=True,
    )

    orchestrator = Orchestrator(
        llm=llm,
        settings=make_settings(tmp_path, compress_every=50),
    )
    orchestrator.run_cycle(
        PlayerInput(text=CYBERPUNK_INPUT, session_id="medium-issue")
    )

    assert llm.call_count("script") == 1
    entries = read_runlog(tmp_path, "medium-issue")
    assert not any(entry["status"] == "review_rejected" for entry in entries)
    script_entry = next(
        entry
        for entry in entries
        if entry["step_id"] == "05_script" and entry["status"] == "success"
    )
    assert "可优化环境描写" in (script_entry["detail"] or "")


def test_filler_retries_when_unknown_npc_id_used(tmp_path: Path):
    llm = MockLLMClient()
    bad_scene = json.dumps(
        {
            "scene_id": "scene_001",
            "summary": "林默在旧城咖啡馆等待委托人。",
            "setting": "雨夜的旧城咖啡馆。",
            "actions": ["林默检查桌上的记忆芯片。"],
            "events": ["街对面的全息广告突然熄灭。"],
            "involved_npcs": ["npc_not_registered"],
        },
        ensure_ascii=False,
    )
    llm.push_responses("filler", bad_scene, front=True)

    orchestrator = Orchestrator(
        llm=llm,
        settings=make_settings(tmp_path, compress_every=50),
    )
    output = orchestrator.run_cycle(
        PlayerInput(text=CYBERPUNK_INPUT, session_id="filler-retry")
    )

    assert llm.call_count("filler") == 2
    assert set(output.scene.involved_npcs) <= {"npc_suli", "npc_kai"}


def test_script_retries_when_unknown_speaker_used(tmp_path: Path):
    llm = MockLLMClient()
    bad_script = json.dumps(
        {
            "scene_id": "scene_001",
            "lines": [
                {
                    "speaker": "旁白",
                    "text": "雨声盖住了旧城的霓虹。",
                    "emotion": "冷静",
                }
            ],
        },
        ensure_ascii=False,
    )
    llm.push_responses("script", bad_script, front=True)

    orchestrator = Orchestrator(
        llm=llm,
        settings=make_settings(tmp_path, compress_every=50),
    )
    output = orchestrator.run_cycle(
        PlayerInput(text=CYBERPUNK_INPUT, session_id="script-retry")
    )

    assert llm.call_count("script") == 2
    assert all(
        line.speaker in {"林默", "苏离", "阿凯", "维拉"}
        for line in output.script.lines
    )


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
