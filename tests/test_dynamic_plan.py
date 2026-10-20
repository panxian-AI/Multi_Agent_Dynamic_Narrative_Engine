from pathlib import Path

from madne.engine.context_store import ContextStore
from madne.engine.plan_runtime import PlanRuntime
from madne.engine.templates import compile_plan
from madne.models import PlannedStep, PlannerDecision, TaskPlan
from madne.storage import SessionDatabase


def test_compile_plan_fills_delivery_chain():
    decision = PlannerDecision(
        steps=[PlannedStep(agent="script", reason="只请求台本")],
        priority_focus=["推进剧情"],
    )
    plan = compile_plan(
        decision,
        has_existing_world=False,
        available_keys={"brief", "player_input"},
        max_steps=8,
    )

    assert plan.mode == "dynamic"
    assert plan.agent_sequence() == [
        "world",
        "framework",
        "npc",
        "filler",
        "script",
    ]
    assert plan.steps[-1].output_key == "script"


def test_compile_plan_reuses_world_and_orders_npc_update_first():
    decision = PlannerDecision(
        steps=[
            PlannedStep(agent="npc_update"),
            PlannedStep(agent="filler"),
            PlannedStep(agent="script"),
        ]
    )
    plan = compile_plan(
        decision,
        has_existing_world=True,
        available_keys={
            "brief",
            "player_input",
            "world",
            "npcs",
            "previous_scene",
        },
        max_steps=8,
    )

    assert "world" not in plan.agent_sequence()
    assert plan.agent_sequence() == [
        "npc_update",
        "framework",
        "filler",
        "script",
    ]


def test_plan_runtime_injects_missing_producers():
    plan = TaskPlan(
        steps=[
            {
                "step_id": "01_filler",
                "agent": "filler",
                "input_keys": ["framework", "world", "npcs"],
                "output_key": "scene",
                "required_inputs": ["framework", "world", "npcs"],
            },
            {
                "step_id": "02_script",
                "agent": "script",
                "input_keys": ["scene", "npcs"],
                "output_key": "script",
                "required_inputs": ["scene", "npcs"],
            },
        ],
        mode="dynamic",
        max_steps=8,
    )
    ctx = ContextStore(session_id="dynamic", cycle=1)
    ctx.set("brief", object())
    ctx.set("player_input", object())

    runtime = PlanRuntime(plan, repair_limit=1, max_steps=8)
    executed: list[str] = []
    while True:
        step = runtime.next_step(ctx)
        if step is None:
            break
        executed.append(step.agent)
        ctx.set(step.output_key, object())
        ctx.mark_completed(step.step_id)
        runtime.complete(step)

    assert executed == ["world", "framework", "npc", "filler", "script"]
    assert {mutation.action for mutation in runtime.mutations} == {"insert"}


def test_database_roundtrip_and_legacy_import(tmp_path: Path):
    database = SessionDatabase(tmp_path / "madne.db")
    database.save_context(
        {
            "format_version": 1,
            "session_id": "db-test",
            "cycle": 2,
            "completed_steps": ["01_world"],
            "warnings": [],
            "data": {},
        }
    )
    database.save_run_log(
        "db-test",
        {
            "cycle": 2,
            "step_id": "01_world",
            "agent": "world",
            "status": "success",
            "timestamp": "2026-01-01T00:00:00+00:00",
        },
    )
    database.append_memory_event(
        {"session_id": "db-test", "raw_content": "玩家进入旧城"}
    )
    database.save_npcs(
        "db-test",
        [{"npc_id": "npc_kai", "name": "阿凯", "role": "掮客", "traits": {}}],
    )

    assert database.load_latest_context("db-test")["cycle"] == 2
    assert database.load_run_log("db-test")[0]["agent"] == "world"
    assert database.memory_event_count("db-test") == 1
    assert database.load_npcs("db-test")[0]["npc_id"] == "npc_kai"

    legacy_dir = tmp_path / "sessions" / "legacy"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "npcs.json").write_text(
        '[{"npc_id": "npc_suli", "name": "苏离", "role": "记忆商人", "traits": {}}]',
        encoding="utf-8",
    )
    (legacy_dir / "runlog.jsonl").write_text(
        '{"cycle": 1, "step_id": "01_world", "agent": "world", '
        '"status": "success", "timestamp": "2026-01-01T00:00:00+00:00"}\n',
        encoding="utf-8",
    )

    assert database.import_legacy_session("legacy", legacy_dir) is True
    assert database.session_exists("legacy")
    assert database.load_npcs("legacy")[0]["npc_id"] == "npc_suli"
    assert database.import_legacy_session("legacy", legacy_dir) is False
