"""M3 工作流引擎：规则驱动状态机，负责调度、审查点、回退与断点续跑。"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from pydantic import BaseModel

from ..agents import (
    ChoiceCacheAI,
    FillerAI,
    FrameworkAI,
    GuardianAI,
    IntentAI,
    NPCAI,
    PlannerAI,
    ScriptAI,
    WorldAI,
)
from ..agents.prompts import format_review_feedback
from ..config import Settings
from ..exceptions import EngineError
from ..image import (
    ImageGenClient,
    ImageRequest,
    build_negative_prompt,
    build_scene_prompt,
    create_image_client,
)
from ..image.consistency import compute_metrics
from ..image.prompts import select_involved_npcs
from ..llm import LLMClient, create_llm_client
from ..memory import MemorySystem
from ..models import (
    CachedChoice,
    ChoiceCache,
    CompressedMemory,
    GameOutput,
    Illustration,
    ImageCharacter,
    MemoryEvent,
    NPCProfile,
    PlayerInput,
    ReviewReport,
    SceneOutline,
    StoryFramework,
    TaskPlan,
    TaskStep,
    WorldBible,
)
from ..npc import NPCSystem
from ..output import OutputAdapter
from .choice_cache import ChoiceCacheStore
from .context_store import ContextStore
from .run_log import RunLog, RunLogEntry


@dataclass(slots=True)
class SessionRuntime:
    session_id: str
    state_dir: Path
    memory: MemorySystem
    npc_system: NPCSystem
    run_log: RunLog
    choice_cache_store: ChoiceCacheStore
    context: ContextStore
    cycle: int = 0
    last_output: GameOutput | None = None
    output_path: Path | None = None


class Orchestrator:
    """不包含 AI 逻辑，只消费 TaskPlan 并按规则执行。"""

    def __init__(
        self,
        llm: LLMClient | None = None,
        settings: Settings | None = None,
        data_dir: str | Path | None = None,
        on_event: Callable[[RunLogEntry], None] | None = None,
    ) -> None:
        base_settings = settings or Settings.from_env()
        if data_dir is not None:
            base_settings = replace(base_settings, data_dir=Path(data_dir))
        self.settings = base_settings
        self.llm = llm or create_llm_client(self.settings)
        self._on_event = on_event

        self.intent_ai = IntentAI(self.llm)
        self.planner_ai = PlannerAI(self.llm)
        self.world_ai = WorldAI(self.llm)
        self.npc_ai = NPCAI(self.llm)
        self.framework_ai = FrameworkAI(self.llm)
        self.filler_ai = FillerAI(self.llm)
        self.script_ai = ScriptAI(self.llm)
        self.choice_cache_ai = ChoiceCacheAI(self.llm)
        self.guardian_ai = GuardianAI(self.llm)
        self.adapter = OutputAdapter()
        self.image_client: ImageGenClient | None = (
            create_image_client(self.settings)
            if self.settings.img_enabled
            else None
        )

        self._runtimes: dict[str, SessionRuntime] = {}

    # ------------------------------------------------------------------ 公共入口
    def run_cycle(
        self,
        player_input: PlayerInput,
        player_choice: str | None = None,
        player_branch_id: str | None = None,
    ) -> GameOutput:
        """执行一次完整叙事生成循环。"""
        runtime = self._runtime(player_input.session_id)
        runtime.cycle += 1
        cycle = runtime.cycle
        previous = runtime.context
        ctx = ContextStore(session_id=runtime.session_id, cycle=cycle)

        # 续写：继承上一轮的世界、角色池与框架
        for key in ("world", "npcs"):
            if previous.has(key):
                ctx.set(key, previous.get(key))
        if not ctx.has("npcs") and runtime.npc_system.all_profiles():
            ctx.set("npcs", runtime.npc_system.all_profiles())
        if previous.has("framework"):
            ctx.set("previous_framework", previous.get("framework"))
        if runtime.last_output is not None:
            ctx.set("previous_scene", runtime.last_output.scene)
        if not ctx.has("previous_scene"):
            previous_output = previous.get("game_output")
            if isinstance(previous_output, GameOutput):
                ctx.set("previous_scene", previous_output.scene)

        prefetched, active_cache, active_branch_id = (
            self._load_prefetched_choices(
                runtime,
                previous,
                player_choice,
                player_branch_id,
            )
        )
        if prefetched:
            ctx.set("prefetched_choices", prefetched)
        if active_cache is not None and active_branch_id:
            ctx.set("active_choice_cache", active_cache)
            ctx.set("active_cached_branch_id", active_branch_id)

        ctx.set("player_input", player_input)
        compressed = runtime.memory.load_compressed() or CompressedMemory(
            session_id=runtime.session_id
        )
        ctx.set("compressed_memory", compressed)
        if player_choice:
            ctx.set("player_choice", player_choice)

        # 完整记忆：玩家输入与选择
        runtime.memory.record(
            MemoryEvent(
                session_id=runtime.session_id,
                raw_content=player_input.text,
                player_choice=player_choice,
            )
        )
        if player_choice:
            runtime.memory.record(
                MemoryEvent(
                    session_id=runtime.session_id,
                    raw_content=f"玩家选择：{player_choice}",
                    player_choice=player_choice,
                )
            )

        # 1. 意图理解
        brief = self.intent_ai.run(player_input, compressed)
        ctx.set("brief", brief)

        # 2. 任务规划（静态模板实例化）
        plan = self.planner_ai.plan(
            brief,
            compressed,
            has_existing_world=ctx.has("world"),
        )
        ctx.set("plan", plan)
        self._save_context(runtime, ctx)

        # 3. 执行工作流
        self._execute_plan(runtime, ctx)

        # 4. 输出适配 + 记忆更新
        output = self._finalize(runtime, ctx)

        runtime.context = ctx
        runtime.last_output = output
        return output

    def resume(self, session_id: str) -> GameOutput:
        """从最近一次持久化的 ContextStore 继续未完成的工作流。"""
        state_dir = self.settings.data_dir / "sessions" / session_id
        cycles_dir = state_dir / "cycles"
        if not cycles_dir.exists():
            raise EngineError(f"会话 '{session_id}' 没有可恢复的运行记录")

        candidates = sorted(
            path
            for path in cycles_dir.glob("cycle_*")
            if (path / "context.json").exists()
        )
        if not candidates:
            raise EngineError(f"会话 '{session_id}' 没有可恢复的上下文文件")

        ctx = ContextStore.load(candidates[-1] / "context.json")
        cached = ctx.get("game_output")
        if isinstance(cached, GameOutput):
            if cached.illustration is not None or self.image_client is None:
                return cached
            # 旧版本/上次关闭生图时留下的 GameOutput：补生成插图
            runtime = self._runtime(session_id)
            runtime.cycle = ctx.cycle
            runtime.context = ctx
            output = self._finalize(runtime, ctx)
            runtime.last_output = output
            return output

        if not ctx.has("plan"):
            raise EngineError("上下文缺少 TaskPlan，无法恢复（可能在上一次规划阶段中断）")

        runtime = self._runtime(session_id)
        runtime.cycle = ctx.cycle
        runtime.context = ctx
        self._execute_plan(runtime, ctx)
        output = self._finalize(runtime, ctx)
        runtime.context = ctx
        runtime.last_output = output
        return output

    def output_path_for(self, session_id: str) -> Path | None:
        runtime = self._runtimes.get(session_id)
        return runtime.output_path if runtime is not None else None

    def choice_cache_for(self, session_id: str) -> ChoiceCache | None:
        runtime = self._runtimes.get(session_id)
        if runtime is not None:
            return runtime.choice_cache_store.load()
        return ChoiceCacheStore(
            self.settings.data_dir / "sessions" / session_id
        ).load()

    def _load_prefetched_choices(
        self,
        runtime: SessionRuntime,
        previous: ContextStore,
        player_choice: str | None,
        player_branch_id: str | None,
    ) -> tuple[list[CachedChoice], ChoiceCache | None, str | None]:
        if not self.settings.prefetch_choices:
            return [], None, None

        previous_framework = previous.get("framework")
        previous_output = runtime.last_output or previous.get("game_output")
        if not isinstance(previous_framework, StoryFramework):
            return [], None, None
        if not isinstance(previous_output, GameOutput):
            return [], None, None

        cache = runtime.choice_cache_store.load()
        if cache is None:
            return [], None, None
        if cache.framework_hash != self._framework_hash(
            previous_framework,
            previous_output.scene,
        ):
            return [], None, None

        branch_id = player_branch_id or self._resolve_branch_id(
            player_choice,
            previous_output,
        )
        if not branch_id:
            return [], None, None
        choices = cache.next_choices(branch_id)
        if len(choices) < 3:
            return [], None, None
        return choices, cache, branch_id

    # ------------------------------------------------------------------ 会话运行时
    def _runtime(self, session_id: str) -> SessionRuntime:
        runtime = self._runtimes.get(session_id)
        if runtime is not None:
            return runtime

        state_dir = self.settings.data_dir / "sessions" / session_id
        state_dir.mkdir(parents=True, exist_ok=True)
        runtime = SessionRuntime(
            session_id=session_id,
            state_dir=state_dir,
            memory=MemorySystem(
                state_dir,
                self.llm,
                compress_every=self.settings.memory_compress_every,
                session_id=session_id,
            ),
            npc_system=NPCSystem(state_dir / "npcs.json").load(),
            run_log=RunLog(state_dir / "runlog.jsonl"),
            choice_cache_store=ChoiceCacheStore(state_dir),
            context=ContextStore(session_id=session_id),
        )
        self._runtimes[session_id] = runtime
        return runtime

    # ------------------------------------------------------------------ 步骤执行
    def _execute_plan(self, runtime: SessionRuntime, ctx: ContextStore) -> None:
        plan = ctx.require("plan")
        if not isinstance(plan, TaskPlan):
            raise EngineError("上下文中的 plan 类型非法")
        for step in plan.steps:
            if ctx.is_completed(step.step_id):
                continue
            self._execute_step(runtime, ctx, step)

    def _execute_step(
        self,
        runtime: SessionRuntime,
        ctx: ContextStore,
        step: TaskStep,
    ) -> object | None:
        feedback: str | None = None
        attempt = 0
        rewrites = 0

        while True:
            attempt += 1
            started = time.perf_counter()
            try:
                output = self._dispatch(runtime, ctx, step, feedback)
                duration_ms = int((time.perf_counter() - started) * 1000)
                tokens = self._estimate_tokens(ctx, step, output)
                report = self._review_output(step, output, ctx)

                if report is not None and not report.passed:
                    self._log(
                        runtime,
                        ctx,
                        step,
                        status="review_rejected",
                        attempt=attempt,
                        tokens=tokens,
                        duration_ms=duration_ms,
                        error=report.suggested_fix,
                        detail=self._issues_text(report),
                    )
                    rewrites += 1
                    if rewrites <= self.settings.review_max_rewrites:
                        feedback = format_review_feedback(report)
                        continue

                    # 重写仍不通过：降级交付并记录告警
                    ctx.set(step.output_key, output)
                    ctx.mark_completed(step.step_id)
                    ctx.add_warning(
                        f"{step.step_id} 审查未通过且已达最大重写次数，按降级产出交付"
                    )
                    self._log(
                        runtime,
                        ctx,
                        step,
                        status="degraded",
                        attempt=attempt,
                        tokens=tokens,
                        duration_ms=duration_ms,
                        detail=self._issues_text(report),
                    )
                    self._save_context(runtime, ctx)
                    return output

                ctx.set(step.output_key, output)
                ctx.mark_completed(step.step_id)
                self._log(
                    runtime,
                    ctx,
                    step,
                    status="success",
                    attempt=attempt,
                    tokens=tokens,
                    duration_ms=duration_ms,
                    detail=self._issues_text(report) if report is not None else None,
                )
                self._save_context(runtime, ctx)
                return output

            except Exception as exc:
                duration_ms = int((time.perf_counter() - started) * 1000)
                can_retry = step.fallback == "retry" and attempt < 2
                self._log(
                    runtime,
                    ctx,
                    step,
                    status="retry" if can_retry else "failed",
                    attempt=attempt,
                    tokens=0,
                    duration_ms=duration_ms,
                    error=str(exc),
                )
                if can_retry:
                    continue

                ctx.add_warning(f"{step.step_id} 执行失败（{step.fallback}）：{exc}")
                ctx.mark_completed(step.step_id)
                self._log(
                    runtime,
                    ctx,
                    step,
                    status="degraded" if step.fallback == "degrade" else "skipped",
                    attempt=attempt,
                    tokens=0,
                    duration_ms=duration_ms,
                    error=str(exc),
                )
                self._save_context(runtime, ctx)
                return None

    def _dispatch(
        self,
        runtime: SessionRuntime,
        ctx: ContextStore,
        step: TaskStep,
        feedback: str | None,
    ) -> object:
        agent = step.agent

        if agent == "world":
            return self.world_ai.run(
                ctx.require("brief"),
                existing_world=ctx.get("world"),
                feedback=feedback,
            )

        if agent == "framework":
            return self.framework_ai.run(
                world=ctx.require("world"),
                compressed_memory=ctx.get("compressed_memory"),
                player_input=ctx.require("player_input"),
                previous_framework=ctx.get("previous_framework"),
                previous_scene=ctx.get("previous_scene"),
                prefetched_choices=ctx.get("prefetched_choices"),
                feedback=feedback,
            )

        if agent == "npc":
            roster = self.npc_ai.run(
                brief=ctx.require("brief"),
                world=ctx.require("world"),
                framework=ctx.require("framework"),
                existing_npcs=runtime.npc_system.all_profiles(),
                feedback=feedback,
            )
            return runtime.npc_system.set_profiles(roster.characters)

        if agent == "npc_update":
            choice = ctx.get("player_choice") or ctx.require("player_input").text
            previous_scene = ctx.get("previous_scene")
            involved = (
                previous_scene.involved_npcs
                if previous_scene is not None
                else None
            )
            profiles = runtime.npc_system.apply_player_choice(choice, involved)
            if runtime.npc_system.last_changes:
                ctx.set("last_npc_changes", list(runtime.npc_system.last_changes))
            return profiles

        if agent == "filler":
            return self.filler_ai.run(
                ctx.require("framework"),
                ctx.require("world"),
                ctx.require("npcs"),
                previous_scene=ctx.get("previous_scene"),
                feedback=feedback,
            )

        if agent == "script":
            return self.script_ai.run(
                ctx.require("scene"),
                ctx.require("npcs"),
                world=ctx.get("world"),
                feedback=feedback,
            )

        raise EngineError(f"未知 Agent: {agent}")

    def _review_output(
        self,
        step: TaskStep,
        output: object,
        ctx: ContextStore,
    ) -> ReviewReport | None:
        if step.agent == "world":
            return self.guardian_ai.review_world(output)
        if step.agent == "framework":
            return self.guardian_ai.review_framework(
                output,
                ctx.require("world"),
                ctx.get("compressed_memory"),
            )
        if step.agent == "script":
            return self.guardian_ai.review_scene(
                ctx.require("scene"),
                output,
                ctx.require("npcs"),
                ctx.get("world"),
            )
        return None

    # ------------------------------------------------------------------ 收尾与持久化
    def _finalize(
        self,
        runtime: SessionRuntime,
        ctx: ContextStore,
    ) -> GameOutput:
        cached = ctx.get("game_output")
        if isinstance(cached, GameOutput):
            if cached.illustration is not None or self.image_client is None:
                return cached
            cached_scene = ctx.get("scene")
            cached_world = ctx.get("world")
            if isinstance(cached_scene, SceneOutline) and isinstance(
                cached_world, WorldBible
            ):
                illustration = self._illustrate(
                    runtime, ctx, cached_scene, cached_world
                )
                if illustration is not None:
                    cached = cached.model_copy(
                        update={"illustration": illustration}
                    )
                    ctx.set("game_output", cached)
                    cycle_dir = (
                        runtime.state_dir
                        / "cycles"
                        / f"cycle_{ctx.cycle:03d}"
                    )
                    runtime.output_path = self.adapter.write(
                        cached, cycle_dir / "output.json"
                    )
                    self._save_context(runtime, ctx)
            return cached

        scene = ctx.require("scene")
        script = ctx.require("script")
        framework = ctx.require("framework")
        world = ctx.require("world")
        illustration = self._illustrate(runtime, ctx, scene, world)
        output = self.adapter.build(
            session_id=runtime.session_id,
            scene=scene,
            script=script,
            framework=framework,
            world=world,
            illustration=illustration,
        )
        ctx.set("game_output", output)

        cycle_dir = runtime.state_dir / "cycles" / f"cycle_{ctx.cycle:03d}"
        runtime.output_path = self.adapter.write(output, cycle_dir / "output.json")

        choices_text = " / ".join(
            f"{branch.branch_id}:{branch.description}"
            for branch in output.choices
        )
        runtime.memory.record(
            MemoryEvent(
                session_id=runtime.session_id,
                raw_content=(
                    f"生成场景 {scene.scene_id}：{scene.summary}；"
                    f"提供选项：{choices_text}"
                ),
            )
        )
        compressed = runtime.memory.maybe_compress()
        if compressed is not None:
            ctx.set("compressed_memory", compressed)
        if runtime.memory.last_error:
            ctx.add_warning(runtime.memory.last_error)

        if not self._derive_forward_cache(runtime, ctx, output, framework):
            self._prefetch_choices(runtime, ctx, output, framework)
        self._save_context(runtime, ctx)
        return output

    # ------------------------------------------------------------------ 插图生成
    def _illustrate(
        self,
        runtime: SessionRuntime,
        ctx: ContextStore,
        scene: SceneOutline,
        world: WorldBible,
    ) -> Illustration | None:
        """生成场景插图；任何失败都只降级为纯文本，不阻断回合交付。"""
        if self.image_client is None:
            return None

        step = TaskStep(
            step_id="05b_illustration",
            agent="image",
            input_keys=["scene", "npcs", "world"],
            output_key="illustration",
            fallback="skip",
        )
        raw_npcs = ctx.get("npcs")
        if not isinstance(raw_npcs, list):
            raw_npcs = runtime.npc_system.all_profiles()
        profiles = [npc for npc in raw_npcs if isinstance(npc, NPCProfile)]
        involved = select_involved_npcs(scene, profiles)

        prompt = build_scene_prompt(
            scene, world, profiles, self.settings.img_style
        )
        requested_mode = (
            self.settings.img_consistency_mode or "self_attn"
        ).strip().lower()
        if requested_mode not in {"self_attn", "id_injection"}:
            requested_mode = "self_attn"
        references = self._reference_images(involved)
        mode = requested_mode
        mode_warnings: list[str] = []
        if mode == "id_injection" and not references:
            mode = "self_attn"
            mode_warnings.append(
                "未找到角色定妆照，id_injection 已回退 self_attn；"
                "可在 npcs.json 中为角色设置 reference_image"
            )

        filename = f"cycle_{ctx.cycle:03d}.png"
        target = runtime.state_dir / "assets" / filename
        started = time.perf_counter()
        self._log(
            runtime,
            ctx,
            step,
            status="running",
            attempt=1,
            tokens=0,
            duration_ms=0,
            detail="正在生成场景插图…",
        )
        try:
            result = self.image_client.generate(
                ImageRequest(
                    prompt=prompt,
                    negative_prompt=build_negative_prompt(),
                    output_path=target,
                    width=self.settings.img_width,
                    height=self.settings.img_height,
                    seed=self._image_seed(runtime.session_id),
                    steps=self.settings.img_steps,
                    guidance_scale=self.settings.img_guidance,
                    reference_images=references,
                    consistency_mode=mode,
                    style=self.settings.img_style,
                )
            )
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            ctx.add_warning(f"插图生成失败，本回合仅交付文本：{exc}")
            self._log(
                runtime,
                ctx,
                step,
                status="degraded",
                attempt=1,
                tokens=0,
                duration_ms=duration_ms,
                error=str(exc),
            )
            return None

        metrics = None
        metric_warnings: list[str] = []
        if self.settings.img_metrics and references:
            metrics, metric_warnings = compute_metrics(result.path, references)

        warnings = [*mode_warnings, *result.warnings, *metric_warnings]
        if warnings:
            ctx.add_warning("插图：" + "；".join(warnings))

        illustration = Illustration(
            scene_id=scene.scene_id,
            backend=result.backend,
            prompt=result.prompt,
            negative_prompt=result.negative_prompt,
            seed=result.seed,
            width=result.width,
            height=result.height,
            consistency_mode=mode,
            relative_path=f"sessions/{runtime.session_id}/assets/{filename}",
            url=f"/assets/{runtime.session_id}/{filename}",
            characters=[
                ImageCharacter(
                    npc_id=npc.npc_id,
                    name=npc.name,
                    appearance=npc.appearance,
                )
                for npc in involved
            ],
            reference_images=[
                self._display_path(path) for path in references
            ],
            metrics=metrics,
            warnings=warnings,
        )
        duration_ms = int((time.perf_counter() - started) * 1000)
        self._log(
            runtime,
            ctx,
            step,
            status="success",
            attempt=1,
            tokens=0,
            duration_ms=duration_ms,
            detail=(
                f"{result.backend} {result.width}x{result.height} · "
                f"{mode} · seed={result.seed}"
            ),
        )
        return illustration

    def _image_seed(self, session_id: str) -> int:
        """同一 session 固定 seed，配合稳定外观描述提升跨回合一致性。"""
        if self.settings.img_seed is not None:
            return self.settings.img_seed
        digest = hashlib.sha256(
            f"{session_id}|madne-image".encode("utf-8")
        ).hexdigest()
        return int(digest[:8], 16)

    def _reference_images(self, npcs: list[NPCProfile]) -> list[Path]:
        paths: list[Path] = []
        for npc in npcs:
            raw = (npc.reference_image or "").strip()
            if not raw:
                continue
            path = Path(raw)
            if not path.is_absolute():
                path = self.settings.data_dir / path
            if path.exists():
                paths.append(path)
        return paths

    def _display_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(
                self.settings.data_dir.resolve()
            ).as_posix()
        except (ValueError, OSError):
            return path.as_posix()

    def _derive_forward_cache(
        self,
        runtime: SessionRuntime,
        ctx: ContextStore,
        output: GameOutput,
        framework: StoryFramework,
    ) -> bool:
        """把上一层缓存的第二层直接派生为下一轮的一层缓存。"""
        if not self.settings.prefetch_choices:
            return False

        active_cache = ctx.get("active_choice_cache")
        active_branch_id = ctx.get("active_cached_branch_id")
        if not isinstance(active_cache, ChoiceCache):
            return False
        if not isinstance(active_branch_id, str) or not active_branch_id:
            return False

        options: list[CachedChoice] = []
        coverage = {branch.branch_id: 0 for branch in framework.branches}
        for new_branch in framework.branches:
            for grandchild in active_cache.grandchildren(
                active_branch_id,
                new_branch.branch_id,
            ):
                options.append(
                    CachedChoice(
                        path=[new_branch.branch_id, grandchild.path[-1]],
                        description=grandchild.description,
                        risk_level=grandchild.risk_level,
                        requirements=grandchild.requirements,
                    )
                )
                coverage[new_branch.branch_id] += 1

        if not options or any(count < 3 for count in coverage.values()):
            return False

        derived = ChoiceCache(
            session_id=runtime.session_id,
            cycle=ctx.cycle,
            framework_hash=self._framework_hash(framework, output.scene),
            options=options,
        )
        runtime.choice_cache_store.save(derived)
        ctx.set("choice_cache", derived)

        step = TaskStep(
            step_id="06_prefetch",
            agent="choice_cache",
            input_keys=["active_choice_cache", "framework"],
            output_key="choice_cache",
            fallback="skip",
        )
        self._log(
            runtime,
            ctx,
            step,
            status="success",
            attempt=1,
            tokens=0,
            duration_ms=0,
            detail=(
                f"已从上一轮两层缓存派生 {len(options)} 个第二层选项"
            ),
        )
        return True

    def _prefetch_choices(
        self,
        runtime: SessionRuntime,
        ctx: ContextStore,
        output: GameOutput,
        framework: StoryFramework,
    ) -> None:
        if not self.settings.prefetch_choices:
            return

        step = TaskStep(
            step_id="06_prefetch",
            agent="choice_cache",
            input_keys=["framework", "scene", "compressed_memory"],
            output_key="choice_cache",
            fallback="skip",
        )
        self._log(
            runtime,
            ctx,
            step,
            status="running",
            attempt=1,
            tokens=0,
            duration_ms=0,
            detail="正在预取未来两层选项（真实模型约 20-30 秒）…",
        )
        started = time.perf_counter()
        try:
            cache = self.choice_cache_ai.run(
                world=ctx.require("world"),
                framework=framework,
                scene=output.scene,
                compressed_memory=ctx.get("compressed_memory"),
            )
            cache = cache.model_copy(
                update={
                    "session_id": runtime.session_id,
                    "cycle": ctx.cycle,
                    "framework_hash": self._framework_hash(
                        framework, output.scene
                    ),
                }
            )
            runtime.choice_cache_store.save(cache)
            ctx.set("choice_cache", cache)

            missing = [
                branch.branch_id
                for branch in framework.branches
                if len(cache.next_choices(branch.branch_id)) < 3
            ]
            if missing:
                ctx.add_warning(
                    "选项预取未完整覆盖分支: " + ", ".join(missing)
                )

            duration_ms = int((time.perf_counter() - started) * 1000)
            self._log(
                runtime,
                ctx,
                step,
                status="success",
                attempt=1,
                tokens=max(1, len(cache.model_dump_json()) // 4),
                duration_ms=duration_ms,
                detail=(
                    f"已缓存 {len(cache.options)} 个选项节点；"
                    f"覆盖分支 {len(framework.branches) - len(missing)}/"
                    f"{len(framework.branches)}"
                ),
            )
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            ctx.add_warning(f"选项预取失败：{exc}")
            self._log(
                runtime,
                ctx,
                step,
                status="failed",
                attempt=1,
                tokens=0,
                duration_ms=duration_ms,
                error=str(exc),
            )

    @staticmethod
    def _resolve_branch_id(
        player_choice: str | None,
        previous_output: GameOutput,
    ) -> str | None:
        if not player_choice:
            return None
        for branch in previous_output.choices:
            if player_choice == branch.branch_id:
                return branch.branch_id
            if player_choice.startswith(f"{branch.branch_id}."):
                return branch.branch_id
            if player_choice.startswith(f"{branch.branch_id} "):
                return branch.branch_id
            if branch.description in player_choice:
                return branch.branch_id
        return None

    @staticmethod
    def _framework_hash(
        framework: StoryFramework,
        scene: SceneOutline,
    ) -> str:
        payload = f"{framework.model_dump_json()}|{scene.scene_id}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _save_context(self, runtime: SessionRuntime, ctx: ContextStore) -> Path:
        path = (
            runtime.state_dir
            / "cycles"
            / f"cycle_{ctx.cycle:03d}"
            / "context.json"
        )
        return ctx.save(path)

    def _log(
        self,
        runtime: SessionRuntime,
        ctx: ContextStore,
        step: TaskStep,
        *,
        status: str,
        attempt: int,
        tokens: int,
        duration_ms: int,
        error: str | None = None,
        detail: str | None = None,
    ) -> None:
        entry = runtime.run_log.record(
            cycle=ctx.cycle,
            step_id=step.step_id,
            agent=step.agent,
            status=status,
            attempt=attempt,
            tokens_used=tokens,
            duration_ms=duration_ms,
            error=error,
            detail=detail,
        )
        if self._on_event is not None:
            self._on_event(entry)

    @staticmethod
    def _issues_text(report: ReviewReport) -> str | None:
        if not report.issues:
            return None
        return "；".join(
            f"[{issue.category}/{issue.severity}] {issue.detail}"
            for issue in report.issues
        )[:2000]

    @staticmethod
    def _estimate_tokens(
        ctx: ContextStore,
        step: TaskStep,
        output: object,
    ) -> int:
        total_chars = 0
        if isinstance(output, BaseModel):
            total_chars += len(output.model_dump_json())
        else:
            total_chars += len(str(output))
        for key in step.input_keys:
            value = ctx.get(key)
            if value is not None:
                total_chars += len(str(value))
        return max(1, total_chars // 4)
