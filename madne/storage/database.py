"""SQLite 会话数据库。

所有可恢复状态（上下文、运行日志、记忆、NPC、选项缓存、动态计划、输出）
都写入同一个数据库文件，替代原先散落的 JSON/JSONL 文件。

约定：
- 每个会话用 ``session_id`` 作为逻辑分区；
- 复杂对象一律以 JSON 文本存储，读取方负责用 Pydantic 还原；
- 每次操作独立建连，天然支持 GUI 的后台线程；
- 首次访问旧版会话目录时自动导入历史文件（仅导入一次）。
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    cycle INTEGER NOT NULL DEFAULT 0,
    plan_mode TEXT NOT NULL DEFAULT 'dynamic'
);

CREATE TABLE IF NOT EXISTS contexts (
    session_id TEXT NOT NULL,
    cycle INTEGER NOT NULL,
    payload TEXT NOT NULL,
    plan_mode TEXT NOT NULL DEFAULT '',
    saved_at TEXT NOT NULL,
    PRIMARY KEY (session_id, cycle)
);

CREATE TABLE IF NOT EXISTS plans (
    session_id TEXT NOT NULL,
    cycle INTEGER NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (session_id, cycle)
);

CREATE TABLE IF NOT EXISTS run_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    cycle INTEGER NOT NULL,
    step_id TEXT NOT NULL,
    agent TEXT NOT NULL,
    status TEXT NOT NULL,
    attempt INTEGER NOT NULL DEFAULT 1,
    tokens_used INTEGER NOT NULL DEFAULT 0,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    detail TEXT,
    timestamp TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_run_log_session
    ON run_log (session_id, id);

CREATE TABLE IF NOT EXISTS memory_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_events_session
    ON memory_events (session_id, id);

CREATE TABLE IF NOT EXISTS compressed_memory (
    session_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_meta (
    session_id TEXT PRIMARY KEY,
    last_compressed_event_count INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS npcs (
    session_id TEXT NOT NULL,
    npc_id TEXT NOT NULL,
    position INTEGER NOT NULL DEFAULT 0,
    payload TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (session_id, npc_id)
);

CREATE TABLE IF NOT EXISTS choice_cache (
    session_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS outputs (
    session_id TEXT NOT NULL,
    cycle INTEGER NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (session_id, cycle)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


class SessionDatabase:
    """单文件 SQLite 存储，提供按会话分区的读写接口。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    # ------------------------------------------------------------------ 连接
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(_SCHEMA)
            connection.execute(
                "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)",
                ("version", str(SCHEMA_VERSION)),
            )

    @staticmethod
    def _ensure_session(
        connection: sqlite3.Connection,
        session_id: str,
        plan_mode: str = "",
    ) -> None:
        now = _now()
        connection.execute(
            """
            INSERT INTO sessions (session_id, created_at, updated_at, cycle, plan_mode)
            VALUES (?, ?, ?, 0, ?)
            ON CONFLICT(session_id) DO NOTHING
            """,
            (session_id, now, now, plan_mode or "dynamic"),
        )
        if plan_mode:
            connection.execute(
                "UPDATE sessions SET plan_mode = ? WHERE session_id = ?",
                (plan_mode, session_id),
            )

    # ------------------------------------------------------------------ 会话
    def ensure_session(self, session_id: str, plan_mode: str = "") -> None:
        with self._connect() as connection:
            self._ensure_session(connection, session_id, plan_mode)

    def session_exists(self, session_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM sessions WHERE session_id = ? LIMIT 1",
                (session_id,),
            ).fetchone()
        return row is not None

    def session_cycle(self, session_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT cycle FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return int(row["cycle"]) if row is not None else 0

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def touch_session(
        self,
        session_id: str,
        cycle: int,
        plan_mode: str | None = None,
    ) -> None:
        with self._connect() as connection:
            self._ensure_session(connection, session_id, plan_mode or "")
            connection.execute(
                """
                UPDATE sessions
                SET cycle = MAX(cycle, ?), updated_at = ?,
                    plan_mode = COALESCE(NULLIF(?, ''), plan_mode)
                WHERE session_id = ?
                """,
                (int(cycle), _now(), plan_mode or "", session_id),
            )

    # ------------------------------------------------------------------ 上下文
    def save_context(
        self,
        snapshot: dict[str, Any],
        *,
        plan_mode: str = "",
    ) -> None:
        session_id = str(snapshot.get("session_id", ""))
        if not session_id:
            raise ValueError("ContextStore 快照缺少 session_id")
        cycle = int(snapshot.get("cycle", 0))
        with self._connect() as connection:
            self._ensure_session(connection, session_id, plan_mode)
            connection.execute(
                """
                INSERT INTO contexts (session_id, cycle, payload, plan_mode, saved_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id, cycle) DO UPDATE SET
                    payload = excluded.payload,
                    plan_mode = excluded.plan_mode,
                    saved_at = excluded.saved_at
                """,
                (
                    session_id,
                    cycle,
                    _dumps(snapshot),
                    plan_mode,
                    _now(),
                ),
            )
            connection.execute(
                """
                UPDATE sessions
                SET cycle = MAX(cycle, ?), updated_at = ?
                WHERE session_id = ?
                """,
                (cycle, _now(), session_id),
            )

    def load_latest_context(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload FROM contexts
                WHERE session_id = ?
                ORDER BY cycle DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        return json.loads(row["payload"]) if row is not None else None

    def load_context(
        self,
        session_id: str,
        cycle: int,
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM contexts WHERE session_id = ? AND cycle = ?",
                (session_id, int(cycle)),
            ).fetchone()
        return json.loads(row["payload"]) if row is not None else None

    def context_cycles(self, session_id: str) -> list[int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT cycle FROM contexts WHERE session_id = ? ORDER BY cycle",
                (session_id,),
            ).fetchall()
        return [int(row["cycle"]) for row in rows]

    # ------------------------------------------------------------------ 动态计划
    def save_plan(
        self,
        session_id: str,
        cycle: int,
        payload: dict[str, Any],
    ) -> None:
        with self._connect() as connection:
            self._ensure_session(connection, session_id)
            connection.execute(
                """
                INSERT INTO plans (session_id, cycle, payload, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id, cycle) DO UPDATE SET
                    payload = excluded.payload,
                    created_at = excluded.created_at
                """,
                (session_id, int(cycle), _dumps(payload), _now()),
            )

    def load_plan(
        self,
        session_id: str,
        cycle: int,
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM plans WHERE session_id = ? AND cycle = ?",
                (session_id, int(cycle)),
            ).fetchone()
        return json.loads(row["payload"]) if row is not None else None

    def load_latest_plan(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM plans WHERE session_id = ? ORDER BY cycle DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        return json.loads(row["payload"]) if row is not None else None

    # ------------------------------------------------------------------ 运行日志
    def save_run_log(self, session_id: str, entry: dict[str, Any]) -> None:
        with self._connect() as connection:
            self._ensure_session(connection, session_id)
            connection.execute(
                """
                INSERT INTO run_log (
                    session_id, cycle, step_id, agent, status, attempt,
                    tokens_used, duration_ms, error, detail, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    int(entry.get("cycle", 0)),
                    str(entry.get("step_id", "")),
                    str(entry.get("agent", "")),
                    str(entry.get("status", "")),
                    int(entry.get("attempt", 1)),
                    int(entry.get("tokens_used", 0)),
                    int(entry.get("duration_ms", 0)),
                    entry.get("error"),
                    entry.get("detail"),
                    str(entry.get("timestamp") or _now()),
                ),
            )

    def load_run_log(
        self,
        session_id: str,
        cycle: int | None = None,
    ) -> list[dict[str, Any]]:
        query = (
            "SELECT cycle, step_id, agent, status, attempt, tokens_used, "
            "duration_ms, error, detail, timestamp FROM run_log "
            "WHERE session_id = ?"
        )
        params: list[Any] = [session_id]
        if cycle is not None:
            query += " AND cycle = ?"
            params.append(int(cycle))
        query += " ORDER BY id"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------ 记忆
    def append_memory_event(self, payload: dict[str, Any]) -> None:
        session_id = str(payload.get("session_id", ""))
        if not session_id:
            raise ValueError("MemoryEvent 缺少 session_id")
        with self._connect() as connection:
            self._ensure_session(connection, session_id)
            connection.execute(
                "INSERT INTO memory_events (session_id, payload, created_at) VALUES (?, ?, ?)",
                (session_id, _dumps(payload), _now()),
            )

    def load_memory_events(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if limit is not None:
            query = (
                "SELECT payload FROM memory_events WHERE session_id = ? "
                "ORDER BY id DESC LIMIT ?"
            )
            params: Iterable[Any] = (session_id, int(limit))
        else:
            query = (
                "SELECT payload FROM memory_events WHERE session_id = ? ORDER BY id"
            )
            params = (session_id,)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        payloads = [json.loads(row["payload"]) for row in rows]
        if limit is not None:
            payloads.reverse()
        return payloads

    def memory_event_count(self, session_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total FROM memory_events WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return int(row["total"]) if row is not None else 0

    def save_compressed_memory(
        self,
        session_id: str,
        payload: dict[str, Any],
    ) -> None:
        with self._connect() as connection:
            self._ensure_session(connection, session_id)
            connection.execute(
                """
                INSERT INTO compressed_memory (session_id, payload, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    payload = excluded.payload,
                    updated_at = excluded.updated_at
                """,
                (session_id, _dumps(payload), _now()),
            )

    def load_compressed_memory(
        self,
        session_id: str,
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM compressed_memory WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return json.loads(row["payload"]) if row is not None else None

    def get_memory_meta(self, session_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT last_compressed_event_count FROM memory_meta WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return {}
        return {
            "last_compressed_event_count": int(
                row["last_compressed_event_count"]
            )
        }

    def set_memory_meta(self, session_id: str, meta: dict[str, Any]) -> None:
        count = int(meta.get("last_compressed_event_count", 0))
        with self._connect() as connection:
            self._ensure_session(connection, session_id)
            connection.execute(
                """
                INSERT INTO memory_meta (
                    session_id, last_compressed_event_count, updated_at
                ) VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    last_compressed_event_count = excluded.last_compressed_event_count,
                    updated_at = excluded.updated_at
                """,
                (session_id, count, _now()),
            )

    # ------------------------------------------------------------------ NPC
    def save_npcs(
        self,
        session_id: str,
        profiles: list[dict[str, Any]],
    ) -> None:
        with self._connect() as connection:
            self._ensure_session(connection, session_id)
            connection.execute(
                "DELETE FROM npcs WHERE session_id = ?",
                (session_id,),
            )
            now = _now()
            for position, profile in enumerate(profiles):
                connection.execute(
                    """
                    INSERT INTO npcs (
                        session_id, npc_id, position, payload, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        str(profile.get("npc_id", "")),
                        position,
                        _dumps(profile),
                        now,
                    ),
                )

    def load_npcs(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload FROM npcs
                WHERE session_id = ?
                ORDER BY position
                """,
                (session_id,),
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    # ------------------------------------------------------------------ 选项缓存
    def save_choice_cache(
        self,
        session_id: str,
        payload: dict[str, Any],
    ) -> None:
        with self._connect() as connection:
            self._ensure_session(connection, session_id)
            connection.execute(
                """
                INSERT INTO choice_cache (session_id, payload, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    payload = excluded.payload,
                    updated_at = excluded.updated_at
                """,
                (session_id, _dumps(payload), _now()),
            )

    def load_choice_cache(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM choice_cache WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return json.loads(row["payload"]) if row is not None else None

    def clear_choice_cache(self, session_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM choice_cache WHERE session_id = ?",
                (session_id,),
            )

    # ------------------------------------------------------------------ 输出
    def save_output(
        self,
        session_id: str,
        cycle: int,
        payload: dict[str, Any],
    ) -> None:
        with self._connect() as connection:
            self._ensure_session(connection, session_id)
            connection.execute(
                """
                INSERT INTO outputs (session_id, cycle, payload, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id, cycle) DO UPDATE SET
                    payload = excluded.payload,
                    created_at = excluded.created_at
                """,
                (session_id, int(cycle), _dumps(payload), _now()),
            )

    def load_output(
        self,
        session_id: str,
        cycle: int,
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM outputs WHERE session_id = ? AND cycle = ?",
                (session_id, int(cycle)),
            ).fetchone()
        return json.loads(row["payload"]) if row is not None else None

    # ------------------------------------------------------------------ 旧数据迁移
    def import_legacy_session(
        self,
        session_id: str,
        session_dir: str | Path,
    ) -> bool:
        """把旧版 ``sessions/<id>/*.json`` 导入数据库。

        返回 True 表示执行了迁移；已迁移过或目录不存在时返回 False。
        """
        directory = Path(session_dir)
        if self.session_exists(session_id) or not directory.exists():
            return False

        self.ensure_session(session_id)
        self._import_legacy_contexts(session_id, directory)
        self._import_legacy_run_log(session_id, directory)
        self._import_legacy_memory(session_id, directory)
        self._import_legacy_npcs(session_id, directory)
        self._import_legacy_choice_cache(session_id, directory)
        self._import_legacy_outputs(session_id, directory)
        return True

    def _import_legacy_contexts(self, session_id: str, directory: Path) -> None:
        for path in sorted(directory.glob("cycles/cycle_*/context.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict) or not payload.get("session_id"):
                continue
            payload["session_id"] = session_id
            try:
                self.save_context(payload)
            except (OSError, ValueError):
                continue

    def _import_legacy_run_log(self, session_id: str, directory: Path) -> None:
        path = directory / "runlog.jsonl"
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                try:
                    self.save_run_log(session_id, payload)
                except (OSError, ValueError, TypeError):
                    continue

    def _import_legacy_memory(self, session_id: str, directory: Path) -> None:
        memory_dir = directory / "memory"
        full_path = memory_dir / "full.jsonl"
        if full_path.exists():
            for line in full_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    payload.setdefault("session_id", session_id)
                    try:
                        self.append_memory_event(payload)
                    except (OSError, ValueError):
                        continue

        compressed_path = memory_dir / "compressed.json"
        if compressed_path.exists():
            try:
                payload = json.loads(compressed_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, dict):
                payload.setdefault("session_id", session_id)
                try:
                    self.save_compressed_memory(session_id, payload)
                except OSError:
                    pass

        meta_path = memory_dir / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                meta = None
            if isinstance(meta, dict):
                try:
                    self.set_memory_meta(session_id, meta)
                except OSError:
                    pass

    def _import_legacy_npcs(self, session_id: str, directory: Path) -> None:
        path = directory / "npcs.json"
        if not path.exists():
            return
        try:
            profiles = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(profiles, list):
            try:
                self.save_npcs(session_id, profiles)
            except (OSError, TypeError):
                pass

    def _import_legacy_choice_cache(
        self,
        session_id: str,
        directory: Path,
    ) -> None:
        path = directory / "choice_cache.json"
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(payload, dict):
            try:
                self.save_choice_cache(session_id, payload)
            except OSError:
                pass

    def _import_legacy_outputs(self, session_id: str, directory: Path) -> None:
        for path in sorted(directory.glob("cycles/cycle_*/output.json")):
            try:
                cycle = int(path.parent.name.removeprefix("cycle_"))
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            if isinstance(payload, dict):
                try:
                    self.save_output(session_id, cycle, payload)
                except OSError:
                    continue
