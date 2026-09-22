"""Persistent local SQLite checkpointer for LangGraph threads."""
import atexit
import os
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver

_sqlite_context: AbstractContextManager[Any] | None = None
_checkpointer: SqliteSaver | None = None


def build_checkpointer() -> SqliteSaver:
    """Create one process-wide SQLite checkpointer for persistent thread memory."""
    global _checkpointer, _sqlite_context
    if _checkpointer is not None:
        return _checkpointer

    load_dotenv()
    database_path = Path(
        os.getenv("SQLITE_CHECKPOINT_PATH", "data/checkpoints.sqlite")
    )
    database_path.parent.mkdir(parents=True, exist_ok=True)
    _sqlite_context = SqliteSaver.from_conn_string(str(database_path))
    _checkpointer = _sqlite_context.__enter__()
    _checkpointer.setup()
    atexit.register(_close_sqlite_context)
    return _checkpointer


def _close_sqlite_context() -> None:
    if _sqlite_context is not None:
        _sqlite_context.__exit__(None, None, None)
