"""Structured JSON logger for the closed-loop orchestrator."""

import json
from datetime import datetime, timezone


import os


class JsonLogger:
    """Emit structured JSON log records to stdout and optionally to a file."""

    def __init__(self, name: str):
        self._name = name
        self._audit_log_path = os.environ.get("AUDIT_LOG_PATH")

    def _emit(self, level: str, event_type: str, **kwargs):
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "event_type": event_type,
            **kwargs,
        }
        log_line = json.dumps(record)
        print(log_line, flush=True)

        if self._audit_log_path:
            try:
                # Ensure the parent directory exists
                parent_dir = os.path.dirname(self._audit_log_path)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)
                with open(self._audit_log_path, "a", encoding="utf-8") as f:
                    f.write(log_line + "\n")
            except Exception:
                pass

    def info(self, event_type: str, **kwargs):
        self._emit("INFO", event_type, **kwargs)

    def warning(self, event_type: str, **kwargs):
        self._emit("WARNING", event_type, **kwargs)

    def error(self, event_type: str, **kwargs):
        self._emit("ERROR", event_type, **kwargs)
