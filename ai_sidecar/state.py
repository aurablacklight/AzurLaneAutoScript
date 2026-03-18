import json
import os
import tempfile
from datetime import datetime
from typing import Optional

from models import EventPayload, EventType, GameState, TaskInfo


class StateManager:
    def __init__(self, state_file: str, max_history: int = 50):
        self.state_file = state_file
        self.max_history = max_history
        self.state = self._load()

    def _load(self) -> GameState:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                return GameState.model_validate(data)
            except (json.JSONDecodeError, Exception):
                return GameState()
        return GameState()

    def _save(self) -> None:
        self.state.last_updated = datetime.now()
        data = self.state.model_dump(mode="json")
        dir_name = os.path.dirname(self.state_file) or "."
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2, default=str)
            os.rename(tmp_path, self.state_file)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def update(self, event: EventPayload) -> None:
        if event.event_type == EventType.cycle_start:
            self.state.pending_tasks = event.pending_tasks or []
            self.state.waiting_tasks = event.waiting_tasks or []
            if event.screenshot is not None:
                self.state.last_screenshot = event.screenshot
        elif event.event_type == EventType.task_complete:
            info = TaskInfo(
                task=event.task,
                success=event.success or True,
                duration=event.duration if event.duration is not None else 0.0,
            )
            self.state.completion_history.append(info)
            self.state.completion_history = self.state.completion_history[-self.max_history:]
        elif event.event_type == EventType.task_failed:
            info = TaskInfo(
                task=event.task,
                success=False,
                duration=event.duration if event.duration is not None else 0.0,
                error=event.error,
                count=event.count if event.count is not None else 0,
            )
            self.state.failure_history.append(info)
            self.state.failure_history = self.state.failure_history[-self.max_history:]
        elif event.event_type == EventType.unknown_state:
            if event.screenshot is not None:
                self.state.last_screenshot = event.screenshot
        self._save()

    def get_summary(self) -> str:
        lines = []
        lines.append(f"Pending tasks: {', '.join(self.state.pending_tasks) or 'none'}")
        lines.append(f"Waiting tasks: {', '.join(self.state.waiting_tasks) or 'none'}")
        if self.state.completion_history:
            recent = self.state.completion_history[-5:]
            lines.append("Recent completions:")
            for t in recent:
                lines.append(f"  - {t.task} ({t.duration:.0f}s)" if t.duration else f"  - {t.task}")
        if self.state.failure_history:
            recent = self.state.failure_history[-5:]
            lines.append("Recent failures:")
            for t in recent:
                lines.append(f"  - {t.task}: {t.error} (count: {t.count})")
        return "\n".join(lines)
