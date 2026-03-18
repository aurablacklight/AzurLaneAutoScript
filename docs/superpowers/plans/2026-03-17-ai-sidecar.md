# AI Sidecar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an event-driven AI sidecar service that provides intelligent task prioritization for ALAS via any OpenAI-compatible API.

**Architecture:** Two Docker containers — ALAS (existing, with a thin hook layer) and a new FastAPI sidecar. ALAS fires HTTP events at key scheduler points; the sidecar consults an AI provider and returns task directives (reprioritize, skip, pause, continue). All AI logic lives in the sidecar; ALAS changes are minimal and fail-safe.

**Tech Stack:** Python 3.11+ (sidecar), Python 3.7 (ALAS hook), FastAPI, Uvicorn, `openai` SDK, Pydantic v2, `requests`, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-03-17-ai-sidecar-design.md`

---

## File Structure

### New Files (sidecar service)

| File | Responsibility |
|------|---------------|
| `ai_sidecar/models.py` | Pydantic models: EventPayload, Directive, GameState |
| `ai_sidecar/state.py` | In-memory state model with atomic JSON persistence |
| `ai_sidecar/ai_client.py` | OpenAI-compatible API client with timeout/retry/fallback |
| `ai_sidecar/directive_engine.py` | Directive validation and decision logging |
| `ai_sidecar/main.py` | FastAPI app: `/health`, `/api/events` |
| `ai_sidecar/config.yaml` | Sidecar configuration (documentation only — config via env vars) |
| `ai_sidecar/requirements.txt` | Python dependencies |
| `ai_sidecar/conftest.py` | pytest path configuration for bare imports |
| `ai_sidecar/Dockerfile` | Python 3.11 container |
| `ai_sidecar/prompts/system.txt` | System prompt template |
| `ai_sidecar/prompts/user_template.txt` | User message template |

### New Files (ALAS hook layer)

| File | Responsibility |
|------|---------------|
| `module/ai_hook/__init__.py` | Package init |
| `module/ai_hook/hook.py` | AISidecarClient: HTTP notify, screenshot encoding, debounce |

### New Files (Docker/config)

| File | Responsibility |
|------|---------------|
| `.env.example` | Template for AI credentials |

### Modified Files

| File | Change |
|------|--------|
| `alas.py:65-86,517-586` | Add hook calls in `run()` exception handler and `loop()`, add `_apply_directive()` method |
| `docker-compose.yml` | Replace with two-service compose (ALAS + sidecar) |

### Test Files

| File | What it tests |
|------|--------------|
| `ai_sidecar/tests/__init__.py` | Package init |
| `ai_sidecar/tests/conftest.py` | Test path configuration |
| `ai_sidecar/tests/test_models.py` | Pydantic model validation |
| `ai_sidecar/tests/test_state.py` | State persistence, atomic writes |
| `ai_sidecar/tests/test_directive_engine.py` | Directive validation logic |
| `ai_sidecar/tests/test_ai_client.py` | AI client with mocked HTTP |
| `ai_sidecar/tests/test_main.py` | FastAPI endpoint integration tests |
| `ai_sidecar/tests/test_hook.py` | AISidecarClient unit tests |

---

## Task 1: Sidecar Pydantic Models

**Files:**
- Create: `ai_sidecar/models.py`
- Create: `ai_sidecar/tests/__init__.py`
- Create: `ai_sidecar/tests/test_models.py`

- [ ] **Step 1: Write failing tests for models**

```python
# ai_sidecar/tests/test_models.py
import pytest
from models import (
    EventType, EventPayload, DirectiveAction, Directive,
    TaskInfo, GameState
)


class TestEventPayload:
    def test_cycle_start_valid(self):
        payload = EventPayload(
            event_type=EventType.CYCLE_START,
            pending_tasks=["Daily", "Commission"],
            waiting_tasks=["Exercise"],
            screenshot="base64data",
        )
        assert payload.event_type == EventType.CYCLE_START
        assert payload.pending_tasks == ["Daily", "Commission"]

    def test_task_complete_valid(self):
        payload = EventPayload(
            event_type=EventType.TASK_COMPLETE,
            task="Daily",
            success=True,
            duration=120.5,
        )
        assert payload.task == "Daily"
        assert payload.duration == 120.5

    def test_task_failed_valid(self):
        payload = EventPayload(
            event_type=EventType.TASK_FAILED,
            task="Exercise",
            error="task_returned_failure",
            count=2,
        )
        assert payload.count == 2

    def test_unknown_state_valid(self):
        payload = EventPayload(
            event_type=EventType.UNKNOWN_STATE,
            screenshot="base64data",
            click_history=["btn1", "btn2", "btn1"],
        )
        assert len(payload.click_history) == 3

    def test_screenshot_optional(self):
        payload = EventPayload(
            event_type=EventType.CYCLE_START,
            pending_tasks=["Daily"],
            waiting_tasks=[],
        )
        assert payload.screenshot is None


class TestDirective:
    def test_continue(self):
        d = Directive(action=DirectiveAction.CONTINUE)
        assert d.action == DirectiveAction.CONTINUE

    def test_reprioritize(self):
        d = Directive(
            action=DirectiveAction.REPRIORITIZE,
            task_order=["Event_A", "Daily"]
        )
        assert d.task_order == ["Event_A", "Daily"]

    def test_skip(self):
        d = Directive(action=DirectiveAction.SKIP, task="Exercise")
        assert d.task == "Exercise"

    def test_pause(self):
        d = Directive(
            action=DirectiveAction.PAUSE,
            reason="Event ending soon"
        )
        assert d.reason == "Event ending soon"

    def test_directive_from_json(self):
        d = Directive.model_validate_json(
            '{"action": "reprioritize", "task_order": ["A", "B"]}'
        )
        assert d.action == DirectiveAction.REPRIORITIZE


class TestGameState:
    def test_empty_state(self):
        state = GameState()
        assert state.pending_tasks == []
        assert state.completion_history == []
        assert state.failure_history == []

    def test_add_completion(self):
        state = GameState()
        state.completion_history.append(
            TaskInfo(task="Daily", success=True, duration=60.0)
        )
        assert len(state.completion_history) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/derek/code_projects/AzurLaneAutoScript/ai_sidecar && python -m pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'models'`

- [ ] **Step 3: Implement models**

```python
# ai_sidecar/models.py
from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class EventType(str, Enum):
    CYCLE_START = "cycle_start"
    TASK_COMPLETE = "task_complete"
    TASK_FAILED = "task_failed"
    UNKNOWN_STATE = "unknown_state"


class EventPayload(BaseModel):
    event_type: EventType
    # cycle_start fields
    pending_tasks: Optional[List[str]] = None
    waiting_tasks: Optional[List[str]] = None
    # task_complete / task_failed fields
    task: Optional[str] = None
    success: Optional[bool] = None
    duration: Optional[float] = None
    error: Optional[str] = None
    count: Optional[int] = None
    # shared fields
    screenshot: Optional[str] = None
    click_history: Optional[List[str]] = None


class DirectiveAction(str, Enum):
    CONTINUE = "continue"
    REPRIORITIZE = "reprioritize"
    SKIP = "skip"
    PAUSE = "pause"


class Directive(BaseModel):
    action: DirectiveAction
    task_order: Optional[List[str]] = None
    task: Optional[str] = None
    reason: Optional[str] = None


class TaskInfo(BaseModel):
    task: str
    success: bool
    duration: Optional[float] = None
    error: Optional[str] = None
    count: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class GameState(BaseModel):
    pending_tasks: List[str] = Field(default_factory=list)
    waiting_tasks: List[str] = Field(default_factory=list)
    completion_history: List[TaskInfo] = Field(default_factory=list)
    failure_history: List[TaskInfo] = Field(default_factory=list)
    last_screenshot: Optional[str] = None
    last_updated: Optional[datetime] = None
```

Also create:
- `ai_sidecar/__init__.py` (empty)
- `ai_sidecar/tests/__init__.py` (empty)
- `ai_sidecar/tests/conftest.py`:

```python
# ai_sidecar/tests/conftest.py
import sys
import os
# Add the ai_sidecar package root to sys.path so bare imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/derek/code_projects/AzurLaneAutoScript/ai_sidecar && python -m pytest tests/test_models.py -v`
Expected: All 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add ai_sidecar/models.py ai_sidecar/__init__.py ai_sidecar/tests/__init__.py ai_sidecar/tests/conftest.py ai_sidecar/tests/test_models.py
git commit -m "feat(sidecar): add Pydantic models for events, directives, and game state"
```

---

## Task 2: State Model with Atomic Persistence

**Files:**
- Create: `ai_sidecar/state.py`
- Create: `ai_sidecar/tests/test_state.py`

- [ ] **Step 1: Write failing tests**

```python
# ai_sidecar/tests/test_state.py
import json
import os
import tempfile

import pytest
from models import EventPayload, EventType, GameState
from state import StateManager


@pytest.fixture
def state_file():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


class TestStateManager:
    def test_initial_state_empty(self, state_file):
        sm = StateManager(state_file)
        assert sm.state.pending_tasks == []
        assert sm.state.completion_history == []

    def test_update_cycle_start(self, state_file):
        sm = StateManager(state_file)
        event = EventPayload(
            event_type=EventType.CYCLE_START,
            pending_tasks=["Daily", "Commission"],
            waiting_tasks=["Exercise"],
            screenshot="abc123",
        )
        sm.update(event)
        assert sm.state.pending_tasks == ["Daily", "Commission"]
        assert sm.state.waiting_tasks == ["Exercise"]
        assert sm.state.last_screenshot == "abc123"

    def test_update_task_complete(self, state_file):
        sm = StateManager(state_file)
        event = EventPayload(
            event_type=EventType.TASK_COMPLETE,
            task="Daily",
            success=True,
            duration=120.0,
        )
        sm.update(event)
        assert len(sm.state.completion_history) == 1
        assert sm.state.completion_history[0].task == "Daily"

    def test_update_task_failed(self, state_file):
        sm = StateManager(state_file)
        event = EventPayload(
            event_type=EventType.TASK_FAILED,
            task="Exercise",
            error="task_returned_failure",
            count=2,
        )
        sm.update(event)
        assert len(sm.state.failure_history) == 1
        assert sm.state.failure_history[0].count == 2

    def test_persistence_survives_reload(self, state_file):
        sm1 = StateManager(state_file)
        event = EventPayload(
            event_type=EventType.TASK_COMPLETE,
            task="Daily",
            success=True,
            duration=60.0,
        )
        sm1.update(event)

        sm2 = StateManager(state_file)
        assert len(sm2.state.completion_history) == 1
        assert sm2.state.completion_history[0].task == "Daily"

    def test_atomic_write_no_corruption(self, state_file):
        sm = StateManager(state_file)
        event = EventPayload(
            event_type=EventType.CYCLE_START,
            pending_tasks=["A"],
            waiting_tasks=["B"],
        )
        sm.update(event)
        # File should be valid JSON
        with open(state_file) as f:
            data = json.load(f)
        assert data["pending_tasks"] == ["A"]

    def test_history_capped(self, state_file):
        sm = StateManager(state_file, max_history=5)
        for i in range(10):
            event = EventPayload(
                event_type=EventType.TASK_COMPLETE,
                task=f"Task_{i}",
                success=True,
                duration=1.0,
            )
            sm.update(event)
        assert len(sm.state.completion_history) == 5
        assert sm.state.completion_history[0].task == "Task_5"

    def test_get_state_summary(self, state_file):
        sm = StateManager(state_file)
        event = EventPayload(
            event_type=EventType.CYCLE_START,
            pending_tasks=["Daily"],
            waiting_tasks=["Commission"],
        )
        sm.update(event)
        summary = sm.get_summary()
        assert "Daily" in summary
        assert isinstance(summary, str)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/derek/code_projects/AzurLaneAutoScript/ai_sidecar && python -m pytest tests/test_state.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'state'`

- [ ] **Step 3: Implement StateManager**

```python
# ai_sidecar/state.py
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
        if event.event_type == EventType.CYCLE_START:
            self.state.pending_tasks = event.pending_tasks or []
            self.state.waiting_tasks = event.waiting_tasks or []
            if event.screenshot is not None:
                self.state.last_screenshot = event.screenshot

        elif event.event_type == EventType.TASK_COMPLETE:
            info = TaskInfo(
                task=event.task,
                success=event.success or True,
                duration=event.duration,
            )
            self.state.completion_history.append(info)
            self.state.completion_history = self.state.completion_history[
                -self.max_history :
            ]

        elif event.event_type == EventType.TASK_FAILED:
            info = TaskInfo(
                task=event.task,
                success=False,
                error=event.error,
                count=event.count,
            )
            self.state.failure_history.append(info)
            self.state.failure_history = self.state.failure_history[
                -self.max_history :
            ]

        elif event.event_type == EventType.UNKNOWN_STATE:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/derek/code_projects/AzurLaneAutoScript/ai_sidecar && python -m pytest tests/test_state.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add ai_sidecar/state.py ai_sidecar/tests/test_state.py
git commit -m "feat(sidecar): add StateManager with atomic JSON persistence"
```

---

## Task 3: Directive Engine

**Files:**
- Create: `ai_sidecar/directive_engine.py`
- Create: `ai_sidecar/tests/test_directive_engine.py`

- [ ] **Step 1: Write failing tests**

```python
# ai_sidecar/tests/test_directive_engine.py
import json
import logging
import os
import tempfile

import pytest
from directive_engine import DirectiveEngine
from models import Directive, DirectiveAction


@pytest.fixture
def log_file():
    fd, path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def engine(log_file):
    return DirectiveEngine(
        decision_log=log_file,
        max_log_bytes=1024 * 1024,
        log_backup_count=2,
    )


class TestDirectiveEngine:
    def test_validate_continue(self, engine):
        d = Directive(action=DirectiveAction.CONTINUE)
        result = engine.validate(d, enabled_tasks=["Daily"])
        assert result.action == DirectiveAction.CONTINUE

    def test_validate_reprioritize_valid(self, engine):
        d = Directive(
            action=DirectiveAction.REPRIORITIZE,
            task_order=["Daily", "Commission"],
        )
        result = engine.validate(d, enabled_tasks=["Daily", "Commission", "Exercise"])
        assert result.action == DirectiveAction.REPRIORITIZE

    def test_validate_reprioritize_unknown_task(self, engine):
        d = Directive(
            action=DirectiveAction.REPRIORITIZE,
            task_order=["Daily", "FakeTask"],
        )
        result = engine.validate(d, enabled_tasks=["Daily", "Commission"])
        assert result.action == DirectiveAction.CONTINUE  # falls back

    def test_validate_reprioritize_disabled_task(self, engine):
        d = Directive(
            action=DirectiveAction.REPRIORITIZE,
            task_order=["Daily", "Exercise"],
        )
        # Exercise not in enabled list
        result = engine.validate(d, enabled_tasks=["Daily", "Commission"])
        assert result.action == DirectiveAction.CONTINUE

    def test_validate_skip_valid(self, engine):
        d = Directive(action=DirectiveAction.SKIP, task="Exercise")
        result = engine.validate(d, enabled_tasks=["Daily", "Exercise"])
        assert result.action == DirectiveAction.SKIP

    def test_validate_skip_unknown_task(self, engine):
        d = Directive(action=DirectiveAction.SKIP, task="FakeTask")
        result = engine.validate(d, enabled_tasks=["Daily"])
        assert result.action == DirectiveAction.CONTINUE

    def test_validate_pause_valid(self, engine):
        d = Directive(action=DirectiveAction.PAUSE, reason="Event ending")
        result = engine.validate(d, enabled_tasks=[])
        assert result.action == DirectiveAction.PAUSE

    def test_validate_pause_empty_reason(self, engine):
        d = Directive(action=DirectiveAction.PAUSE, reason="")
        result = engine.validate(d, enabled_tasks=[])
        assert result.action == DirectiveAction.CONTINUE

    def test_validate_pause_no_reason(self, engine):
        d = Directive(action=DirectiveAction.PAUSE, reason=None)
        result = engine.validate(d, enabled_tasks=[])
        assert result.action == DirectiveAction.CONTINUE

    def test_log_decision(self, engine, log_file):
        d = Directive(action=DirectiveAction.CONTINUE)
        engine.log_decision(
            event_type="cycle_start",
            ai_response='{"action": "continue"}',
            final_directive=d,
        )
        with open(log_file) as f:
            content = f.read()
        assert "cycle_start" in content
        assert "continue" in content

    def test_parse_ai_response_valid(self, engine):
        raw = '{"action": "skip", "task": "Daily"}'
        d = engine.parse_ai_response(raw)
        assert d.action == DirectiveAction.SKIP

    def test_parse_ai_response_invalid_json(self, engine):
        d = engine.parse_ai_response("not json at all")
        assert d.action == DirectiveAction.CONTINUE

    def test_parse_ai_response_invalid_action(self, engine):
        d = engine.parse_ai_response('{"action": "destroy_everything"}')
        assert d.action == DirectiveAction.CONTINUE

    def test_parse_ai_response_with_markdown(self, engine):
        raw = '```json\n{"action": "continue"}\n```'
        d = engine.parse_ai_response(raw)
        assert d.action == DirectiveAction.CONTINUE
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/derek/code_projects/AzurLaneAutoScript/ai_sidecar && python -m pytest tests/test_directive_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'directive_engine'`

- [ ] **Step 3: Implement DirectiveEngine**

```python
# ai_sidecar/directive_engine.py
import json
import logging
import re
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import List, Optional

from models import Directive, DirectiveAction

FALLBACK = Directive(action=DirectiveAction.CONTINUE)


class DirectiveEngine:
    def __init__(
        self,
        decision_log: str,
        max_log_bytes: int = 10 * 1024 * 1024,
        log_backup_count: int = 5,
    ):
        self.logger = logging.getLogger("directive_engine")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = RotatingFileHandler(
                decision_log,
                maxBytes=max_log_bytes,
                backupCount=log_backup_count,
            )
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(message)s")
            )
            self.logger.addHandler(handler)

    def validate(
        self, directive: Directive, enabled_tasks: List[str]
    ) -> Directive:
        if directive.action == DirectiveAction.CONTINUE:
            return directive

        if directive.action == DirectiveAction.REPRIORITIZE:
            if not directive.task_order:
                return FALLBACK
            for task in directive.task_order:
                if task not in enabled_tasks:
                    self.logger.warning(
                        f"Reprioritize references unknown/disabled task: {task}"
                    )
                    return FALLBACK
            return directive

        if directive.action == DirectiveAction.SKIP:
            if not directive.task or directive.task not in enabled_tasks:
                self.logger.warning(
                    f"Skip references unknown/disabled task: {directive.task}"
                )
                return FALLBACK
            return directive

        if directive.action == DirectiveAction.PAUSE:
            if not directive.reason:
                self.logger.warning("Pause directive missing reason")
                return FALLBACK
            return directive

        return FALLBACK

    def parse_ai_response(self, raw: str) -> Directive:
        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
        cleaned = cleaned.rstrip("`").strip()
        try:
            return Directive.model_validate_json(cleaned)
        except Exception:
            self.logger.warning(f"Failed to parse AI response: {raw[:200]}")
            return FALLBACK

    def log_decision(
        self,
        event_type: str,
        ai_response: str,
        final_directive: Directive,
        state_summary: Optional[str] = None,
    ) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "ai_response": ai_response[:500],
            "final_directive": final_directive.model_dump(),
        }
        if state_summary:
            entry["state_summary"] = state_summary[:500]
        self.logger.info(json.dumps(entry))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/derek/code_projects/AzurLaneAutoScript/ai_sidecar && python -m pytest tests/test_directive_engine.py -v`
Expected: All 14 tests PASS

- [ ] **Step 5: Commit**

```bash
git add ai_sidecar/directive_engine.py ai_sidecar/tests/test_directive_engine.py
git commit -m "feat(sidecar): add DirectiveEngine with validation and decision logging"
```

---

## Task 4: AI Client

**Files:**
- Create: `ai_sidecar/ai_client.py`
- Create: `ai_sidecar/tests/test_ai_client.py`

- [ ] **Step 1: Write failing tests**

```python
# ai_sidecar/tests/test_ai_client.py
from unittest.mock import MagicMock, patch

import pytest
from ai_client import AIClient


@pytest.fixture
def client():
    return AIClient(
        base_url="https://api.example.com/v1",
        api_key="test-key",
        model="test-model",
        timeout=5,
        max_retries=1,
    )


class TestAIClient:
    def test_init(self, client):
        assert client.model == "test-model"

    @patch("ai_client.OpenAI")
    def test_consult_success(self, mock_openai_cls, client):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        # Re-init to use mocked class
        client = AIClient(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="test-model",
            timeout=5,
            max_retries=1,
        )
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"action": "continue"}'
        mock_client.chat.completions.create.return_value = mock_response

        result = client.consult(
            system_prompt="You are a game AI.",
            user_message="What should I do?",
        )
        assert result == '{"action": "continue"}'
        mock_client.chat.completions.create.assert_called_once()

    @patch("ai_client.OpenAI")
    def test_consult_with_screenshot(self, mock_openai_cls, client):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        client = AIClient(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="test-model",
            timeout=5,
            max_retries=1,
        )
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"action": "continue"}'
        mock_client.chat.completions.create.return_value = mock_response

        result = client.consult(
            system_prompt="You are a game AI.",
            user_message="What should I do?",
            screenshot_base64="iVBORw0KGgo...",
        )
        assert result is not None
        # Verify the user message was constructed with image content
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        user_msg = messages[1]
        assert isinstance(user_msg["content"], list)

    @patch("ai_client.OpenAI")
    def test_consult_api_error_returns_none(self, mock_openai_cls, client):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        client = AIClient(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="test-model",
            timeout=5,
            max_retries=1,
        )
        mock_client.chat.completions.create.side_effect = Exception("API down")

        result = client.consult(
            system_prompt="test",
            user_message="test",
        )
        assert result is None

    @patch("ai_client.OpenAI")
    def test_consult_empty_response_returns_none(self, mock_openai_cls, client):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        client = AIClient(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="test-model",
            timeout=5,
            max_retries=1,
        )
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_client.chat.completions.create.return_value = mock_response

        result = client.consult(
            system_prompt="test",
            user_message="test",
        )
        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/derek/code_projects/AzurLaneAutoScript/ai_sidecar && python -m pytest tests/test_ai_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ai_client'`

- [ ] **Step 3: Implement AIClient**

```python
# ai_sidecar/ai_client.py
import logging
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


class AIClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int = 30,
        max_retries: int = 1,
    ):
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    def consult(
        self,
        system_prompt: str,
        user_message: str,
        screenshot_base64: Optional[str] = None,
    ) -> Optional[str]:
        try:
            messages = [
                {"role": "system", "content": system_prompt},
            ]

            if screenshot_base64:
                user_content = [
                    {"type": "text", "text": user_message},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{screenshot_base64}",
                            "detail": "low",
                        },
                    },
                ]
                messages.append({"role": "user", "content": user_content})
            else:
                messages.append({"role": "user", "content": user_message})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=256,
            )

            content = response.choices[0].message.content
            if not content:
                logger.warning("AI returned empty response")
                return None
            return content

        except Exception as e:
            logger.warning(f"AI consultation failed: {e}")
            return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/derek/code_projects/AzurLaneAutoScript/ai_sidecar && python -m pytest tests/test_ai_client.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add ai_sidecar/ai_client.py ai_sidecar/tests/test_ai_client.py
git commit -m "feat(sidecar): add AIClient with OpenAI-compatible API support"
```

---

## Task 5: Prompt Templates

**Files:**
- Create: `ai_sidecar/prompts/system.txt`
- Create: `ai_sidecar/prompts/user_template.txt`

- [ ] **Step 1: Create system prompt**

```text
# ai_sidecar/prompts/system.txt
You are an AI assistant managing task prioritization for Azur Lane Auto Script (ALAS), a game automation tool for the mobile game Azur Lane.

## Game Context
Azur Lane is a side-scrolling shoot 'em up mobile game. ALAS automates repetitive tasks:
- **Daily missions**: Reset every day at server reset time. Must complete before reset.
- **Commissions**: Timed expeditions that yield resources. Should be collected and re-dispatched promptly.
- **Exercises**: PvP battles with limited daily attempts. Reset at specific times.
- **Events**: Time-limited content with deadlines. Highest priority when active.
- **Campaign**: Main story battles for leveling and drops.
- **Research**: Equipment blueprints from time-gated research projects.
- **Tactical**: Skill training for ships.
- **Dorm**: Morale and experience management.
- **Meowfficer**: Cat-based buffs.
- **Guild**: Guild contributions and logistics.
- **Reward**: Collecting mail and mission rewards.
- **Shop**: Purchasing items from various shops.

## Resources
- **Oil**: Primary stamina resource. Consumed by most combat tasks.
- **Coins**: Currency for upgrades and purchases.
- **Cubes**: Gacha currency for building ships. Scarce and valuable.

## Your Role
You receive game state events and decide task execution order. Your decisions should:
1. Prioritize time-limited content (events ending soon, daily tasks before reset)
2. Keep commissions cycling (collect + redispatch is high value per effort)
3. Deprioritize tasks that have been failing (likely a bug, don't retry endlessly)
4. Consider resource efficiency (don't grind campaign when oil is low)

## Output Format
Respond with ONLY a JSON object matching one of these formats. No markdown, no explanation, no extra text.

Continue (no changes):
{"action": "continue"}

Reorder task queue (only reference enabled tasks from the provided list):
{"action": "reprioritize", "task_order": ["TaskA", "TaskB", "TaskC"]}

Skip a task this cycle:
{"action": "skip", "task": "TaskName"}

Pause automation (emergency only):
{"action": "pause", "reason": "Brief explanation"}

## Constraints
- Only reference task names from the provided enabled task list
- Never invent task names
- Prefer "continue" when the current order is already reasonable
- Use "pause" sparingly — only for situations requiring human intervention
```

- [ ] **Step 2: Create user message template**

```text
# ai_sidecar/prompts/user_template.txt
## Event
Type: {event_type}
{event_details}

## Current Game State
{state_summary}

## Enabled Tasks
{enabled_tasks}

## Time Context
Server time: {server_time}

Please decide the best action.
```

- [ ] **Step 3: Commit**

```bash
git add ai_sidecar/prompts/system.txt ai_sidecar/prompts/user_template.txt
git commit -m "feat(sidecar): add AI prompt templates for task prioritization"
```

---

## Task 6: FastAPI Application

**Files:**
- Create: `ai_sidecar/main.py`
- Create: `ai_sidecar/config.yaml`
- Create: `ai_sidecar/tests/test_main.py`

- [ ] **Step 1: Write failing tests**

```python
# ai_sidecar/tests/test_main.py
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def state_file():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def log_file():
    fd, path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def app(state_file, log_file, tmp_path):
    # Write minimal config
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "system.txt").write_text("You are a test AI.")
    (prompts_dir / "user_template.txt").write_text(
        "Event: {event_type}\n{event_details}\n{state_summary}\n{enabled_tasks}\n{server_time}"
    )

    env = {
        "AI_API_KEY": "test-key",
        "AI_BASE_URL": "https://api.example.com/v1",
        "AI_MODEL": "test-model",
        "SIDECAR_STATE_FILE": state_file,
        "SIDECAR_DECISION_LOG": log_file,
        "SIDECAR_PROMPTS_DIR": str(prompts_dir),
        "SIDECAR_CONSULT_ON": "cycle_start,task_failed,unknown_state",
        "SIDECAR_COOLDOWN": "0",
    }
    with patch.dict(os.environ, env):
        # Must import after env is set
        import importlib
        import main as main_module
        importlib.reload(main_module)
        yield main_module.app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "api_version" in data

    def test_health_has_version(self, client):
        resp = client.get("/health")
        assert resp.json()["api_version"] == "1.0"


class TestEventsEndpoint:
    def test_cycle_start_no_ai(self, client):
        """cycle_start with AI mocked to return continue"""
        with patch("main.ai_client") as mock_ai:
            mock_ai.consult.return_value = '{"action": "continue"}'
            resp = client.post("/api/events", json={
                "event_type": "cycle_start",
                "pending_tasks": ["Daily", "Commission"],
                "waiting_tasks": ["Exercise"],
            })
        assert resp.status_code == 200
        assert resp.json()["action"] == "continue"

    def test_task_complete_no_consult(self, client):
        """task_complete should update state but not consult AI by default"""
        with patch("main.ai_client") as mock_ai:
            resp = client.post("/api/events", json={
                "event_type": "task_complete",
                "task": "Daily",
                "success": True,
                "duration": 120.0,
            })
            mock_ai.consult.assert_not_called()
        assert resp.status_code == 200
        assert resp.json()["action"] == "continue"

    def test_task_failed_consults_ai(self, client):
        with patch("main.ai_client") as mock_ai:
            mock_ai.consult.return_value = '{"action": "skip", "task": "Exercise"}'
            resp = client.post("/api/events", json={
                "event_type": "task_failed",
                "task": "Exercise",
                "error": "task_returned_failure",
                "count": 2,
            })
        assert resp.status_code == 200
        # May be continue if Exercise not in enabled tasks, but endpoint returns something valid
        assert resp.json()["action"] in ["skip", "continue"]

    def test_invalid_event_type(self, client):
        resp = client.post("/api/events", json={
            "event_type": "nonexistent",
        })
        assert resp.status_code == 422

    def test_ai_unreachable_returns_continue(self, client):
        with patch("main.ai_client") as mock_ai:
            mock_ai.consult.return_value = None
            resp = client.post("/api/events", json={
                "event_type": "cycle_start",
                "pending_tasks": ["Daily"],
                "waiting_tasks": [],
            })
        assert resp.status_code == 200
        assert resp.json()["action"] == "continue"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/derek/code_projects/AzurLaneAutoScript/ai_sidecar && python -m pytest tests/test_main.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'main'`

- [ ] **Step 3: Implement FastAPI app**

```python
# ai_sidecar/main.py
import os
import time
from datetime import datetime

from fastapi import FastAPI

from ai_client import AIClient
from directive_engine import DirectiveEngine
from models import Directive, DirectiveAction, EventPayload, EventType
from state import StateManager

# Config from environment
AI_BASE_URL = os.environ.get("AI_BASE_URL", "https://api.openai.com/v1")
AI_API_KEY = os.environ.get("AI_API_KEY", "")
AI_MODEL = os.environ.get("AI_MODEL", "gpt-4o")
AI_TIMEOUT = int(os.environ.get("AI_TIMEOUT", "30"))
AI_MAX_RETRIES = int(os.environ.get("AI_MAX_RETRIES", "1"))

STATE_FILE = os.environ.get("SIDECAR_STATE_FILE", "/data/state.json")
DECISION_LOG = os.environ.get("SIDECAR_DECISION_LOG", "/data/decisions.log")
PROMPTS_DIR = os.environ.get("SIDECAR_PROMPTS_DIR", "/app/prompts")
CONSULT_ON = os.environ.get("SIDECAR_CONSULT_ON", "task_failed,cycle_start,unknown_state").split(",")
COOLDOWN = int(os.environ.get("SIDECAR_COOLDOWN", "60"))

# Validate consult_on
VALID_EVENTS = {e.value for e in EventType}
for event in CONSULT_ON:
    if event.strip() not in VALID_EVENTS:
        raise ValueError(f"Invalid event type in SIDECAR_CONSULT_ON: '{event}'. Valid: {VALID_EVENTS}")

# Load prompt templates
def _load_prompt(name: str) -> str:
    path = os.path.join(PROMPTS_DIR, name)
    with open(path, "r") as f:
        return f.read()

system_prompt = _load_prompt("system.txt")
user_template = _load_prompt("user_template.txt")

# Init components
state_manager = StateManager(STATE_FILE)
directive_engine = DirectiveEngine(decision_log=DECISION_LOG)
ai_client = AIClient(
    base_url=AI_BASE_URL,
    api_key=AI_API_KEY,
    model=AI_MODEL,
    timeout=AI_TIMEOUT,
    max_retries=AI_MAX_RETRIES,
)

FALLBACK = Directive(action=DirectiveAction.CONTINUE)
_last_consult_time = 0.0

app = FastAPI(title="ALAS AI Sidecar", version="1.0")


@app.get("/health")
def health():
    return {"status": "ok", "api_version": "1.0"}


@app.post("/api/events")
def handle_event(event: EventPayload) -> dict:
    global _last_consult_time

    # Update state
    state_manager.update(event)

    # Determine if we should consult AI
    should_consult = event.event_type.value in CONSULT_ON

    # Debounce cycle_start
    if event.event_type == EventType.CYCLE_START and should_consult:
        now = time.time()
        if now - _last_consult_time < COOLDOWN:
            should_consult = False

    if not should_consult:
        return FALLBACK.model_dump()

    # Build user message
    event_details = _format_event_details(event)
    state_summary = state_manager.get_summary()
    enabled_tasks = state_manager.state.pending_tasks + state_manager.state.waiting_tasks

    user_message = user_template.format(
        event_type=event.event_type.value,
        event_details=event_details,
        state_summary=state_summary,
        enabled_tasks=", ".join(enabled_tasks) if enabled_tasks else "none",
        server_time=datetime.now().isoformat(),
    )

    # Consult AI
    screenshot = event.screenshot or state_manager.state.last_screenshot
    raw_response = ai_client.consult(
        system_prompt=system_prompt,
        user_message=user_message,
        screenshot_base64=screenshot,
    )

    _last_consult_time = time.time()

    if raw_response is None:
        return FALLBACK.model_dump()

    # Parse and validate
    directive = directive_engine.parse_ai_response(raw_response)
    directive = directive_engine.validate(directive, enabled_tasks=enabled_tasks)

    # Log decision
    directive_engine.log_decision(
        event_type=event.event_type.value,
        ai_response=raw_response,
        final_directive=directive,
        state_summary=state_summary,
    )

    return directive.model_dump()


def _format_event_details(event: EventPayload) -> str:
    if event.event_type == EventType.CYCLE_START:
        return (
            f"Scheduler cycle starting.\n"
            f"Pending: {', '.join(event.pending_tasks or [])}\n"
            f"Waiting: {', '.join(event.waiting_tasks or [])}"
        )
    elif event.event_type == EventType.TASK_COMPLETE:
        dur = f" in {event.duration:.0f}s" if event.duration else ""
        return f"Task '{event.task}' completed successfully{dur}."
    elif event.event_type == EventType.TASK_FAILED:
        return f"Task '{event.task}' failed. Error: {event.error}. Failure count: {event.count}."
    elif event.event_type == EventType.UNKNOWN_STATE:
        history = ", ".join(event.click_history or [])
        return f"Game stuck. Recent clicks: {history}"
    return ""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/derek/code_projects/AzurLaneAutoScript/ai_sidecar && python -m pytest tests/test_main.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Create config.yaml**

```yaml
# ai_sidecar/config.yaml
# This file documents the available configuration.
# All config is read from environment variables at startup.
#
# AI_BASE_URL: OpenAI-compatible API base URL (default: https://api.openai.com/v1)
# AI_API_KEY: API key for the AI provider
# AI_MODEL: Model name (default: gpt-4o)
# AI_TIMEOUT: Request timeout in seconds (default: 30)
# AI_MAX_RETRIES: Max retries on failure (default: 1)
#
# SIDECAR_STATE_FILE: Path to state persistence file (default: /data/state.json)
# SIDECAR_DECISION_LOG: Path to decision log file (default: /data/decisions.log)
# SIDECAR_PROMPTS_DIR: Path to prompt template directory (default: /app/prompts)
# SIDECAR_CONSULT_ON: Comma-separated event types that trigger AI consultation
#                     (default: task_failed,cycle_start,unknown_state)
# SIDECAR_COOLDOWN: Minimum seconds between AI consultations on cycle_start (default: 60)
```

- [ ] **Step 6: Commit**

```bash
git add ai_sidecar/main.py ai_sidecar/config.yaml ai_sidecar/tests/test_main.py
git commit -m "feat(sidecar): add FastAPI app with /health and /api/events endpoints"
```

---

## Task 7: Sidecar Docker Setup

**Files:**
- Create: `ai_sidecar/requirements.txt`
- Create: `ai_sidecar/Dockerfile`

- [ ] **Step 1: Create requirements.txt**

```text
# ai_sidecar/requirements.txt
fastapi==0.115.6
uvicorn[standard]==0.34.0
openai==1.58.1
pydantic==2.10.4
```

- [ ] **Step 2: Create Dockerfile**

```dockerfile
# ai_sidecar/Dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8484

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8484", "--workers", "1"]
```

- [ ] **Step 3: Verify sidecar builds**

Run: `cd /Users/derek/code_projects/AzurLaneAutoScript && docker build -t alas-sidecar-test ./ai_sidecar`
Expected: Successful build

- [ ] **Step 4: Commit**

```bash
git add ai_sidecar/requirements.txt ai_sidecar/Dockerfile
git commit -m "feat(sidecar): add Dockerfile and requirements for sidecar container"
```

---

## Task 8: ALAS Hook Layer

**Files:**
- Create: `module/ai_hook/__init__.py`
- Create: `module/ai_hook/hook.py`
- Create: `ai_sidecar/tests/test_hook.py`

- [ ] **Step 1: Write failing tests**

```python
# ai_sidecar/tests/test_hook.py
"""
Tests for the ALAS hook layer.
Note: This code must be Python 3.7 compatible, but we test it in the sidecar's
Python 3.11 environment for convenience. The tests verify the logic; actual
3.7 compatibility is verified by running ALAS.
"""
import base64
import json
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# We need to add the module path for the hook
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "module", "ai_hook"))

from hook import AISidecarClient


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.AiSidecar_Enabled = True
    config.AiSidecar_SidecarUrl = "http://localhost:8484"
    return config


@pytest.fixture
def client(mock_config):
    return AISidecarClient(mock_config)


class TestScreenshotToBase64:
    def test_converts_numpy_array(self, client):
        # Create a small test image (10x10 black image)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        result = client.screenshot_to_base64(img)
        assert isinstance(result, str)
        # Should be valid base64
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_none_image_returns_none(self, client):
        result = client.screenshot_to_base64(None)
        assert result is None


class TestNotify:
    @patch("hook.requests.post")
    def test_notify_success(self, mock_post, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"action": "continue"}
        mock_post.return_value = mock_resp

        result = client.notify("task_complete", {"task": "Daily", "success": True})
        assert result is not None
        assert result["action"] == "continue"
        mock_post.assert_called_once()

    @patch("hook.requests.post")
    def test_notify_sidecar_down(self, mock_post, client):
        mock_post.side_effect = Exception("Connection refused")
        result = client.notify("task_complete", {"task": "Daily"})
        assert result is None

    @patch("hook.requests.post")
    def test_notify_non_200_returns_none(self, mock_post, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_post.return_value = mock_resp
        result = client.notify("task_complete", {"task": "Daily"})
        assert result is None

    @patch("hook.requests.post")
    def test_debounce_cycle_start(self, mock_post, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"action": "continue"}
        mock_post.return_value = mock_resp

        # First call should go through
        client.notify("cycle_start", {"pending_tasks": []})
        assert mock_post.call_count == 1

        # Second call within cooldown should be skipped
        # (default cooldown is 60s, so immediate call is within it)
        result = client.notify("cycle_start", {"pending_tasks": []})
        assert mock_post.call_count == 1  # not called again
        assert result is None

    @patch("hook.requests.post")
    def test_no_debounce_on_other_events(self, mock_post, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"action": "continue"}
        mock_post.return_value = mock_resp

        client.notify("task_failed", {"task": "A"})
        client.notify("task_failed", {"task": "B"})
        assert mock_post.call_count == 2


class TestDisabled:
    def test_disabled_returns_none(self):
        config = MagicMock()
        config.AiSidecar_Enabled = False
        c = AISidecarClient(config)
        result = c.notify("cycle_start", {})
        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/derek/code_projects/AzurLaneAutoScript/ai_sidecar && pip install numpy && python -m pytest tests/test_hook.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hook'`

- [ ] **Step 3: Implement AISidecarClient**

```python
# module/ai_hook/__init__.py
```

```python
# module/ai_hook/hook.py
"""
AI Sidecar hook layer for ALAS.

IMPORTANT: This file runs inside the ALAS container (Python 3.7).
Do NOT use 3.8+ features:
- No walrus operator (:=)
- No typing.Literal
- Use typing.Optional, not X | None
- No positional-only parameters (/)
"""
import base64
import logging
import time

import cv2
import requests

logger = logging.getLogger(__name__)

# Typing imports compatible with 3.7
from typing import Any, Dict, Optional


class AISidecarClient:
    def __init__(self, config):
        self.enabled = getattr(config, 'AiSidecar_Enabled', False)
        self.url = getattr(config, 'AiSidecar_SidecarUrl', 'http://sidecar:8484')
        self._last_cycle_start = 0.0
        self._cooldown = 60  # seconds

    def screenshot_to_base64(self, image):
        # type: (Any) -> Optional[str]
        if image is None:
            return None
        try:
            _, buffer = cv2.imencode('.png', image)
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            logger.warning('Failed to encode screenshot: %s', e)
            return None

    def notify(self, event_type, payload):
        # type: (str, Dict[str, Any]) -> Optional[Dict[str, Any]]
        if not self.enabled:
            return None

        # Debounce cycle_start
        if event_type == 'cycle_start':
            now = time.time()
            if now - self._last_cycle_start < self._cooldown:
                return None
            self._last_cycle_start = now

        try:
            payload['event_type'] = event_type
            resp = requests.post(
                self.url + '/api/events',
                json=payload,
                timeout=(5, 30),
            )
            if resp.status_code == 200:
                return resp.json()
            else:
                logger.warning(
                    'Sidecar returned status %d for event %s',
                    resp.status_code,
                    event_type,
                )
                return None
        except Exception as e:
            logger.warning('Sidecar unreachable for event %s: %s', event_type, e)
            return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/derek/code_projects/AzurLaneAutoScript/ai_sidecar && python -m pytest tests/test_hook.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Verify `requests` is available in ALAS**

`requests==2.18.4` is already in `requirements.txt` (line 72) and is installed by the ALAS Dockerfile. No changes needed. Verify:

Run: `grep requests /Users/derek/code_projects/AzurLaneAutoScript/requirements.txt`
Expected: `requests==2.18.4` appears in output

- [ ] **Step 6: Commit**

```bash
git add module/ai_hook/__init__.py module/ai_hook/hook.py ai_sidecar/tests/test_hook.py
git commit -m "feat(hook): add AISidecarClient with debounce and screenshot encoding"
```

---

## Task 9: Add ALAS Config Section for AI Sidecar

**Files:**
- Modify: `module/config/config_generated.py` (or equivalent config definition)

The ALAS config system uses generated config classes. We need to add `AiSidecar_Enabled` and `AiSidecar_SidecarUrl` so they are accessible via `self.config`.

- [ ] **Step 1: Find the config definition location**

Run: `grep -rn "Enabled\|SidecarUrl\|class GeneratedConfig\|class ManualConfig" /Users/derek/code_projects/AzurLaneAutoScript/module/config/ | head -20`

Look for where config attributes like `Scheduler_Enabled` or similar boolean config fields are defined to follow the same pattern.

- [ ] **Step 2: Add AiSidecar config section**

Follow the existing pattern for config sections. Add to the appropriate config class (likely `ManualConfig` in `module/config/manual_config.py` or the generated config):

```python
# AiSidecar
AiSidecar_Enabled = False
AiSidecar_SidecarUrl = 'http://sidecar:8484'
```

- [ ] **Step 3: Verify config is accessible**

Run: `cd /Users/derek/code_projects/AzurLaneAutoScript && python -c "from module.config.config import AzurLaneConfig; c = AzurLaneConfig('alas'); print(c.AiSidecar_Enabled)"`
Expected: `False` (the default)

- [ ] **Step 4: Commit**

```bash
git add module/config/
git commit -m "feat(config): add AiSidecar config section (Enabled, SidecarUrl)"
```

---

## Task 10: Integrate Hook into ALAS Main Loop

**Files:**
- Modify: `alas.py:1-19,65-84,517-586`

This task modifies the existing ALAS code. Changes are minimal and fail-safe.

- [ ] **Step 1: Add import and `_apply_directive` method to `alas.py`**

Add after existing imports (after line 15):

```python
# After line 15 in alas.py
from module.ai_hook.hook import AISidecarClient
```

Add the `_apply_directive` method to the `AzurLaneAutoScript` class (after `__init__`, around line 28):

```python
    def _apply_directive(self, directive):
        """Apply an AI sidecar directive to the scheduler."""
        if directive is None:
            return
        action = directive.get('action')
        if action == 'reprioritize':
            task_order = directive.get('task_order', [])
            now = datetime.now()
            # Batch: set all next_run times directly, then save once
            for i, task_name in enumerate(task_order):
                target = now + timedelta(seconds=i + 1)
                key = f'{task_name}.Scheduler.NextRun'
                self.config.modified[key] = target
                logger.info(f'AI sidecar: reprioritize `{task_name}` to {target}')
            try:
                self.config.update()
            except Exception as e:
                logger.warning(f'Failed to apply reprioritize: {e}')
        elif action == 'skip':
            task_name = directive.get('task')
            if task_name:
                try:
                    self.config.task_delay(minute=1440, task=task_name)
                    logger.info(f'AI sidecar: skipping task `{task_name}` for 24h')
                except Exception as e:
                    logger.warning(f'Failed to skip task {task_name}: {e}')
        elif action == 'pause':
            reason = directive.get('reason', 'AI requested pause')
            logger.warning(f'AI sidecar: pausing automation — {reason}')
            if self.stop_event is not None:
                self.stop_event.set()
            else:
                logger.warning('Pause requested but stop_event not available (standalone mode)')
```

- [ ] **Step 2: Add hook into `run()` exception handler**

In `alas.py` line 77-84 (the `GameStuckError, GameTooManyClickError` handler), add sidecar notification before the existing logic. The modified block should be:

```python
        except (GameStuckError, GameTooManyClickError) as e:
            logger.error(e)
            self.save_error_log()
            # Notify AI sidecar of stuck state
            if hasattr(self, '_sidecar') and self._sidecar:
                try:
                    directive = self._sidecar.notify("unknown_state", {
                        "screenshot": self._sidecar.screenshot_to_base64(self.device.image),
                        "click_history": [str(c) for c in self.device.click_record],
                    })
                    self._apply_directive(directive)
                except Exception:
                    pass  # sidecar errors must never break ALAS
            logger.warning(f'Game stuck, {self.device.package} will be restarted in 10 seconds')
            logger.warning('If you are playing by hand, please stop Alas')
            self.config.task_call('Restart')
            self.device.sleep(10)
            return False
```

- [ ] **Step 3: Add hooks into `loop()` method**

In `alas.py` `loop()` method, add sidecar initialization and event hooks. Insert sidecar init after line 519 (`logger.info` line), and hooks at the appropriate points:

After line 519 (setup):
```python
        # Init AI sidecar
        self._sidecar = None
        try:
            if getattr(self.config, 'AiSidecar_Enabled', False):
                self._sidecar = AISidecarClient(self.config)
                logger.info('AI sidecar enabled')
        except Exception as e:
            logger.warning(f'AI sidecar init failed: {e}')
```

AFTER line 539 (`task = self.get_next_task()`), add cycle_start hook. This placement is critical — `get_next_task()` populates `self.config.pending_task` and `self.config.waiting_task`. Before that call, they are empty lists. The AI's reprioritization is applied before `self.run()`, so it takes effect for the current cycle:
```python
            # AI sidecar: cycle_start (must be AFTER get_next_task which populates task lists)
            if self._sidecar:
                try:
                    screenshot = None
                    try:
                        self.device.screenshot()
                        screenshot = self._sidecar.screenshot_to_base64(self.device.image)
                    except Exception:
                        pass
                    directive = self._sidecar.notify("cycle_start", {
                        "pending_tasks": [str(t) for t in self.config.pending_task],
                        "waiting_tasks": [str(t) for t in self.config.waiting_task],
                        "screenshot": screenshot,
                    })
                    if directive and directive.get('action') == 'reprioritize':
                        self._apply_directive(directive)
                        # Re-fetch task after reprioritization
                        task = self.get_next_task()
                    else:
                        self._apply_directive(directive)
                except Exception:
                    pass
```

After line 556 (`logger.info(f'Scheduler: End task')`) and before line 559 (failure checking), add task_complete/task_failed hooks. Wrap the timing around the `run()` call:

Replace line 555 (`success = self.run(...)`) with:
```python
            _task_start = time.time()
            success = self.run(inflection.underscore(task))
            _task_duration = time.time() - _task_start
```

After the `self.is_first_task = False` line, before failure checking:
```python
            # AI sidecar: task result
            if self._sidecar:
                try:
                    if success:
                        self._sidecar.notify("task_complete", {
                            "task": task,
                            "success": True,
                            "duration": _task_duration,
                        })
                    else:
                        directive = self._sidecar.notify("task_failed", {
                            "task": task,
                            "error": "task_returned_failure",
                            "count": deep_get(self.failure_record, keys=task, default=0),
                        })
                        self._apply_directive(directive)
                except Exception:
                    pass
```

`import time` is already at line 4 of `alas.py`, no change needed.

- [ ] **Step 4: Verify ALAS still imports cleanly**

Run: `cd /Users/derek/code_projects/AzurLaneAutoScript && python -c "from alas import AzurLaneAutoScript; print('OK')"`
Expected: `OK` (or an import error from a missing deep dependency like `uiautomator2`, which is fine — it means our code parsed correctly. The key is no `SyntaxError` or `ImportError` from our new code.)

- [ ] **Step 5: Commit**

```bash
git add alas.py
git commit -m "feat(hook): integrate AI sidecar hooks into ALAS scheduler loop"
```

---

## Task 11: Docker Compose and Environment

**Files:**
- Modify: `docker-compose.yml`
- Create: `.env.example`

- [ ] **Step 1: Update docker-compose.yml**

Replace the entire file with:

```yaml
# docker-compose.yml
version: '3.7'
services:
  alas:
    build:
      context: ./deploy/docker/
      dockerfile: ./Dockerfile
    ports:
      - "22267:22267"
    volumes:
      - '.:/app/AzurLaneAutoScript:rw'
      - '/etc/localtime:/etc/localtime:ro'
    environment:
      - AI_SIDECAR_URL=http://sidecar:8484
      - EMULATOR_SERIAL=${EMULATOR_SERIAL:-host.docker.internal:5555}
    extra_hosts:
      - "host.docker.internal:host-gateway"
    depends_on:
      sidecar:
        condition: service_healthy
    container_name: alas

  sidecar:
    build:
      context: ./ai_sidecar
      dockerfile: Dockerfile
    ports:
      - "8484:8484"
    volumes:
      - sidecar-data:/data
      - ./ai_sidecar/prompts:/app/prompts:ro
    environment:
      - AI_API_KEY=${AI_API_KEY:-}
      - AI_BASE_URL=${AI_BASE_URL:-https://api.openai.com/v1}
      - AI_MODEL=${AI_MODEL:-gpt-4o}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8484/health"]
      interval: 10s
      timeout: 5s
      retries: 3
    container_name: alas-sidecar

volumes:
  sidecar-data:
```

Note: `env_file` removed — all config uses `environment` with defaults so the compose works without a `.env` file. Users who want to use a `.env` file can add `env_file: [.env]` or export the variables in their shell. This avoids breaking existing users who run `docker-compose up` without a `.env`.

- [ ] **Step 2: Create .env.example**

```bash
# .env.example
# AI Provider Configuration
# Supports any OpenAI-compatible API (OpenAI, Ollama, LM Studio, vLLM, etc.)
AI_API_KEY=your-api-key-here
AI_BASE_URL=https://api.openai.com/v1
AI_MODEL=gpt-4o

# Emulator Configuration
# EMULATOR_SERIAL=host.docker.internal:5555
```

- [ ] **Step 3: Add `.env` to `.gitignore`**

Verify `.env` is already in `.gitignore`. If not, append it:

Run: `grep -q '^\.env$' /Users/derek/code_projects/AzurLaneAutoScript/.gitignore && echo "already there" || echo ".env" >> /Users/derek/code_projects/AzurLaneAutoScript/.gitignore`

- [ ] **Step 4: Verify docker-compose config is valid**

Run: `cd /Users/derek/code_projects/AzurLaneAutoScript && docker compose config --quiet`
Expected: No errors (exit code 0)

- [ ] **Step 5: Commit**

```bash
git add docker-compose.yml .env.example .gitignore
git commit -m "feat(docker): add two-service compose with AI sidecar and macOS support"
```

---

## Task 12: Integration Smoke Test

**Files:** No new files — verifies the full stack works together.

- [ ] **Step 1: Run all sidecar unit tests**

Run: `cd /Users/derek/code_projects/AzurLaneAutoScript/ai_sidecar && pip install -r requirements.txt && pip install pytest numpy opencv-python-headless && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Build and start sidecar container**

Run: `cd /Users/derek/code_projects/AzurLaneAutoScript && docker compose build sidecar && docker compose up -d sidecar`
Expected: Container starts and becomes healthy

- [ ] **Step 3: Test health endpoint**

Run: `curl -s http://localhost:8484/health | python -m json.tool`
Expected:
```json
{
    "status": "ok",
    "api_version": "1.0"
}
```

- [ ] **Step 4: Test event endpoint with mock event**

Run:
```bash
curl -s -X POST http://localhost:8484/api/events \
  -H "Content-Type: application/json" \
  -d '{"event_type": "cycle_start", "pending_tasks": ["Daily", "Commission"], "waiting_tasks": ["Exercise"]}' \
  | python -m json.tool
```
Expected: `{"action": "continue"}` (or a real directive if AI_API_KEY is configured)

- [ ] **Step 5: Clean up**

Run: `cd /Users/derek/code_projects/AzurLaneAutoScript && docker compose down`

- [ ] **Step 6: Commit (if any test-related fixes were needed)**

```bash
git add -A
git commit -m "fix: address integration test findings"
```

---

## Task Summary

| Task | Description | Depends On |
|------|-------------|-----------|
| 1 | Pydantic models | — |
| 2 | State manager | 1 |
| 3 | Directive engine | 1 |
| 4 | AI client | — |
| 5 | Prompt templates | — |
| 6 | FastAPI app | 1, 2, 3, 4, 5 |
| 7 | Sidecar Docker | 6 |
| 8 | ALAS hook layer | 1 |
| 9 | ALAS config section | — |
| 10 | ALAS main loop integration | 8, 9 |
| 11 | Docker Compose + env | 7, 10 |
| 12 | Integration smoke test | 11 |

**Parallelizable groups:**
- Tasks 1, 4, 5, 9 can run in parallel (no dependencies)
- Tasks 2, 3 can run in parallel (both depend only on 1)
- Tasks 8 can run in parallel with 6, 7 (independent codebases)
