import json
import pytest
from models import (
    EventType,
    EventPayload,
    DirectiveAction,
    Directive,
    TaskInfo,
    GameState,
)


# ---------------------------------------------------------------------------
# EventPayload tests
# ---------------------------------------------------------------------------

class TestEventPayload:
    def test_cycle_start(self):
        payload = EventPayload(
            event_type=EventType.cycle_start,
            pending_tasks=["Task1", "Task2"],
            waiting_tasks=["Task3"],
        )
        assert payload.event_type == EventType.cycle_start
        assert payload.pending_tasks == ["Task1", "Task2"]
        assert payload.waiting_tasks == ["Task3"]
        assert payload.task is None

    def test_task_complete(self):
        payload = EventPayload(
            event_type=EventType.task_complete,
            task="Task1",
            success=True,
            duration=5.3,
        )
        assert payload.event_type == EventType.task_complete
        assert payload.task == "Task1"
        assert payload.success is True
        assert payload.duration == pytest.approx(5.3)
        assert payload.error is None

    def test_task_failed(self):
        payload = EventPayload(
            event_type=EventType.task_failed,
            task="Task2",
            success=False,
            duration=1.0,
            error="Timeout",
        )
        assert payload.event_type == EventType.task_failed
        assert payload.success is False
        assert payload.error == "Timeout"

    def test_unknown_state(self):
        payload = EventPayload(
            event_type=EventType.unknown_state,
            count=3,
            screenshot="base64encodeddata",
            click_history=["100,200", "300,400"],
        )
        assert payload.event_type == EventType.unknown_state
        assert payload.count == 3
        assert payload.screenshot == "base64encodeddata"
        assert len(payload.click_history) == 2

    def test_screenshot_can_be_none(self):
        payload = EventPayload(
            event_type=EventType.unknown_state,
            count=1,
        )
        assert payload.screenshot is None


# ---------------------------------------------------------------------------
# Directive tests
# ---------------------------------------------------------------------------

class TestDirective:
    def test_continue_directive(self):
        directive = Directive(action=DirectiveAction.continue_)
        assert directive.action == DirectiveAction.continue_
        assert directive.task_order is None
        assert directive.task is None
        assert directive.reason is None

    def test_reprioritize_directive(self):
        directive = Directive(
            action=DirectiveAction.reprioritize,
            task_order=["Task2", "Task1", "Task3"],
            reason="Task2 is time-sensitive",
        )
        assert directive.action == DirectiveAction.reprioritize
        assert directive.task_order == ["Task2", "Task1", "Task3"]
        assert directive.reason == "Task2 is time-sensitive"

    def test_skip_directive(self):
        directive = Directive(
            action=DirectiveAction.skip,
            task="Task1",
            reason="Task1 is not worth running now",
        )
        assert directive.action == DirectiveAction.skip
        assert directive.task == "Task1"

    def test_pause_directive(self):
        directive = Directive(
            action=DirectiveAction.pause,
            reason="Server maintenance",
        )
        assert directive.action == DirectiveAction.pause

    def test_directive_deserialize_from_json(self):
        raw = json.dumps({
            "action": "reprioritize",
            "task_order": ["B", "A"],
            "reason": "B first",
        })
        directive = Directive.model_validate_json(raw)
        assert directive.action == DirectiveAction.reprioritize
        assert directive.task_order == ["B", "A"]
        assert directive.reason == "B first"

    def test_directive_deserialize_continue(self):
        raw = json.dumps({"action": "continue"})
        directive = Directive.model_validate_json(raw)
        assert directive.action == DirectiveAction.continue_

    def test_directive_roundtrip(self):
        original = Directive(
            action=DirectiveAction.skip,
            task="Task3",
            reason="Low priority",
        )
        serialized = original.model_dump_json()
        restored = Directive.model_validate_json(serialized)
        assert restored.action == original.action
        assert restored.task == original.task
        assert restored.reason == original.reason


# ---------------------------------------------------------------------------
# GameState tests
# ---------------------------------------------------------------------------

class TestGameState:
    def test_empty_defaults(self):
        state = GameState()
        assert state.pending_tasks == []
        assert state.waiting_tasks == []
        assert state.completion_history == []
        assert state.failure_history == []
        assert state.last_screenshot is None
        assert state.last_updated is not None

    def test_with_data(self):
        task_info = TaskInfo(
            task="Task1",
            success=True,
            duration=10.0,
            count=1,
        )
        state = GameState(
            pending_tasks=["Task2", "Task3"],
            waiting_tasks=["Task4"],
            completion_history=[task_info],
        )
        assert len(state.pending_tasks) == 2
        assert len(state.completion_history) == 1
        assert state.completion_history[0].task == "Task1"

    def test_last_screenshot_optional(self):
        state = GameState(last_screenshot=None)
        assert state.last_screenshot is None

        state_with_screenshot = GameState(last_screenshot="imgdata")
        assert state_with_screenshot.last_screenshot == "imgdata"


# ---------------------------------------------------------------------------
# TaskInfo tests
# ---------------------------------------------------------------------------

class TestTaskInfo:
    def test_basic_construction(self):
        info = TaskInfo(task="Task1", success=True, duration=3.5, count=2)
        assert info.task == "Task1"
        assert info.success is True
        assert info.duration == pytest.approx(3.5)
        assert info.count == 2
        assert info.error is None
        assert info.timestamp is not None

    def test_with_error(self):
        info = TaskInfo(task="Task2", success=False, duration=0.5, error="Crash", count=1)
        assert info.success is False
        assert info.error == "Crash"
