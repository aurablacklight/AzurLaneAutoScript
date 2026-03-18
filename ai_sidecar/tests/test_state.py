import json
import os
import tempfile

import pytest

from models import EventPayload, EventType
from state import StateManager


@pytest.fixture
def state_file(tmp_path):
    return str(tmp_path / "state.json")


def test_initial_state_is_empty(state_file):
    sm = StateManager(state_file)
    assert sm.state.pending_tasks == []
    assert sm.state.waiting_tasks == []
    assert sm.state.completion_history == []
    assert sm.state.failure_history == []
    assert sm.state.last_screenshot is None


def test_cycle_start_populates_tasks_and_screenshot(state_file):
    sm = StateManager(state_file)
    event = EventPayload(
        event_type=EventType.cycle_start,
        pending_tasks=["TaskA", "TaskB"],
        waiting_tasks=["TaskC"],
        screenshot="base64encodedimage",
    )
    sm.update(event)

    assert sm.state.pending_tasks == ["TaskA", "TaskB"]
    assert sm.state.waiting_tasks == ["TaskC"]
    assert sm.state.last_screenshot == "base64encodedimage"


def test_cycle_start_without_screenshot_leaves_screenshot_unchanged(state_file):
    sm = StateManager(state_file)
    # Set an initial screenshot via unknown_state
    sm.update(EventPayload(event_type=EventType.unknown_state, screenshot="original"))
    sm.update(EventPayload(
        event_type=EventType.cycle_start,
        pending_tasks=["TaskA"],
        waiting_tasks=[],
    ))
    assert sm.state.last_screenshot == "original"


def test_task_complete_adds_to_completion_history(state_file):
    sm = StateManager(state_file)
    event = EventPayload(
        event_type=EventType.task_complete,
        task="DailyMission",
        success=True,
        duration=42.5,
    )
    sm.update(event)

    assert len(sm.state.completion_history) == 1
    entry = sm.state.completion_history[0]
    assert entry.task == "DailyMission"
    assert entry.success is True
    assert entry.duration == 42.5


def test_task_failed_adds_to_failure_history(state_file):
    sm = StateManager(state_file)
    event = EventPayload(
        event_type=EventType.task_failed,
        task="HardMode",
        error="Timeout waiting for stage",
        count=3,
    )
    sm.update(event)

    assert len(sm.state.failure_history) == 1
    entry = sm.state.failure_history[0]
    assert entry.task == "HardMode"
    assert entry.success is False
    assert entry.error == "Timeout waiting for stage"
    assert entry.count == 3


def test_state_persistence_survives_reload(state_file):
    sm = StateManager(state_file)
    sm.update(EventPayload(
        event_type=EventType.cycle_start,
        pending_tasks=["TaskX", "TaskY"],
        waiting_tasks=["TaskZ"],
        screenshot="img_data",
    ))
    sm.update(EventPayload(
        event_type=EventType.task_complete,
        task="TaskX",
        success=True,
        duration=10.0,
    ))

    # Create a new StateManager loading from the same file
    sm2 = StateManager(state_file)
    assert sm2.state.pending_tasks == ["TaskX", "TaskY"]
    assert sm2.state.waiting_tasks == ["TaskZ"]
    assert sm2.state.last_screenshot == "img_data"
    assert len(sm2.state.completion_history) == 1
    assert sm2.state.completion_history[0].task == "TaskX"


def test_atomic_write_produces_valid_json(state_file):
    sm = StateManager(state_file)
    sm.update(EventPayload(
        event_type=EventType.cycle_start,
        pending_tasks=["AtomicTask"],
        waiting_tasks=[],
    ))

    assert os.path.exists(state_file)
    with open(state_file, "r") as f:
        data = json.load(f)
    assert "pending_tasks" in data
    assert data["pending_tasks"] == ["AtomicTask"]


def test_atomic_write_no_tmp_files_left_behind(state_file):
    sm = StateManager(state_file)
    sm.update(EventPayload(
        event_type=EventType.cycle_start,
        pending_tasks=["Task1"],
        waiting_tasks=[],
    ))

    dir_name = os.path.dirname(state_file)
    tmp_files = [f for f in os.listdir(dir_name) if f.endswith(".tmp")]
    assert tmp_files == []


def test_history_capping(state_file):
    sm = StateManager(state_file, max_history=5)
    for i in range(10):
        sm.update(EventPayload(
            event_type=EventType.task_complete,
            task=f"Task{i}",
            success=True,
            duration=float(i),
        ))

    assert len(sm.state.completion_history) == 5
    # Should contain the last 5 tasks (Task5 through Task9)
    task_names = [t.task for t in sm.state.completion_history]
    assert task_names == ["Task5", "Task6", "Task7", "Task8", "Task9"]


def test_failure_history_capping(state_file):
    sm = StateManager(state_file, max_history=5)
    for i in range(10):
        sm.update(EventPayload(
            event_type=EventType.task_failed,
            task=f"FailTask{i}",
            error="some error",
            count=i,
        ))

    assert len(sm.state.failure_history) == 5
    task_names = [t.task for t in sm.state.failure_history]
    assert task_names == ["FailTask5", "FailTask6", "FailTask7", "FailTask8", "FailTask9"]


def test_get_summary_contains_task_names(state_file):
    sm = StateManager(state_file)
    sm.update(EventPayload(
        event_type=EventType.cycle_start,
        pending_tasks=["MissionAlpha", "MissionBeta"],
        waiting_tasks=["MissionGamma"],
    ))
    sm.update(EventPayload(
        event_type=EventType.task_complete,
        task="CompletedTask",
        success=True,
        duration=30.0,
    ))
    sm.update(EventPayload(
        event_type=EventType.task_failed,
        task="FailedTask",
        error="network error",
        count=2,
    ))

    summary = sm.get_summary()

    assert isinstance(summary, str)
    assert "MissionAlpha" in summary
    assert "MissionBeta" in summary
    assert "MissionGamma" in summary
    assert "CompletedTask" in summary
    assert "FailedTask" in summary


def test_get_summary_empty_state(state_file):
    sm = StateManager(state_file)
    summary = sm.get_summary()
    assert "none" in summary
    assert isinstance(summary, str)


def test_load_from_corrupted_file_returns_empty_state(state_file):
    # Write invalid JSON to the state file
    with open(state_file, "w") as f:
        f.write("{ not valid json }")

    sm = StateManager(state_file)
    assert sm.state.pending_tasks == []
    assert sm.state.completion_history == []


def test_unknown_state_updates_screenshot(state_file):
    sm = StateManager(state_file)
    event = EventPayload(
        event_type=EventType.unknown_state,
        screenshot="unknown_screen_data",
    )
    sm.update(event)
    assert sm.state.last_screenshot == "unknown_screen_data"
