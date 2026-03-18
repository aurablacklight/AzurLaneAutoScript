"""Tests for the FastAPI application in main.py."""
from __future__ import annotations

import importlib
import os
import sys
import tempfile
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prompts_dir(tmp_path) -> str:
    """Create a temporary prompts directory with required template files."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "system.txt").write_text("You are a helpful assistant.")
    (prompts_dir / "user_template.txt").write_text(
        "Event: {event_type}\n{event_details}\nState: {state_summary}\n"
        "Tasks: {enabled_tasks}\nTime: {server_time}"
    )
    return str(prompts_dir)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def app_client(tmp_path) -> Generator[TestClient, None, None]:
    """
    Set up environment variables, reload main module, yield TestClient.
    Cleans up module and env vars afterwards.
    """
    state_file = str(tmp_path / "state.json")
    decision_log = str(tmp_path / "decisions.log")
    prompts_dir = _make_prompts_dir(tmp_path)

    env_vars = {
        "AI_BASE_URL": "http://fake-ai:11434/v1",
        "AI_API_KEY": "test-key",
        "AI_MODEL": "test-model",
        "AI_TIMEOUT": "5",
        "AI_MAX_RETRIES": "0",
        "SIDECAR_STATE_FILE": state_file,
        "SIDECAR_DECISION_LOG": decision_log,
        "SIDECAR_PROMPTS_DIR": prompts_dir,
        "SIDECAR_CONSULT_ON": "cycle_start,task_failed,unknown_state",
        "SIDECAR_COOLDOWN": "0",  # disable debounce in tests
    }

    # Apply env vars
    old_env = {k: os.environ.get(k) for k in env_vars}
    os.environ.update(env_vars)

    # Ensure ai_sidecar root is on sys.path
    sidecar_root = os.path.join(os.path.dirname(__file__), "..")
    sidecar_root = os.path.abspath(sidecar_root)
    if sidecar_root not in sys.path:
        sys.path.insert(0, sidecar_root)

    # Remove cached module so it gets re-imported with fresh env vars
    for mod_name in list(sys.modules.keys()):
        if mod_name in ("main",) or mod_name.startswith("main."):
            del sys.modules[mod_name]

    import main as main_module

    # Reset debounce state between tests
    main_module._last_cycle_start_consult = 0.0

    client = TestClient(main_module.app, raise_server_exceptions=True)
    yield client, main_module

    # Restore env vars
    for k, v in old_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

    # Clean up module cache
    for mod_name in list(sys.modules.keys()):
        if mod_name in ("main",) or mod_name.startswith("main."):
            del sys.modules[mod_name]


# ---------------------------------------------------------------------------
# Tests: GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self, app_client):
        client, _ = app_client
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status_ok(self, app_client):
        client, _ = app_client
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_returns_api_version(self, app_client):
        client, _ = app_client
        data = client.get("/health").json()
        assert data["api_version"] == "1.0"


# ---------------------------------------------------------------------------
# Tests: POST /api/events
# ---------------------------------------------------------------------------

class TestPostEvents:
    def test_cycle_start_with_mocked_ai_returns_directive(self, app_client):
        client, main_module = app_client
        mock_response = '{"action": "continue"}'
        with patch.object(main_module.ai_client, "consult", return_value=mock_response):
            response = client.post(
                "/api/events",
                json={
                    "event_type": "cycle_start",
                    "pending_tasks": ["TaskA", "TaskB"],
                    "waiting_tasks": [],
                },
            )
        assert response.status_code == 200
        data = response.json()
        assert data["action"] == "continue"

    def test_task_complete_does_not_consult_ai_by_default(self, app_client, tmp_path):
        """task_complete is NOT in the default SIDECAR_CONSULT_ON, so AI should not be called."""
        client, main_module = app_client
        # Patch ai_client.consult to detect if it's called
        with patch.object(main_module.ai_client, "consult", return_value='{"action": "continue"}') as mock_consult:
            response = client.post(
                "/api/events",
                json={
                    "event_type": "task_complete",
                    "task": "DailyMission",
                    "success": True,
                    "duration": 42.0,
                },
            )
        assert response.status_code == 200
        # AI should NOT have been consulted for task_complete
        mock_consult.assert_not_called()
        data = response.json()
        assert data["action"] == "continue"

    def test_task_failed_consults_ai(self, app_client):
        client, main_module = app_client
        mock_response = '{"action": "skip", "task": "TaskA"}'
        with patch.object(main_module.ai_client, "consult", return_value=mock_response) as mock_consult:
            response = client.post(
                "/api/events",
                json={
                    "event_type": "task_failed",
                    "task": "TaskA",
                    "error": "Connection timeout",
                    "count": 3,
                },
            )
        assert response.status_code == 200
        mock_consult.assert_called_once()

    def test_task_failed_returns_directive_from_ai(self, app_client):
        client, main_module = app_client
        # First set up state so TaskA is a known enabled task
        client.post(
            "/api/events",
            json={
                "event_type": "cycle_start",
                "pending_tasks": ["TaskA"],
                "waiting_tasks": [],
            },
        )
        mock_response = '{"action": "skip", "task": "TaskA"}'
        with patch.object(main_module.ai_client, "consult", return_value=mock_response):
            response = client.post(
                "/api/events",
                json={
                    "event_type": "task_failed",
                    "task": "TaskA",
                    "error": "Timeout",
                    "count": 2,
                },
            )
        assert response.status_code == 200
        data = response.json()
        assert data["action"] == "skip"
        assert data["task"] == "TaskA"

    def test_invalid_event_type_returns_422(self, app_client):
        client, _ = app_client
        response = client.post(
            "/api/events",
            json={"event_type": "not_a_real_event"},
        )
        assert response.status_code == 422

    def test_ai_unreachable_returns_continue(self, app_client):
        """When AI returns None (unreachable / error), fallback to continue."""
        client, main_module = app_client
        with patch.object(main_module.ai_client, "consult", return_value=None):
            response = client.post(
                "/api/events",
                json={
                    "event_type": "cycle_start",
                    "pending_tasks": [],
                    "waiting_tasks": [],
                },
            )
        assert response.status_code == 200
        data = response.json()
        assert data["action"] == "continue"

    def test_ai_unparseable_response_returns_continue(self, app_client):
        """When AI returns garbage that can't be parsed, fallback to continue."""
        client, main_module = app_client
        with patch.object(main_module.ai_client, "consult", return_value="this is not json"):
            response = client.post(
                "/api/events",
                json={
                    "event_type": "unknown_state",
                    "click_history": ["btn_start", "btn_confirm"],
                },
            )
        assert response.status_code == 200
        data = response.json()
        assert data["action"] == "continue"

    def test_cycle_start_debounce(self, tmp_path):
        """Rapid cycle_start events should be debounced after the first."""
        import time

        state_file = str(tmp_path / "state.json")
        decision_log = str(tmp_path / "decisions.log")
        prompts_dir = _make_prompts_dir(tmp_path)

        env_vars = {
            "AI_BASE_URL": "http://fake-ai:11434/v1",
            "AI_API_KEY": "test-key",
            "AI_MODEL": "test-model",
            "AI_TIMEOUT": "5",
            "AI_MAX_RETRIES": "0",
            "SIDECAR_STATE_FILE": state_file,
            "SIDECAR_DECISION_LOG": decision_log,
            "SIDECAR_PROMPTS_DIR": prompts_dir,
            "SIDECAR_CONSULT_ON": "cycle_start",
            "SIDECAR_COOLDOWN": "60",  # long cooldown so second call is debounced
        }
        old_env = {k: os.environ.get(k) for k in env_vars}
        os.environ.update(env_vars)

        sidecar_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if sidecar_root not in sys.path:
            sys.path.insert(0, sidecar_root)

        for mod_name in list(sys.modules.keys()):
            if mod_name in ("main",) or mod_name.startswith("main."):
                del sys.modules[mod_name]

        import main as main_module
        main_module._last_cycle_start_consult = 0.0

        client = TestClient(main_module.app)

        call_count = 0

        def fake_consult(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return '{"action": "continue"}'

        with patch.object(main_module.ai_client, "consult", side_effect=fake_consult):
            # First call — should consult AI
            client.post("/api/events", json={"event_type": "cycle_start", "pending_tasks": [], "waiting_tasks": []})
            # Second call immediately — should be debounced (no AI call)
            client.post("/api/events", json={"event_type": "cycle_start", "pending_tasks": [], "waiting_tasks": []})

        assert call_count == 1, f"Expected 1 AI call (debounce), got {call_count}"

        # Restore
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for mod_name in list(sys.modules.keys()):
            if mod_name in ("main",) or mod_name.startswith("main."):
                del sys.modules[mod_name]

    def test_reprioritize_directive_returned(self, app_client):
        """AI reprioritize directive is validated and returned."""
        client, main_module = app_client
        # Seed state with known tasks
        client.post(
            "/api/events",
            json={
                "event_type": "cycle_start",
                "pending_tasks": ["TaskA", "TaskB"],
                "waiting_tasks": [],
            },
        )
        mock_response = '{"action": "reprioritize", "task_order": ["TaskB", "TaskA"]}'
        with patch.object(main_module.ai_client, "consult", return_value=mock_response):
            response = client.post(
                "/api/events",
                json={
                    "event_type": "task_failed",
                    "task": "TaskA",
                    "error": "Repeated failure",
                    "count": 5,
                },
            )
        assert response.status_code == 200
        data = response.json()
        assert data["action"] == "reprioritize"
        assert data["task_order"] == ["TaskB", "TaskA"]
