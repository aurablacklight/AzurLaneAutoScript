"""FastAPI application for the AI sidecar."""
from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime
from typing import List

from fastapi import FastAPI

# Allow imports from the ai_sidecar package directory
sys.path.insert(0, os.path.dirname(__file__))

from ai_client import AIClient
from directive_engine import DirectiveEngine
from models import Directive, DirectiveAction, EventPayload, EventType
from state import StateManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------

AI_BASE_URL: str = os.environ.get("AI_BASE_URL", "http://localhost:11434/v1")
AI_API_KEY: str = os.environ.get("AI_API_KEY", "ollama")
AI_MODEL: str = os.environ.get("AI_MODEL", "llama3")
AI_TIMEOUT: int = int(os.environ.get("AI_TIMEOUT", "30"))
AI_MAX_RETRIES: int = int(os.environ.get("AI_MAX_RETRIES", "1"))

SIDECAR_STATE_FILE: str = os.environ.get("SIDECAR_STATE_FILE", "/tmp/sidecar_state.json")
SIDECAR_DECISION_LOG: str = os.environ.get("SIDECAR_DECISION_LOG", "/tmp/sidecar_decisions.log")
SIDECAR_PROMPTS_DIR: str = os.environ.get("SIDECAR_PROMPTS_DIR", os.path.join(os.path.dirname(__file__), "prompts"))
SIDECAR_COOLDOWN: int = int(os.environ.get("SIDECAR_COOLDOWN", "60"))

# Comma-separated list of event types that trigger AI consultation
_consult_on_raw: str = os.environ.get("SIDECAR_CONSULT_ON", "cycle_start,task_failed,unknown_state")

# ---------------------------------------------------------------------------
# Validate consult_on values against known EventType enum values
# ---------------------------------------------------------------------------
KNOWN_EVENT_TYPES = {e.value for e in EventType}
_consult_on_list: List[str] = [v.strip() for v in _consult_on_raw.split(",") if v.strip()]
for _val in _consult_on_list:
    if _val not in KNOWN_EVENT_TYPES:
        raise ValueError(
            f"SIDECAR_CONSULT_ON contains unknown event type: {_val!r}. "
            f"Valid values: {sorted(KNOWN_EVENT_TYPES)}"
        )
CONSULT_ON: set = set(_consult_on_list)

# ---------------------------------------------------------------------------
# Load prompt templates
# ---------------------------------------------------------------------------

def _load_prompt(filename: str) -> str:
    path = os.path.join(SIDECAR_PROMPTS_DIR, filename)
    with open(path, "r") as f:
        return f.read()


SYSTEM_PROMPT: str = _load_prompt("system.txt")
USER_TEMPLATE: str = _load_prompt("user_template.txt")

# ---------------------------------------------------------------------------
# Initialize components
# ---------------------------------------------------------------------------

state_manager = StateManager(state_file=SIDECAR_STATE_FILE)

directive_engine = DirectiveEngine(decision_log=SIDECAR_DECISION_LOG)

ai_client = AIClient(
    base_url=AI_BASE_URL,
    api_key=AI_API_KEY,
    model=AI_MODEL,
    timeout=AI_TIMEOUT,
    max_retries=AI_MAX_RETRIES,
)

# Debounce: track timestamp of last cycle_start AI consultation
_last_cycle_start_consult: float = 0.0

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="AI Sidecar", version="1.0")

FALLBACK = Directive(action=DirectiveAction.continue_)


def _format_event_details(event: EventPayload) -> str:
    """Format human-readable event details for the user prompt."""
    if event.event_type == EventType.cycle_start:
        pending = ", ".join(event.pending_tasks) if event.pending_tasks else "none"
        waiting = ", ".join(event.waiting_tasks) if event.waiting_tasks else "none"
        return f"Pending tasks: {pending}\nWaiting tasks: {waiting}"
    elif event.event_type == EventType.task_complete:
        duration_str = f" in {event.duration:.0f}s" if event.duration else ""
        return f"Task: {event.task}{duration_str}"
    elif event.event_type == EventType.task_failed:
        return f"Task: {event.task}\nError: {event.error}\nFailure count: {event.count}"
    elif event.event_type == EventType.unknown_state:
        history = ", ".join(event.click_history) if event.click_history else "none"
        return f"Click history: {history}"
    return ""


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "api_version": "1.0"}


@app.post("/api/events")
def handle_event(event: EventPayload) -> dict:
    global _last_cycle_start_consult

    # Update state
    state_manager.update(event)

    # Determine whether to consult AI
    should_consult = event.event_type.value in CONSULT_ON

    # Debounce cycle_start events
    if event.event_type == EventType.cycle_start and should_consult:
        now = time.time()
        if now - _last_cycle_start_consult < SIDECAR_COOLDOWN:
            should_consult = False
        else:
            _last_cycle_start_consult = now

    if not should_consult:
        return FALLBACK.model_dump()

    # Build prompt
    state_summary = state_manager.get_summary()
    enabled_tasks = (
        state_manager.state.pending_tasks + state_manager.state.waiting_tasks
    )
    event_details = _format_event_details(event)
    server_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    user_message = USER_TEMPLATE.format(
        event_type=event.event_type.value,
        event_details=event_details,
        state_summary=state_summary,
        enabled_tasks=", ".join(enabled_tasks) if enabled_tasks else "none",
        server_time=server_time,
    )

    # Consult AI
    raw_response = ai_client.consult(
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
        screenshot_base64=state_manager.state.last_screenshot,
    )

    if raw_response is None:
        directive_engine.log_decision(
            event_type=event.event_type.value,
            ai_response="(no response)",
            final_directive=FALLBACK,
            state_summary=state_summary,
        )
        return FALLBACK.model_dump()

    # Parse and validate directive
    parsed = directive_engine.parse_ai_response(raw_response)
    validated = directive_engine.validate(parsed, enabled_tasks=enabled_tasks)

    directive_engine.log_decision(
        event_type=event.event_type.value,
        ai_response=raw_response,
        final_directive=validated,
        state_summary=state_summary,
    )

    return validated.model_dump()
