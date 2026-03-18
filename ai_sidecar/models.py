from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class EventType(str, Enum):
    cycle_start = "cycle_start"
    task_complete = "task_complete"
    task_failed = "task_failed"
    unknown_state = "unknown_state"


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

    # unknown_state fields
    count: Optional[int] = None
    screenshot: Optional[str] = None
    click_history: Optional[List[str]] = None


class DirectiveAction(str, Enum):
    continue_ = "continue"
    reprioritize = "reprioritize"
    skip = "skip"
    pause = "pause"


class Directive(BaseModel):
    action: DirectiveAction
    task_order: Optional[List[str]] = None
    task: Optional[str] = None
    reason: Optional[str] = None


class TaskInfo(BaseModel):
    task: str
    success: bool
    duration: float
    error: Optional[str] = None
    count: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class GameState(BaseModel):
    pending_tasks: List[str] = Field(default_factory=list)
    waiting_tasks: List[str] = Field(default_factory=list)
    completion_history: List[TaskInfo] = Field(default_factory=list)
    failure_history: List[TaskInfo] = Field(default_factory=list)
    last_screenshot: Optional[str] = None
    last_updated: datetime = Field(default_factory=datetime.utcnow)
