import json
import logging
import re
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import List, Optional

from models import Directive, DirectiveAction

FALLBACK = Directive(action=DirectiveAction.continue_)


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
            handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            self.logger.addHandler(handler)

    def validate(self, directive: Directive, enabled_tasks: List[str]) -> Directive:
        if directive.action == DirectiveAction.continue_:
            return directive
        if directive.action == DirectiveAction.reprioritize:
            if not directive.task_order:
                return FALLBACK
            for task in directive.task_order:
                if task not in enabled_tasks:
                    self.logger.warning(f"Reprioritize references unknown/disabled task: {task}")
                    return FALLBACK
            return directive
        if directive.action == DirectiveAction.skip:
            if not directive.task or directive.task not in enabled_tasks:
                self.logger.warning(f"Skip references unknown/disabled task: {directive.task}")
                return FALLBACK
            return directive
        if directive.action == DirectiveAction.pause:
            if not directive.reason:
                self.logger.warning("Pause directive missing reason")
                return FALLBACK
            return directive
        return FALLBACK

    def parse_ai_response(self, raw: str) -> Directive:
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
