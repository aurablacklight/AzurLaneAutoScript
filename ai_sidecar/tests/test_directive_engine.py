import json
import logging
import os
import tempfile

import pytest

from directive_engine import DirectiveEngine, FALLBACK
from models import Directive, DirectiveAction


@pytest.fixture
def engine(tmp_path):
    log_file = str(tmp_path / "decisions.log")
    eng = DirectiveEngine(decision_log=log_file)
    yield eng
    # Clean up logger handlers to avoid cross-test pollution
    logger = logging.getLogger("directive_engine")
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# validate tests
# ---------------------------------------------------------------------------

class TestValidateContinue:
    def test_continue_passes_through(self, engine):
        directive = Directive(action=DirectiveAction.continue_)
        result = engine.validate(directive, enabled_tasks=["TaskA"])
        assert result is directive
        assert result.action == DirectiveAction.continue_


class TestValidateReprioritize:
    def test_reprioritize_with_valid_tasks_passes(self, engine):
        directive = Directive(action=DirectiveAction.reprioritize, task_order=["TaskA", "TaskB"])
        result = engine.validate(directive, enabled_tasks=["TaskA", "TaskB", "TaskC"])
        assert result is directive
        assert result.action == DirectiveAction.reprioritize

    def test_reprioritize_with_unknown_task_falls_back(self, engine):
        directive = Directive(action=DirectiveAction.reprioritize, task_order=["TaskA", "UnknownTask"])
        result = engine.validate(directive, enabled_tasks=["TaskA", "TaskB"])
        assert result.action == DirectiveAction.continue_

    def test_reprioritize_with_disabled_task_falls_back(self, engine):
        directive = Directive(action=DirectiveAction.reprioritize, task_order=["TaskA", "TaskC"])
        result = engine.validate(directive, enabled_tasks=["TaskA", "TaskB"])
        assert result.action == DirectiveAction.continue_

    def test_reprioritize_with_empty_task_order_falls_back(self, engine):
        directive = Directive(action=DirectiveAction.reprioritize, task_order=[])
        result = engine.validate(directive, enabled_tasks=["TaskA"])
        assert result.action == DirectiveAction.continue_

    def test_reprioritize_with_none_task_order_falls_back(self, engine):
        directive = Directive(action=DirectiveAction.reprioritize, task_order=None)
        result = engine.validate(directive, enabled_tasks=["TaskA"])
        assert result.action == DirectiveAction.continue_


class TestValidateSkip:
    def test_skip_with_valid_task_passes(self, engine):
        directive = Directive(action=DirectiveAction.skip, task="TaskA")
        result = engine.validate(directive, enabled_tasks=["TaskA", "TaskB"])
        assert result is directive
        assert result.action == DirectiveAction.skip

    def test_skip_with_unknown_task_falls_back(self, engine):
        directive = Directive(action=DirectiveAction.skip, task="UnknownTask")
        result = engine.validate(directive, enabled_tasks=["TaskA", "TaskB"])
        assert result.action == DirectiveAction.continue_

    def test_skip_with_none_task_falls_back(self, engine):
        directive = Directive(action=DirectiveAction.skip, task=None)
        result = engine.validate(directive, enabled_tasks=["TaskA"])
        assert result.action == DirectiveAction.continue_


class TestValidatePause:
    def test_pause_with_reason_passes(self, engine):
        directive = Directive(action=DirectiveAction.pause, reason="Maintenance window")
        result = engine.validate(directive, enabled_tasks=["TaskA"])
        assert result is directive
        assert result.action == DirectiveAction.pause

    def test_pause_with_empty_reason_falls_back(self, engine):
        directive = Directive(action=DirectiveAction.pause, reason="")
        result = engine.validate(directive, enabled_tasks=["TaskA"])
        assert result.action == DirectiveAction.continue_

    def test_pause_with_none_reason_falls_back(self, engine):
        directive = Directive(action=DirectiveAction.pause, reason=None)
        result = engine.validate(directive, enabled_tasks=["TaskA"])
        assert result.action == DirectiveAction.continue_


# ---------------------------------------------------------------------------
# log_decision tests
# ---------------------------------------------------------------------------

class TestLogDecision:
    def test_log_decision_writes_to_file(self, tmp_path):
        log_file = str(tmp_path / "decisions.log")
        engine = DirectiveEngine(decision_log=log_file)
        directive = Directive(action=DirectiveAction.continue_)
        engine.log_decision(
            event_type="cycle_start",
            ai_response='{"action": "continue"}',
            final_directive=directive,
            state_summary="3 pending tasks",
        )
        # Flush handlers
        for handler in engine.logger.handlers:
            handler.flush()

        assert os.path.exists(log_file)
        with open(log_file) as f:
            content = f.read()
        assert "cycle_start" in content
        assert "continue" in content
        assert "3 pending tasks" in content

        # Clean up
        for handler in engine.logger.handlers[:]:
            handler.close()
            engine.logger.removeHandler(handler)

    def test_log_decision_without_state_summary(self, tmp_path):
        log_file = str(tmp_path / "decisions.log")
        engine = DirectiveEngine(decision_log=log_file)
        directive = Directive(action=DirectiveAction.continue_)
        engine.log_decision(
            event_type="task_complete",
            ai_response='{"action": "continue"}',
            final_directive=directive,
        )
        for handler in engine.logger.handlers:
            handler.flush()

        with open(log_file) as f:
            content = f.read()

        # Parse the JSON portion of the log line
        json_start = content.index("{")
        entry = json.loads(content[json_start:].strip())
        assert "state_summary" not in entry
        assert entry["event_type"] == "task_complete"

        for handler in engine.logger.handlers[:]:
            handler.close()
            engine.logger.removeHandler(handler)

    def test_log_decision_truncates_long_ai_response(self, tmp_path):
        log_file = str(tmp_path / "decisions.log")
        engine = DirectiveEngine(decision_log=log_file)
        directive = Directive(action=DirectiveAction.continue_)
        long_response = "x" * 1000
        engine.log_decision(
            event_type="cycle_start",
            ai_response=long_response,
            final_directive=directive,
        )
        for handler in engine.logger.handlers:
            handler.flush()

        with open(log_file) as f:
            content = f.read()
        json_start = content.index("{")
        entry = json.loads(content[json_start:].strip())
        assert len(entry["ai_response"]) == 500

        for handler in engine.logger.handlers[:]:
            handler.close()
            engine.logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# parse_ai_response tests
# ---------------------------------------------------------------------------

class TestParseAiResponse:
    def test_valid_json_continue(self, engine):
        result = engine.parse_ai_response('{"action": "continue"}')
        assert result.action == DirectiveAction.continue_

    def test_valid_json_skip(self, engine):
        result = engine.parse_ai_response('{"action": "skip", "task": "TaskA"}')
        assert result.action == DirectiveAction.skip
        assert result.task == "TaskA"

    def test_valid_json_reprioritize(self, engine):
        result = engine.parse_ai_response('{"action": "reprioritize", "task_order": ["TaskA", "TaskB"]}')
        assert result.action == DirectiveAction.reprioritize
        assert result.task_order == ["TaskA", "TaskB"]

    def test_invalid_json_falls_back(self, engine):
        result = engine.parse_ai_response("this is not json at all")
        assert result.action == DirectiveAction.continue_

    def test_invalid_action_falls_back(self, engine):
        result = engine.parse_ai_response('{"action": "invalid_action"}')
        assert result.action == DirectiveAction.continue_

    def test_strips_markdown_code_fence_json(self, engine):
        raw = "```json\n{\"action\": \"continue\"}\n```"
        result = engine.parse_ai_response(raw)
        assert result.action == DirectiveAction.continue_

    def test_strips_markdown_code_fence_no_lang(self, engine):
        raw = "```\n{\"action\": \"pause\", \"reason\": \"need rest\"}\n```"
        result = engine.parse_ai_response(raw)
        assert result.action == DirectiveAction.pause
        assert result.reason == "need rest"

    def test_empty_string_falls_back(self, engine):
        result = engine.parse_ai_response("")
        assert result.action == DirectiveAction.continue_

    def test_partial_json_falls_back(self, engine):
        result = engine.parse_ai_response('{"action":')
        assert result.action == DirectiveAction.continue_
