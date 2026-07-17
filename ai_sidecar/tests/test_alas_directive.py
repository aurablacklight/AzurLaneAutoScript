"""
Tests for alas.py AI-sidecar directive handling:
- pause guard in _apply_directive (defer to ALAS auto-recovery when the task
  has fewer than 2 prior consecutive failures)
- enriched unknown_state payload (task / error / count) on GameStuckError
"""
import sys
import os
import threading
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from alas import AzurLaneAutoScript
from module.exception import GameStuckError


def make_instance():
    instance = AzurLaneAutoScript(config_name='alas')
    instance.stop_event = threading.Event()
    return instance


class TestPauseGuard:
    def test_pause_with_low_failure_count_defers_to_alas(self):
        """pause directive for a task with 0 prior failures must NOT stop the scheduler."""
        instance = make_instance()
        instance._apply_directive(
            {'action': 'pause', 'reason': 'looks stuck'}, task='Commission')
        assert not instance.stop_event.is_set()

    def test_pause_with_failure_count_two_is_honored(self):
        """pause directive for a task with >= 2 prior failures sets stop_event."""
        instance = make_instance()
        instance.failure_record['Commission'] = 2
        instance._apply_directive(
            {'action': 'pause', 'reason': 'repeated failures'}, task='Commission')
        assert instance.stop_event.is_set()

    def test_pause_without_task_is_honored(self):
        """pause with no task context (e.g. cycle_start) stays honored unconditionally."""
        instance = make_instance()
        instance._apply_directive({'action': 'pause', 'reason': 'user asked'})
        assert instance.stop_event.is_set()

    def test_skip_directive_still_works_with_new_signature(self):
        """skip directive keeps working when a task kwarg is passed."""
        instance = make_instance()
        instance.__dict__['config'] = MagicMock()
        instance._apply_directive(
            {'action': 'skip', 'task': 'Commission'}, task='Commission')
        instance.config.task_delay.assert_called_once_with(
            minute=1440, task='Commission')
        assert not instance.stop_event.is_set()


class TestUnknownStatePayload:
    def test_unknown_state_payload_is_enriched(self):
        """GameStuckError during run() sends task, error and count to the sidecar."""
        instance = make_instance()
        device = MagicMock()
        device.click_record = ['CLICK_A', 'CLICK_B']
        instance.__dict__['device'] = device
        instance.__dict__['config'] = MagicMock()
        instance._sidecar = MagicMock()
        instance._sidecar.notify.return_value = None
        instance._sidecar.screenshot_to_base64.return_value = 'b64data'
        instance.save_error_log = lambda: None

        def commission():
            raise GameStuckError('Triggered commission list flashing bug')
        instance.commission = commission

        result = instance.run('commission', skip_first_screenshot=True)

        assert result is False
        instance._sidecar.notify.assert_called_once()
        event, payload = instance._sidecar.notify.call_args[0]
        assert event == 'unknown_state'
        assert payload['screenshot'] == 'b64data'
        assert payload['click_history'] == ['CLICK_A', 'CLICK_B']
        assert payload['task'] == 'Commission'
        assert 'flashing bug' in payload['error']
        assert payload['count'] == 0
