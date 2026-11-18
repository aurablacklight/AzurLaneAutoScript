"""
Tests for alas.py AI-sidecar directive handling:
- pause guard in _apply_directive (defer to ALAS auto-recovery when the task
  has fewer than 2 prior consecutive failures)
- enriched unknown_state payload (task / error / count) on GameStuckError
- push notification when a pause is honoured
"""
import sys
import os
import threading
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import alas
from alas import AzurLaneAutoScript
from module.exception import GameStuckError


@pytest.fixture(autouse=True)
def notifier(monkeypatch):
    """A honoured pause sends ALAS's push notification; no test here may reach a real notifier."""
    sent = MagicMock()
    monkeypatch.setattr(alas, 'handle_notify', sent)
    return sent


def make_instance():
    instance = AzurLaneAutoScript(config_name='alas')
    instance.stop_event = threading.Event()
    # A honoured pause reads Error_OnePushConfig. Inject a fake config so the
    # cached_property never loads config/alas.json from the working directory.
    instance.__dict__['config'] = MagicMock()
    return instance


class TestPauseGuard:
    def test_pause_with_low_failure_count_defers_to_alas(self):
        """pause directive for a task with 0 prior failures must NOT stop the scheduler."""
        instance = make_instance()
        instance._apply_directive(
            {'action': 'pause', 'reason': 'looks stuck'}, task='Commission')
        assert not instance.stop_event.is_set()
        assert instance._ai_pause is False

    def test_pause_with_failure_count_two_is_honored(self):
        """pause with >= 2 prior failures sets the instance pause flag.

        It must NOT set stop_event: that is the GUI's shared updater event,
        and setting it poisons every future Start until the GUI restarts.
        """
        instance = make_instance()
        instance.failure_record['Commission'] = 2
        instance._apply_directive(
            {'action': 'pause', 'reason': 'repeated failures'}, task='Commission')
        assert not instance.stop_event.is_set()
        assert instance._ai_pause is True

    def test_pause_without_task_is_honored(self):
        """pause with no task context (e.g. cycle_start) stays honored unconditionally.

        Same contract: instance flag only, never the shared stop_event.
        """
        instance = make_instance()
        instance._apply_directive({'action': 'pause', 'reason': 'user asked'})
        assert not instance.stop_event.is_set()
        assert instance._ai_pause is True


class TestPauseNotification:
    def test_honoured_task_pause_sends_push_notification_with_reason(self, notifier):
        """The loop ends quietly on a pause; without an alert nobody learns that ALAS stopped."""
        instance = make_instance()
        instance.config.Error_OnePushConfig = 'provider: null'
        instance.failure_record['Commission'] = 2
        instance._apply_directive(
            {'action': 'pause', 'reason': 'repeated failures'}, task='Commission')
        assert instance._ai_pause is True
        notifier.assert_called_once()
        args, kwargs = notifier.call_args
        assert args == ('provider: null',)
        assert sorted(kwargs) == ['content', 'title']
        assert '<alas>' in kwargs['title']
        assert '<alas>' in kwargs['content']
        assert 'repeated failures' in kwargs['content']

    def test_honoured_taskless_pause_sends_push_notification(self, notifier):
        instance = make_instance()
        instance._apply_directive({'action': 'pause', 'reason': 'user asked'})
        assert instance._ai_pause is True
        notifier.assert_called_once()
        assert 'user asked' in notifier.call_args[1]['content']

    def test_pause_without_reason_still_notifies(self, notifier):
        instance = make_instance()
        instance._apply_directive({'action': 'pause'})
        assert instance._ai_pause is True
        notifier.assert_called_once()
        assert 'AI requested pause' in notifier.call_args[1]['content']

    def test_pause_deferred_to_alas_recovery_sends_nothing(self, notifier):
        """A pause that is not honoured stops nothing, so there is nothing to alert about."""
        instance = make_instance()
        instance._apply_directive(
            {'action': 'pause', 'reason': 'looks stuck'}, task='Commission')
        assert instance._ai_pause is False
        notifier.assert_not_called()

    def test_failed_notification_does_not_stop_the_pause(self, notifier):
        notifier.side_effect = RuntimeError('push provider down')
        instance = make_instance()
        instance._apply_directive({'action': 'pause', 'reason': 'user asked'})
        assert instance._ai_pause is True
        notifier.assert_called_once()

    def test_unreadable_notification_config_does_not_stop_the_pause(self, notifier):
        instance = make_instance()

        class BrokenConfig:
            @property
            def Error_OnePushConfig(self):
                raise RuntimeError('config unreadable')
        instance.__dict__['config'] = BrokenConfig()
        instance._apply_directive({'action': 'pause', 'reason': 'user asked'})
        assert instance._ai_pause is True
        notifier.assert_not_called()


class TestPauseStopsLoop:
    def test_loop_exits_immediately_when_ai_pause_set(self):
        """loop() must return right away when _ai_pause is already True.

        If the pause check were missing, loop() would reach self.checker and
        the cached_property would build a real ServerChecker (SystemExit/hang).
        """
        instance = AzurLaneAutoScript(config_name='alas')
        # No GUI event: pause must not depend on stop_event to exit the loop
        instance.stop_event = None
        # Inject a mock config so the cached_property doesn't load a real
        # config file. MagicMock makes AiSidecar_Enabled truthy, which is fine:
        # AISidecarClient.__init__ only reads attributes, and no notify()
        # happens before the pause break.
        instance.__dict__['config'] = MagicMock()
        instance._ai_pause = True

        result = instance.loop()

        assert result is None


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


class TestClampSkipDelay:
    def test_commission_is_clamped_below_the_12h_ceiling(self):
        """Commission caps at 12h; ALAS resets an over-cap NextRun to now."""
        instance = make_instance()
        assert instance._clamp_skip_delay('Commission', 1440) == 719

    def test_reward_is_clamped_below_the_12h_ceiling(self):
        instance = make_instance()
        assert instance._clamp_skip_delay('Reward', 5000) == 719

    def test_unlisted_task_is_clamped_below_the_24h_ceiling(self):
        instance = make_instance()
        assert instance._clamp_skip_delay('Dorm', 1440) == 1439

    def test_research_uses_the_24h_ceiling(self):
        instance = make_instance()
        assert instance._clamp_skip_delay('Research', 99999) == 1439

    def test_opsi_archive_uses_the_7d_ceiling(self):
        instance = make_instance()
        assert instance._clamp_skip_delay('OpsiArchive', 99999) == 7 * 24 * 60 - 1

    def test_value_under_the_ceiling_is_untouched(self):
        instance = make_instance()
        assert instance._clamp_skip_delay('Commission', 90) == 90

    def test_value_below_the_floor_is_raised(self):
        """A tiny delay would busy-loop the scheduler."""
        instance = make_instance()
        assert instance._clamp_skip_delay('Commission', 1) == 15

    def test_numeric_string_is_accepted(self):
        instance = make_instance()
        assert instance._clamp_skip_delay('Commission', '90') == 90

    def test_unparseable_value_returns_none(self):
        instance = make_instance()
        assert instance._clamp_skip_delay('Commission', 'soon') is None
        assert instance._clamp_skip_delay('Commission', None) is None
