"""
Tests for alas.py AI-sidecar directive handling:
- pause guard in _apply_directive (defer to ALAS auto-recovery when the task
  has fewer than 2 prior consecutive failures)
- enriched unknown_state payload (task / error / count) on GameStuckError
"""
import sys
import os
import threading
from datetime import datetime, timedelta
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

    def test_skip_directive_still_works_with_new_signature(self):
        """skip directive keeps working when a task kwarg is passed."""
        instance = make_instance()
        instance.__dict__['config'] = MagicMock()
        instance.config.data = {}
        instance._apply_directive(
            {'action': 'skip', 'task': 'Commission'}, task='Commission')
        instance.config.task_delay.assert_called_once_with(
            success=False, task='Commission')
        assert not instance.stop_event.is_set()


class TestSkipDelayHandling:
    def test_proposed_delay_is_clamped_and_passed_with_success_false(self):
        instance = make_instance()
        instance.__dict__['config'] = MagicMock()
        instance.config.data = {}
        instance._apply_directive(
            {'action': 'skip', 'task': 'Commission', 'delay_minutes': 1440},
            task='Commission')
        instance.config.task_delay.assert_called_once_with(
            minute=719, success=False, task='Commission')

    def test_reasonable_delay_passes_through(self):
        instance = make_instance()
        instance.__dict__['config'] = MagicMock()
        instance.config.data = {}
        instance._apply_directive(
            {'action': 'skip', 'task': 'Commission', 'delay_minutes': 180},
            task='Commission')
        instance.config.task_delay.assert_called_once_with(
            minute=180, success=False, task='Commission')

    def test_absent_delay_falls_back_to_failure_interval(self):
        instance = make_instance()
        instance.__dict__['config'] = MagicMock()
        instance.config.data = {}
        instance._apply_directive(
            {'action': 'skip', 'task': 'Commission'}, task='Commission')
        instance.config.task_delay.assert_called_once_with(
            success=False, task='Commission')

    def test_unparseable_delay_falls_back_to_failure_interval(self):
        instance = make_instance()
        instance.__dict__['config'] = MagicMock()
        instance.config.data = {}
        instance._apply_directive(
            {'action': 'skip', 'task': 'Commission', 'delay_minutes': 'tomorrow'},
            task='Commission')
        instance.config.task_delay.assert_called_once_with(
            success=False, task='Commission')

    def test_skip_without_task_name_does_nothing(self):
        instance = make_instance()
        instance.__dict__['config'] = MagicMock()
        instance.config.data = {}
        instance._apply_directive({'action': 'skip'}, task=None)
        instance.config.task_delay.assert_not_called()

    def test_skip_uses_the_bound_task_not_the_directive_claim(self):
        """success= resolves FailureInterval from the BOUND task while writing
        to {task}.Scheduler.NextRun. Those must be the same task, or one task's
        interval lands in another's slot -- so the bound task must win even
        when the directive claims a different target."""
        instance = make_instance()
        instance.__dict__['config'] = MagicMock()
        instance.config.data = {}
        instance._apply_directive(
            {'action': 'skip', 'task': 'Dorm', 'delay_minutes': 60},
            task='Commission')
        kwargs = instance.config.task_delay.call_args[1]
        assert kwargs['task'] == 'Commission'

    def test_skip_falls_back_to_directive_task_when_unbound(self):
        """With no bound task, the directive's claimed target is the only
        signal available, so it is used as-is."""
        instance = make_instance()
        instance.__dict__['config'] = MagicMock()
        instance.config.data = {}
        instance._apply_directive(
            {'action': 'skip', 'task': 'Dorm', 'delay_minutes': 60},
            task=None)
        kwargs = instance.config.task_delay.call_args[1]
        assert kwargs['task'] == 'Dorm'


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


class TestNextRunGuard:
    def test_skip_does_not_overwrite_a_further_out_next_run(self):
        """A task that already scheduled itself further out keeps its value."""
        instance = make_instance()
        instance.__dict__['config'] = MagicMock()
        instance.config.data = {
            'Commission': {
                'Scheduler': {
                    'NextRun': datetime.now() + timedelta(minutes=600)
                }
            }
        }
        instance._apply_directive(
            {'action': 'skip', 'task': 'Commission', 'delay_minutes': 60},
            task='Commission')
        instance.config.task_delay.assert_not_called()

    def test_skip_applies_when_stored_next_run_is_sooner(self):
        instance = make_instance()
        instance.__dict__['config'] = MagicMock()
        instance.config.data = {
            'Commission': {
                'Scheduler': {
                    'NextRun': datetime.now() + timedelta(minutes=5)
                }
            }
        }
        instance._apply_directive(
            {'action': 'skip', 'task': 'Commission', 'delay_minutes': 60},
            task='Commission')
        instance.config.task_delay.assert_called_once_with(
            minute=60, success=False, task='Commission')

    def test_skip_applies_when_no_next_run_is_stored(self):
        instance = make_instance()
        instance.__dict__['config'] = MagicMock()
        instance.config.data = {}
        instance._apply_directive(
            {'action': 'skip', 'task': 'Commission', 'delay_minutes': 60},
            task='Commission')
        instance.config.task_delay.assert_called_once_with(
            minute=60, success=False, task='Commission')
