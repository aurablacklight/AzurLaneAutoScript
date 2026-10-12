"""Regression tests against real ALAS configuration, isolated from live profiles."""
import copy
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import alas
import module.config.config as config_module
import module.config.config_updater as updater_module
from alas import AzurLaneAutoScript
from module.config.config import AzurLaneConfig
from module.config.config_updater import ConfigGenerator
from module.config.deep import deep_get, deep_set
from module.exception import GameStuckError, GameTooManyClickError

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 9, 8, 22, 0, 0)


class ClockMeta(type):
    def __instancecheck__(cls, value):
        return isinstance(value, datetime)


class Clock(datetime, metaclass=ClockMeta):
    @classmethod
    def now(cls, tz=None):
        return NOW


def sample_time(value, **kwargs):
    if value == '30-60':
        return 45
    if value == '120-240':
        return 180
    return float(value)


@pytest.fixture
def profile(tmp_path, monkeypatch):
    # Templates are tracked, non-secret seed data; never read config/alas.json.
    seed = json.loads((ROOT / 'config/template.json').read_text())
    for task in seed.values():
        if 'Scheduler' in task:
            task['Scheduler']['Enable'] = False
            task['Scheduler']['NextRun'] = str(NOW + timedelta(hours=6))
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'config').mkdir()
    monkeypatch.setattr(updater_module, 'filepath_args', lambda filename='args', **kw:
                        str(ROOT / 'module/config/argument' / (filename + '.json')))
    monkeypatch.setattr(updater_module, 'filepath_argument', lambda filename, **kw:
                        str(ROOT / 'module/config/argument' / (filename + '.yaml')))
    monkeypatch.setattr(config_module, 'datetime', Clock)
    monkeypatch.setattr(alas, 'datetime', Clock)
    monkeypatch.setattr(config_module, 'ensure_time', sample_time)
    monkeypatch.setattr(alas, 'ensure_time', sample_time, raising=False)
    monkeypatch.setattr(AzurLaneConfig, 'is_hoarding_task', False)

    def create(task='Research', enabled=True, next_minutes=-1, extra=None):
        data = copy.deepcopy(seed)
        data[task]['Scheduler'].update(Enable=enabled, NextRun=str(NOW + timedelta(minutes=next_minutes)))
        for name, values in (extra or {}).items():
            data[name]['Scheduler'].update(values)
        path = tmp_path / 'config/regression.json'
        path.write_text(json.dumps(data))
        config = AzurLaneConfig('regression', task=task)
        runner = AzurLaneAutoScript('regression')
        runner.__dict__['config'] = config
        return runner, path
    return create


@pytest.mark.parametrize('enabled', [False, True])
def test_research_enable_survives_save_reload_and_new_instance(profile, enabled):
    runner, path = profile(enabled=enabled)
    assert runner.config.is_task_enabled('Research') is enabled
    runner.config.modified['Research.Scheduler.Enable'] = enabled
    runner.config.update()
    assert json.loads(path.read_text())['Research']['Scheduler']['Enable'] is enabled
    reloaded = AzurLaneConfig('regression', task='Research')
    assert reloaded.is_task_enabled('Research') is enabled
    reloaded.get_next_task()
    assert ('Research' in [t.command for t in reloaded.pending_task]) is enabled
    assert reloaded.is_task_enabled('Commission')
    assert reloaded.is_task_enabled('Reward')


def test_generated_research_enable_matches_editable_source(profile):
    profile()
    generated = ConfigGenerator().args['Research']['Scheduler']['Enable']
    checked_in = json.loads((ROOT / 'module/config/argument/args.json').read_text())['Research']['Scheduler']['Enable']
    assert generated == checked_in
    assert generated['type'] == 'checkbox'
    assert generated['option'] == [True, False]
    assert generated.get('display') != 'hide'


@pytest.mark.parametrize('task,proposed,expected', [
    ('Research', 1440, 1439), ('Commission', 1440, 719),
    ('Research', 90, 90), ('Research', 1, 15), ('Research', -5, 15),
    ('Research', '90', 90), ('Research', None, 45),
    ('Research', 'tomorrow', 45), ('Research', True, 45),
    ('Research', float('inf'), 45), ('Research', 1.5, 45),
])
def test_skip_deadline_survives_actual_save_reload(profile, task, proposed, expected):
    runner, path = profile(task=task)
    runner._apply_directive({'action': 'skip', 'task': task, 'delay_minutes': proposed}, task=task)
    reloaded = AzurLaneConfig('regression', task=task)
    assert deep_get(reloaded.data, f'{task}.Scheduler.NextRun') == NOW + timedelta(minutes=expected)


@pytest.mark.parametrize('proposed', [60, None, 'invalid'])
def test_skip_never_shortens_existing_later_deadline(profile, proposed):
    runner, _ = profile(next_minutes=120)
    runner._apply_directive({'action': 'skip', 'task': 'Research', 'delay_minutes': proposed}, task='Research')
    assert deep_get(AzurLaneConfig('regression').data, 'Research.Scheduler.NextRun') == NOW + timedelta(minutes=120)


def test_long_skip_extends_two_hour_deadline(profile):
    runner, _ = profile(next_minutes=120)
    runner._apply_directive({'action': 'skip', 'delay_minutes': 1440}, task='Research')
    assert deep_get(AzurLaneConfig('regression').data, 'Research.Scheduler.NextRun') == NOW + timedelta(minutes=1439)


def test_bound_task_wins_over_mismatched_directive(profile):
    runner, _ = profile(task='Commission')
    original_dorm = deep_get(runner.config.data, 'Dorm.Scheduler.NextRun')
    runner._apply_directive({'action': 'skip', 'task': 'Dorm', 'delay_minutes': 90}, task='Commission')
    assert deep_get(runner.config.data, 'Commission.Scheduler.NextRun') == NOW + timedelta(minutes=90)
    assert deep_get(runner.config.data, 'Dorm.Scheduler.NextRun') == original_dorm


def test_unbound_skip_uses_target_tasks_failure_interval(profile):
    runner, _ = profile(task='Tactical')
    runner.config.init_task('Alas')
    runner._apply_directive({'action': 'skip', 'task': 'Tactical'})
    assert deep_get(runner.config.data, 'Tactical.Scheduler.NextRun') == NOW + timedelta(minutes=180)


@pytest.mark.parametrize('task', ['Research', 'NoSuchTask', '', None])
def test_skip_cannot_change_disabled_or_unknown_tasks(profile, task):
    runner, path = profile(enabled=False)
    before = path.read_bytes()
    runner._apply_directive({'action': 'skip', 'task': task, 'delay_minutes': 90})
    assert path.read_bytes() == before


@pytest.mark.parametrize('directive', [None, {}, [], 'skip', {'action': 'reprioritize', 'task_order': 'Research'}])
def test_malformed_or_absent_directive_is_inert(profile, directive):
    runner, path = profile()
    before = path.read_bytes()
    runner._apply_directive(directive)
    assert path.read_bytes() == before
    assert not runner._ai_pause


def test_reprioritize_changes_ready_order_without_changing_deadlines(profile):
    runner, path = profile(next_minutes=120, extra={
        'Commission': {'Enable': True, 'NextRun': str(NOW - timedelta(minutes=1))},
        'Dorm': {'Enable': True, 'NextRun': str(NOW - timedelta(minutes=1))},
        'Tactical': {'Enable': False, 'NextRun': str(NOW - timedelta(minutes=1))},
        'Restart': {'Enable': True, 'NextRun': str(NOW - timedelta(minutes=1))},
    })
    before = path.read_bytes()
    runner._apply_directive({'action': 'reprioritize', 'task_order': ['Research', 'Tactical', 'Dorm', 'Dorm', None, {}, 'Unknown', 'Commission', 'Restart']})
    runner.config.get_next_task()
    assert [t.command for t in runner.config.pending_task] == ['Restart', 'Dorm', 'Commission']
    assert 'Research' in [t.command for t in runner.config.waiting_task]
    assert path.read_bytes() == before
    reloaded = AzurLaneConfig('regression')
    reloaded.get_next_task()
    assert [t.command for t in reloaded.pending_task] == ['Restart', 'Commission', 'Dorm']


def test_disabled_research_never_selected_while_ready_tasks_progress(profile):
    runner, _ = profile(enabled=False, extra={
        'Commission': {'Enable': True, 'NextRun': str(NOW - timedelta(minutes=1))},
        'Dorm': {'Enable': True, 'NextRun': str(NOW - timedelta(minutes=1))},
    })
    assert runner.config.get_next().command == 'Commission'
    runner.config.task_delay(minute=60, task='Commission')
    assert runner.config.get_next().command == 'Dorm'
    runner.config.task_delay(minute=60, task='Dorm')
    runner.config.get_next_task()
    assert 'Research' not in [t.command for t in runner.config.pending_task + runner.config.waiting_task]


@pytest.mark.parametrize('error', [GameStuckError, GameTooManyClickError])
@pytest.mark.parametrize('next_minutes,delay,expected', [(-1, 1440, 1439), (120, 60, 120)])
def test_real_stuck_handler_defers_confirmed_skip_without_restart(profile, error, next_minutes, delay, expected):
    runner, _ = profile(next_minutes=next_minutes)
    runner.__dict__['device'] = MagicMock()
    runner._sidecar = MagicMock()
    runner._sidecar.notify.return_value = {'action': 'skip', 'delay_minutes': delay}
    runner.save_error_log = lambda: None
    runner.research = MagicMock(side_effect=error('locked research'))
    runner.config.task_call = MagicMock()
    assert runner.run('research', skip_first_screenshot=True) is False
    runner.config.task_call.assert_not_called()
    runner.device.sleep.assert_not_called()
    assert runner._task_deferred
    assert deep_get(AzurLaneConfig('regression').data, 'Research.Scheduler.NextRun') == NOW + timedelta(minutes=expected)


@pytest.mark.parametrize('directive', [None, {}, [], {'action': 'continue'}])
def test_real_stuck_handler_retains_native_restart_without_skip(profile, directive):
    runner, _ = profile()
    runner.__dict__['device'] = MagicMock()
    runner._sidecar = MagicMock()
    runner._sidecar.notify.return_value = directive
    runner.save_error_log = lambda: None
    runner.research = MagicMock(side_effect=GameStuckError('locked research'))
    assert runner.run('research', skip_first_screenshot=True) is False
    assert runner.config.is_task_enabled('Restart')
    runner.device.sleep.assert_called_once_with(10)


@pytest.mark.parametrize('failure', ['disabled', 'write_raises', 'write_noop', 'pending_only', 'read_raises'])
def test_unconfirmed_skip_preserves_native_recovery(profile, failure):
    runner, path = profile(enabled=failure != 'disabled')
    runner.__dict__['device'] = MagicMock()
    runner._sidecar = MagicMock()
    runner._sidecar.notify.return_value = {'action': 'skip', 'delay_minutes': 90}
    runner.save_error_log = lambda: None
    runner.research = MagicMock(side_effect=GameTooManyClickError('locked research'))
    config = runner.config
    config.task_call = MagicMock()
    if failure == 'write_raises':
        config.write_file = MagicMock(side_effect=OSError('write failed'))
    elif failure == 'write_noop':
        config.write_file = MagicMock()
    elif failure == 'pending_only':
        def pending_delay(target, task):
            config.modified[f'{task}.Scheduler.NextRun'] = target
            deep_set(config.data, f'{task}.Scheduler.NextRun', target)
        config.task_delay = pending_delay
    elif failure == 'read_raises':
        read_file = config.read_file
        config.read_file = MagicMock(side_effect=[read_file('regression'), read_file('regression'), OSError('read failed')])
    assert runner.run('research', skip_first_screenshot=True) is False
    assert not runner._task_deferred
    config.task_call.assert_called_once_with('Restart')
    runner.device.sleep.assert_called_once_with(10)


def test_deferred_flag_does_not_suppress_next_run_recovery(profile):
    runner, _ = profile()
    runner.__dict__['device'] = MagicMock()
    runner._sidecar = MagicMock()
    runner._sidecar.notify.side_effect = [{'action': 'skip', 'delay_minutes': 90}, None]
    runner.save_error_log = lambda: None
    runner.research = MagicMock(side_effect=GameTooManyClickError('locked research'))
    runner.config.task_call = MagicMock()
    assert runner.run('research') is False
    assert runner._task_deferred
    assert runner.run('research') is False
    assert not runner._task_deferred
    runner.config.task_call.assert_called_once_with('Restart')
    runner.device.sleep.assert_called_once_with(10)


def test_scheduler_proceeds_after_deferred_task_without_result_or_failure(profile, monkeypatch):
    runner, _ = profile(extra={'Dorm': {'Enable': True, 'NextRun': str(NOW - timedelta(minutes=1))}})
    runner.config.override(AiSidecar_Enabled=True, Error_HandleError=False)
    runner.__dict__['device'] = MagicMock()
    checker = MagicMock()
    checker.is_recovered.return_value = False
    runner.__dict__['checker'] = checker
    runner.stop_event = threading.Event()
    runner.failure_record['Research'] = 2
    runner.save_error_log = lambda: None
    runner.research = MagicMock(side_effect=GameTooManyClickError('locked research'))
    runner.dorm = MagicMock(side_effect=runner.stop_event.set)
    tasks = iter(['Research', 'Dorm'])
    def next_task():
        task = next(tasks)
        runner.config.init_task(task)
        return task
    runner.get_next_task = next_task
    client = MagicMock()
    client.notify.side_effect = lambda event, payload: {'action': 'skip', 'delay_minutes': 90} if event == 'unknown_state' else None
    monkeypatch.setattr(alas, 'AISidecarClient', lambda _: client)
    monkeypatch.setattr(alas.logger, 'set_file_logger', lambda _: None)
    runner.loop()
    runner.dorm.assert_called_once()
    assert runner.failure_record['Research'] == 2
    results = [(call.args[0], call.args[1]['task']) for call in client.notify.call_args_list
               if call.args[0] in ('task_complete', 'task_failed')]
    assert results == [('task_complete', 'Dorm')]
    assert not runner._task_deferred
    assert deep_get(AzurLaneConfig('regression').data, 'Research.Scheduler.NextRun') == NOW + timedelta(minutes=90)


def test_native_three_failure_stop_with_unavailable_sidecar(monkeypatch):
    runner = AzurLaneAutoScript('fake_recovery')
    config = MagicMock()
    config.AiSidecar_Enabled = True
    config.Error_HandleError = True
    # Recreate the fake config on cache invalidation without reading any real profile.
    monkeypatch.setattr(AzurLaneAutoScript, 'config', property(lambda self: config))
    runner.__dict__['device'] = MagicMock()
    checker = MagicMock()
    checker.is_recovered.return_value = False
    runner.__dict__['checker'] = checker
    runner.get_next_task = MagicMock(return_value='Research')
    runner.run = MagicMock(return_value=False)
    client = MagicMock()
    client.notify.return_value = None
    monkeypatch.setattr(alas, 'AISidecarClient', lambda _: client)
    monkeypatch.setattr(alas, 'handle_notify', MagicMock())
    with pytest.raises(SystemExit) as stopped:
        runner.loop()
    assert stopped.value.code == 1
    assert runner.run.call_count == 3
    assert runner.failure_record['Research'] == 3
    assert not runner._ai_pause


def _task_failed_runner(monkeypatch, run_results, notify, device=None, handle_error=True):
    """Loop-level runner on a fake config: no profile, no device, no network."""
    runner = AzurLaneAutoScript('fake_task_failed')
    config = MagicMock()
    config.AiSidecar_Enabled = True
    config.Error_HandleError = handle_error
    monkeypatch.setattr(AzurLaneAutoScript, 'config', property(lambda self: config))
    runner.__dict__['device'] = device if device is not None else MagicMock()
    checker = MagicMock()
    checker.is_recovered.return_value = False
    runner.__dict__['checker'] = checker
    runner.stop_event = threading.Event()
    runner.get_next_task = MagicMock(return_value='Research')
    results = iter(run_results)

    def run(command, skip_first_screenshot=False):
        result = next(results)
        if result == 'stop':
            runner.stop_event.set()
            return True
        return result
    runner.run = MagicMock(side_effect=run)
    client = MagicMock()
    client.notify.side_effect = notify
    monkeypatch.setattr(alas, 'AISidecarClient', lambda _: client)
    monkeypatch.setattr(alas.logger, 'set_file_logger', lambda _: None)
    return runner, client


def _task_failed_calls(client):
    return [call.args[1] for call in client.notify.call_args_list if call.args[0] == 'task_failed']


def test_task_failed_carries_last_stored_screenshot_without_new_capture(monkeypatch):
    device = MagicMock()
    device.image = 'last-frame'
    runner, client = _task_failed_runner(monkeypatch, [False], lambda event, payload: None,
                                         device=device, handle_error=False)
    client.screenshot_to_base64.side_effect = lambda image: 'b64:' + image
    runner.loop()
    sent = _task_failed_calls(client)
    assert len(sent) == 1
    assert sent[0]['screenshot'] == 'b64:last-frame'
    assert sent[0]['task'] == 'Research'
    assert sent[0]['error'] == 'task_returned_failure'
    assert sent[0]['count'] == 0
    # The only fresh capture in the loop is the existing cycle_start one; task_failed
    # reads the stored frame, because device.screenshot() runs stuck detection.
    assert device.screenshot.call_count == 1
    # cycle_start encodes its own fresh frame, task_failed re-encodes the stored one.
    assert client.screenshot_to_base64.call_count == 2
    client.screenshot_to_base64.assert_called_with('last-frame')


@pytest.mark.parametrize('failure', ['encode_raises', 'encode_none', 'no_image'])
def test_task_failed_still_posts_without_screenshot(monkeypatch, failure):
    device = MagicMock()
    device.image = None if failure == 'no_image' else 'last-frame'
    runner, client = _task_failed_runner(monkeypatch, [False], lambda event, payload: None,
                                         device=device, handle_error=False)
    if failure == 'encode_raises':
        client.screenshot_to_base64.side_effect = RuntimeError('encode failed')
    else:
        client.screenshot_to_base64.return_value = None
    runner.loop()
    sent = _task_failed_calls(client)
    assert len(sent) == 1
    assert 'screenshot' not in sent[0]
    assert sent[0]['task'] == 'Research'
    assert sent[0]['error'] == 'task_returned_failure'


def test_task_failed_does_not_build_a_device_to_find_a_frame(monkeypatch):
    runner = AzurLaneAutoScript('fake_no_device')
    built = []
    monkeypatch.setattr(AzurLaneAutoScript, 'device', property(lambda self: built.append(1)))
    runner._sidecar = MagicMock()
    assert 'device' not in runner.__dict__
    assert runner._last_screenshot_base64() is None
    assert not built
    runner._sidecar.screenshot_to_base64.assert_not_called()


def test_task_failed_hook_error_does_not_stop_scheduler(monkeypatch):
    device = MagicMock()
    device.image = 'last-frame'

    def notify(event, payload):
        if event == 'task_failed':
            raise RuntimeError('sidecar exploded')
        return None
    runner, client = _task_failed_runner(monkeypatch, [False, 'stop'], notify, device=device)
    client.screenshot_to_base64.return_value = 'b64'
    runner.loop()
    assert runner.run.call_count == 2
    assert len(_task_failed_calls(client)) == 1
    # Failure accounting still ran after the hook raised, and the next task succeeded.
    assert runner.failure_record['Research'] == 0
