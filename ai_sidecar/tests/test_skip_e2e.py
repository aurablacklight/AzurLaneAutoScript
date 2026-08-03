"""End-to-end: a skip directive must survive a config reload.

ALAS clamps Scheduler.NextRun on every load and resets an over-ceiling value
to *now*. Asserting NextRun right after task_delay passes even when the value
is doomed, so this test reloads the config before asserting.

Scope -- what this test does NOT cover: it proves a skip survives a config
reload via the success=False FailureInterval candidate in task_delay()'s
min(run) selection. It does NOT exercise the per-task ceiling clamp
(_clamp_skip_delay) itself: the task bound here is Alas/General (see
AzurLaneConfig.init_task's default bind), whose Scheduler_FailureInterval
(~120 minutes) is always smaller than the clamped 719-minute Commission
ceiling, so FailureInterval always wins task_delay's min() and the clamped
ceiling value is never the one actually written to NextRun. Do not "fix"
this by binding Commission directly -- Commission's own FailureInterval is a
randomized 30-60 minute range, which can fall under this test's 30-minute
assertion threshold and make the test flaky. The clamp itself (every
ceiling, the floor, boundaries, numeric strings, unparseable input) is
covered directly by TestClampSkipDelay in test_alas_directive.py; that is
where clamp regressions would be caught.

Isolation: this test never reads config/alas.json for anything other than
seeding a throwaway copy, and never writes to it. The `isolated_config`
fixture copies it to config/pytest_skip_e2e.json, the test operates only
against that throwaway name, and the fixture deletes it afterward -- safe to
run any number of times, by anyone, on any machine, with zero effect on the
live config.
"""
import os
import shutil
import sys
from datetime import datetime, timedelta

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from alas import AzurLaneAutoScript
from module.config.config import AzurLaneConfig
from module.config.deep import deep_get

# Deliberately cwd-relative, matching module.config.utils.filepath_config's
# own convention (./config/<name>.json) -- ALAS resolves config paths against
# the process cwd, not this file's location, so this fixture must too.
_LIVE_CONFIG_PATH = './config/alas.json'
_THROWAWAY_CONFIG_NAME = 'pytest_skip_e2e'
_THROWAWAY_CONFIG_PATH = f'./config/{_THROWAWAY_CONFIG_NAME}.json'


@pytest.fixture
def isolated_config():
    """Seed a throwaway config from the live one; never touch the live file.

    Copies config/alas.json -> config/pytest_skip_e2e.json before the test
    runs (a read of the live file, no write), yields the throwaway config
    name for the test to construct AzurLaneConfig/AzurLaneAutoScript against,
    then deletes the throwaway file on teardown regardless of test outcome.
    """
    shutil.copyfile(_LIVE_CONFIG_PATH, _THROWAWAY_CONFIG_PATH)
    try:
        yield _THROWAWAY_CONFIG_NAME
    finally:
        if os.path.exists(_THROWAWAY_CONFIG_PATH):
            os.remove(_THROWAWAY_CONFIG_PATH)


def test_commission_skip_survives_a_config_reload(isolated_config):
    config_name = isolated_config
    instance = AzurLaneAutoScript(config_name=config_name)
    instance.__dict__['config'] = AzurLaneConfig(config_name=config_name)

    instance._apply_directive(
        {'action': 'skip', 'task': 'Commission', 'delay_minutes': 1440},
        task='Commission')

    reloaded = AzurLaneConfig(config_name=config_name)
    next_run = deep_get(
        reloaded.data, keys='Commission.Scheduler.NextRun', default=None)

    assert isinstance(next_run, datetime)
    assert next_run > datetime.now() + timedelta(minutes=30), (
        f'Commission skip collapsed to {next_run}; the 12h ceiling reset it to now'
    )
