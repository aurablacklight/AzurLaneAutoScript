"""End-to-end: a skip directive must survive a config reload.

ALAS clamps Scheduler.NextRun on every load and resets an over-ceiling value
to *now*. Asserting NextRun right after task_delay passes even when the value
is doomed, so this test reloads the config before asserting.
"""
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from alas import AzurLaneAutoScript
from module.config.config import AzurLaneConfig
from module.config.deep import deep_get


def test_commission_skip_survives_a_config_reload():
    instance = AzurLaneAutoScript(config_name='alas')
    instance.__dict__['config'] = AzurLaneConfig(config_name='alas')

    instance._apply_directive(
        {'action': 'skip', 'task': 'Commission', 'delay_minutes': 1440},
        task='Commission')

    reloaded = AzurLaneConfig(config_name='alas')
    next_run = deep_get(
        reloaded.data, keys='Commission.Scheduler.NextRun', default=None)

    assert isinstance(next_run, datetime)
    assert next_run > datetime.now() + timedelta(minutes=30), (
        f'Commission skip collapsed to {next_run}; the 12h ceiling reset it to now'
    )
