"""A long Skip must survive reprioritization and real config reloads."""
import json
from datetime import datetime, timedelta
from pathlib import Path

import module.config.config_updater as updater_module
from alas import AzurLaneAutoScript
from module.config.config import AzurLaneConfig
from module.config.deep import deep_get

ROOT = Path(__file__).resolve().parents[2]


def test_commission_skip_survives_reprioritization_and_reload(tmp_path, monkeypatch):
    # Never read or write the live profile; all writes stay inside tmp_path.
    seed = json.loads((ROOT / 'config/template.json').read_text())
    seed['Commission']['Scheduler']['NextRun'] = '2020-01-01 00:00:00'
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config/regression.json').write_text(json.dumps(seed))
    monkeypatch.setattr(updater_module, 'filepath_args', lambda filename='args', **kw:
                        str(ROOT / 'module/config/argument' / (filename + '.json')))
    monkeypatch.setattr(AzurLaneConfig, 'is_hoarding_task', False)
    runner = AzurLaneAutoScript('regression')
    runner.__dict__['config'] = AzurLaneConfig('regression', task='Commission')
    before = datetime.now().replace(microsecond=0)
    runner._apply_directive(
        {'action': 'skip', 'task': 'Commission', 'delay_minutes': 1440}, task='Commission')
    deadline = deep_get(runner.config.data, 'Commission.Scheduler.NextRun')
    assert before + timedelta(minutes=719) <= deadline <= datetime.now() + timedelta(minutes=719)
    runner._apply_directive({'action': 'reprioritize', 'task_order': ['Commission']})
    reloaded = AzurLaneConfig('regression', task='Commission')
    assert deep_get(reloaded.data, 'Commission.Scheduler.NextRun') == deadline
    reloaded.get_next_task()
    assert 'Commission' not in [task.command for task in reloaded.pending_task]
