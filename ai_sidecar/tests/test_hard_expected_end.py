"""Hard battle dispatch and combat end detection using the real campaign loader."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from campaign.campaign_hard.campaign_hard import Campaign as HardCampaign
from campaign.campaign_main import campaign_3_2
from module.base.base import ModuleBase
from module.hard.hard import CampaignHard
from module.map.fleet import Fleet


@pytest.fixture
def hard_campaign(monkeypatch):
    # Keep the real loader/classes, but avoid config, OCR and device startup.
    def init(self, config, device=None, task=None):
        self.config = config
        self.device = device

    monkeypatch.setattr(ModuleBase, '__init__', init)
    config = MagicMock()
    config.merge.return_value = SimpleNamespace()
    runner = CampaignHard(config=config, device=object())
    runner.load_campaign(name='campaign_hard', folder='campaign_hard')
    # CampaignHard.run assigns only the main campaign's MAP after loading.
    runner.campaign.MAP = campaign_3_2.MAP
    campaign = runner.campaign
    campaign.map = campaign.MAP
    campaign.battle_count = 0
    assert type(campaign) is HardCampaign
    assert campaign._expected_end.__func__ is HardCampaign._expected_end
    assert campaign_3_2.Campaign not in type(campaign).__mro__
    assert super(HardCampaign, campaign)._expected_end.__func__ is Fleet._expected_end
    return campaign


def test_first_hard_3_2_mob_returns_to_map(hard_campaign):
    base_end = Fleet._expected_end(hard_campaign, 'enemy')
    assert base_end == 'with_searching'
    assert hard_campaign._expected_end('enemy') == base_end


@pytest.mark.parametrize('expected', ['boss', 'boss_special'])
@pytest.mark.parametrize('battle_count', [0, 3])
def test_hard_boss_returns_to_stage(hard_campaign, expected, battle_count):
    hard_campaign.battle_count = battle_count
    assert hard_campaign._expected_end(expected) == 'in_stage'


@pytest.mark.parametrize('battle_count', [0, 6])
def test_hard_switches_from_mobs_to_visible_boss(hard_campaign, monkeypatch, battle_count):
    campaign = hard_campaign
    campaign.map = deepcopy(campaign.MAP)
    campaign.map.reset()
    campaign.battle_count = battle_count
    campaign.config.POOR_MAP_DATA = False
    campaign.config.MAP_CLEAR_ALL_THIS_TIME = False
    campaign.config.Error_HandleError = False
    clear_enemy = MagicMock(return_value=True)
    clear_boss = MagicMock(return_value=True)
    monkeypatch.setattr(campaign, 'clear_enemy', clear_enemy)
    monkeypatch.setattr(campaign, 'brute_clear_boss', clear_boss)

    assert not campaign.map.select(is_boss=True)
    assert campaign.execute_a_battle() is True
    clear_enemy.assert_called_once_with()
    clear_boss.assert_not_called()

    # The live failure occurred when H1 became the boss after six mob fights.
    campaign.map[(7, 0)].is_boss = True
    clear_enemy.reset_mock()
    clear_enemy.return_value = False
    assert campaign.execute_a_battle() is True
    clear_boss.assert_called_once_with()
    clear_enemy.assert_not_called()
