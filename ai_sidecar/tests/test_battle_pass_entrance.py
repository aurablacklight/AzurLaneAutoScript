"""
Device-free checks of the battle pass entrance on the reward page (EN).

The Freebies task looks for REWARD_GOTO_BATTLE_PASS (the "Cruise Missions" icon in the column
right of the reward list) right after the reward page is recognised. Since 2026-11-17 it was
found in 6 main-account runs and missed in 6. The EN server had no template of its own and used
the CN one, which scores 0.836 to 0.901 on the recorded EN reward pages against a threshold of
0.85. The icon sits on a see-through column over the main page, so the secretary and background
behind it change the score, and the column is still sliding in when ALAS first sees the page.

The frames are ALAS's own screenshots from live runs on 2026-11-17 and 2026-11-18 (the last
screenshot of the Tactical task, which ends on the reward page; a cycle_start hook frame; and one
screencap taken 0.75 s after a Reward task opened the page), cut down to the column at x 520-700,
y 0-400. None of them is the screenshot of a Freebies check that missed: no run kept one.
The EN template was cut from a fifth recording (2026-11-19) that is not among these frames.

Run from the repository root: python -m pytest ai_sidecar/tests/test_battle_pass_entrance.py
"""
import itertools
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import module.config.server as server
from module.base.base import ModuleBase
from module.base.button import Button
from module.base.utils import load_image
from module.freebies.assets import BATTLE_PASS_RED_DOT
from module.freebies.battle_pass import BattlePass
from module.logger import logger
from module.ui.assets import REWARD_GOTO_BATTLE_PASS

ROOT = Path(__file__).resolve().parents[2]
FRAMES = Path(__file__).resolve().parent / 'frames'
OFFSET = (50, 150)  # as battle_pass.py searches
THRESHOLD = 0.85  # ModuleBase.appear default
MARGIN = 0.05  # every recorded entrance must clear the threshold by at least this much

# Reward page with the entrance: frame -> red dot drawn on the entrance.
ENTRANCE = {
    'battle_pass_reward_20261118_assetsdialog_slide_dark.png': False,  # CN template 0.836
    'battle_pass_reward_20261118a_slide_red.png': False,  # CN template 0.848
    'battle_pass_reward_20261118d_slide.png': False,  # CN template 0.850
    'battle_pass_reward_20261118c_settled_toast.png': False,
    'battle_pass_reward_20261118_researchnames_slide_badge.png': True,
    'battle_pass_reward_20261118e_slide_badge.png': True,
    'battle_pass_reward_20261117a_settled.png': False,
    'battle_pass_reward_20261117b_reviewfixes_slide_dark.png': False,
}
OTHER_PAGES = [
    'battle_pass_other_main_page_20261118c.png',
    'battle_pass_other_main_page_20261118d.png',
    'battle_pass_other_dorm_20261118a.png',
    'battle_pass_other_commission_20261118a.png',
    'battle_pass_other_mail_20261117a.png',
]


def frame(name):
    return load_image(str(FRAMES / name))


class Stop(Exception):
    """Raised by the fake device when its frames run out."""


@pytest.fixture
def battle_pass(monkeypatch):
    """BattlePass on ALAS's real assets with a fake config and device; nothing is written."""
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(server, 'server', 'en')
    monkeypatch.setattr(ModuleBase, 'EARLY_OCR_IMPORT', True)
    config = MagicMock()
    config.SERVER = 'en'
    config.BUTTON_OFFSET = 30
    device = MagicMock()
    # Templates are cached per Button object; start from the files of this checkout.
    for button in (REWARD_GOTO_BATTLE_PASS, BATTLE_PASS_RED_DOT):
        button.resource_release()
        button.clear_offset()
    yield BattlePass(config=config, device=device)
    for button in (REWARD_GOTO_BATTLE_PASS, BATTLE_PASS_RED_DOT):
        button.resource_release()
        button.clear_offset()


def show(handler, frames):
    """Make the fake device return these frames, one per screenshot, then stop the loop."""
    queue = iter(frames)
    shots = []

    def screenshot():
        try:
            handler.device.image = next(queue)
        except StopIteration:
            raise Stop()
        shots.append(handler.device.image)
        return handler.device.image

    handler.device.screenshot.side_effect = screenshot
    return shots


@pytest.fixture
def log():
    records = []

    class Collect(logging.Handler):
        def emit(self, record):
            records.append(record)

    collector = Collect(level=logging.DEBUG)
    logger.addHandler(collector)
    yield records
    logger.removeHandler(collector)


def messages(records):
    return [record.getMessage() for record in records]


# --- The template --------------------------------------------------------------------------------

def test_en_has_its_own_entrance_template(battle_pass):
    assert REWARD_GOTO_BATTLE_PASS.file == './assets/en/ui/REWARD_GOTO_BATTLE_PASS.png'
    # Where the icon is once the column has settled on the EN client.
    assert REWARD_GOTO_BATTLE_PASS.area == (595, 101, 635, 139)


@pytest.mark.parametrize('name', ENTRANCE)
def test_the_entrance_is_found_with_margin(battle_pass, name):
    image = frame(name)
    battle_pass.device.image = image
    assert battle_pass.appear(REWARD_GOTO_BATTLE_PASS, offset=OFFSET)
    assert REWARD_GOTO_BATTLE_PASS.match(image, offset=OFFSET, similarity=THRESHOLD + MARGIN)


@pytest.mark.parametrize('name', ENTRANCE)
def test_the_entrance_is_found_on_the_icon_and_not_elsewhere_in_the_column(battle_pass, name):
    REWARD_GOTO_BATTLE_PASS.match(frame(name), offset=OFFSET)
    x1, y1, _, _ = REWARD_GOTO_BATTLE_PASS.button
    # The column was recorded between x=578 (sliding in) and x=595 (settled), always at y=101;
    # the next icon down (Specialized Core) is about 110 px lower.
    assert 575 <= x1 <= 598
    assert y1 == 101


@pytest.mark.parametrize('name', OTHER_PAGES)
def test_the_entrance_is_not_found_on_other_pages(battle_pass, name):
    image = frame(name)
    battle_pass.device.image = image
    assert not battle_pass.appear(REWARD_GOTO_BATTLE_PASS, offset=OFFSET)
    # Far below the threshold, not just under it.
    assert not REWARD_GOTO_BATTLE_PASS.match(image, offset=OFFSET, similarity=0.6)


def test_the_cn_template_used_before_was_below_the_threshold_on_a_recorded_page(monkeypatch):
    """What EN used before: the CN template, 0.836 on this page, so the entrance was missed."""
    monkeypatch.chdir(ROOT)
    cn = Button(area=(580, 117, 620, 155), color=(92, 113, 139), button=(580, 117, 620, 155),
                file='./assets/cn/ui/REWARD_GOTO_BATTLE_PASS.png')
    image = frame('battle_pass_reward_20261118_assetsdialog_slide_dark.png')
    assert not cn.match(image, offset=OFFSET, similarity=THRESHOLD)
    assert cn.match(image, offset=OFFSET, similarity=0.8)


# --- The red dot ---------------------------------------------------------------------------------

@pytest.mark.parametrize('name, red_dot', ENTRANCE.items())
def test_the_red_dot_is_read_next_to_the_entrance(battle_pass, log, name, red_dot):
    battle_pass.device.image = frame(name)
    assert battle_pass.battle_pass_red_dot_appear() is red_dot
    assert ('Found battle pass red dot' if red_dot else 'No battle pass red dot') in messages(log)


# --- Waiting for the column ----------------------------------------------------------------------

def test_the_entrance_is_looked_for_again_while_the_page_settles(battle_pass, log):
    """The first screenshots after "Page arrive" may not show the entrance yet."""
    battle_pass.device.image = frame('battle_pass_other_main_page_20261118c.png')
    later = [frame('battle_pass_other_main_page_20261118c.png'),
             frame('battle_pass_reward_20261118e_slide_badge.png')]
    shots = show(battle_pass, later)
    assert battle_pass.battle_pass_red_dot_appear() is True
    assert len(shots) == 2
    assert 'Battle pass entrance found on screenshot 3' in messages(log)
    assert 'No battle pass entrance' not in messages(log)


def test_an_entrance_that_never_shows_is_given_up_after_a_short_wait(battle_pass, log):
    other = frame('battle_pass_other_main_page_20261118c.png')
    battle_pass.device.image = other
    shots = show(battle_pass, itertools.repeat(other))
    assert battle_pass.battle_pass_red_dot_appear() is False
    # Message kept as before, so log queries for it still match.
    assert 'No battle pass entrance' in messages(log)
    assert len(shots) >= 2
    battle_pass.device.click.assert_not_called()


def test_an_entrance_on_the_first_screenshot_takes_no_new_screenshot(battle_pass):
    battle_pass.device.image = frame('battle_pass_reward_20261117a_settled.png')
    shots = show(battle_pass, [])
    assert battle_pass.battle_pass_red_dot_appear() is False
    assert shots == []
