"""
Device-free checks of the "Download Game Assets" dialog the EN client shows when it starts.

Seen first on 2026-11-18, after the Play Store updated the game to 9.3.23. The dialog's red X
matches MAINTENANCE_ANNOUNCE, the close button of a maintenance notice, and on this dialog the X
exits the game: ALAS tapped it, the game closed, and the Restart task failed every time. The
two frames are that dialog over two different loading pictures, cut down to the dialog itself.

Run from the repository root: python -m pytest ai_sidecar/tests/test_login_assets_dialog.py
"""
import logging
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

import module.config.server as server
from module.base.base import ModuleBase
from module.base.utils import load_image
from module.handler.assets import LOGIN_ASSETS_DOWNLOAD, LOGIN_ASSETS_DOWNLOAD_ZERO, MAINTENANCE_ANNOUNCE
from module.handler.login import LoginHandler
from module.logger import logger

ROOT = Path(__file__).resolve().parents[2]
FRAMES = Path(__file__).resolve().parent / 'frames'
DIALOGS = ['assets_download_dialog_sakura.png', 'assets_download_dialog_eagle.png']
# Where things are on the dialog, in the 1280x720 frame.
CONFIRM = (700, 447, 873, 507)
TITLE = (312, 153, 892, 199)
SIZE = (650, 316, 732, 350)


def frame(name):
    return load_image(str(FRAMES / name))


def painted(image, area, colour):
    image = image.copy()
    x1, y1, x2, y2 = area
    image[y1:y2, x1:x2] = colour
    return image


def with_another_size(image):
    """The same dialog with other text where "(0B)" is: a download that is not empty."""
    image = image.copy()
    # "assets" from the same line, in the same font, moved over the size.
    x1, y1, x2, y2 = SIZE
    image[y1:y2, x1:x2] = image[y1:y2, x1 - 90:x2 - 90]
    return image


class Stop(Exception):
    """Raised by the fake device when its frames run out."""


@pytest.fixture
def login(monkeypatch):
    """A LoginHandler on ALAS's real assets with a fake config and device; nothing is written."""
    # Assets are loaded by a path relative to the repository root; read only.
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(server, 'server', 'en')
    # Otherwise ALAS starts its background OCR import.
    monkeypatch.setattr(ModuleBase, 'EARLY_OCR_IMPORT', True)
    config = MagicMock()
    config.SERVER = 'en'
    config.BUTTON_OFFSET = 30
    device = MagicMock()
    handler = LoginHandler(config=config, device=device)
    return handler


def show(handler, frames):
    """Make the fake device return these frames, one per screenshot, then stop the loop."""
    queue = iter(frames)

    def screenshot():
        try:
            handler.device.image = next(queue)
        except StopIteration:
            raise Stop()
        return handler.device.image

    handler.device.screenshot.side_effect = screenshot


def clicked(handler):
    return [call.args[0].name for call in handler.device.click.call_args_list]


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


def lines(records, level):
    return [record.getMessage() for record in records if record.levelno == level]


# --- The dialog is recognised ------------------------------------------------------------------

@pytest.mark.parametrize('name', DIALOGS)
def test_the_dialog_is_recognised_over_either_loading_picture(login, name):
    login.device.image = frame(name)
    assert login.appear(LOGIN_ASSETS_DOWNLOAD, offset=(30, 30))
    assert login.appear(LOGIN_ASSETS_DOWNLOAD_ZERO, offset=(30, 30), similarity=0.8)


@pytest.mark.parametrize('name', DIALOGS)
def test_the_maintenance_close_button_matches_the_dialog_too(login, name):
    # The cause of the failure: this is why the dialog has to be looked for first.
    login.device.image = frame(name)
    assert login.appear(MAINTENANCE_ANNOUNCE, offset=(30, 30))


def test_the_button_to_tap_is_inside_confirm(login):
    x1, y1, x2, y2 = LOGIN_ASSETS_DOWNLOAD.button
    assert CONFIRM[0] <= x1 < x2 <= CONFIRM[2]
    assert CONFIRM[1] <= y1 < y2 <= CONFIRM[3]


def test_other_screens_are_not_the_dialog(login):
    login.device.image = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert not login.handle_assets_download()
    login.device.image = np.random.RandomState(0).randint(0, 255, (720, 1280, 3)).astype(np.uint8)
    assert not login.handle_assets_download()
    assert clicked(login) == []


# --- What ALAS does with it --------------------------------------------------------------------

@pytest.mark.parametrize('name', DIALOGS)
def test_an_empty_download_is_confirmed(login, log, name):
    login.device.image = frame(name)
    assert login.handle_assets_download() is True
    assert clicked(login) == ['LOGIN_ASSETS_DOWNLOAD']
    assert lines(log, logging.INFO)[-1] == 'Download Game Assets dialog with nothing to download (0B), confirm'


def test_confirm_is_tapped_once_while_the_dialog_stays_up(login):
    # The dialog is still there on the next screenshots: no second tap inside the interval,
    # and still the caller is told the dialog is on screen.
    login.device.image = frame(DIALOGS[0])
    assert [login.handle_assets_download() for _ in range(3)] == [True, True, True]
    assert clicked(login) == ['LOGIN_ASSETS_DOWNLOAD']


def test_confirm_is_tapped_again_after_the_interval(login):
    login.device.image = frame(DIALOGS[0])
    login.handle_assets_download()
    login.get_interval_timer(LOGIN_ASSETS_DOWNLOAD, interval=5).clear()
    login.handle_assets_download()
    assert clicked(login) == ['LOGIN_ASSETS_DOWNLOAD', 'LOGIN_ASSETS_DOWNLOAD']


@pytest.mark.parametrize('name', DIALOGS)
def test_a_download_that_is_not_empty_is_left_to_a_human(login, log, name):
    # Nothing is tapped: not Confirm, which would start a download nobody chose, and not the X.
    login.device.image = with_another_size(frame(name))
    assert login.appear(LOGIN_ASSETS_DOWNLOAD, offset=(30, 30))
    assert [login.handle_assets_download() for _ in range(3)] == [True, True, True]
    assert clicked(login) == []
    assert lines(log, logging.WARNING) == [
        'Download Game Assets dialog: the size to download was not read as 0B, not confirming. '
        'Choose in the game what to download; ALAS taps nothing on this dialog']


# --- The login loop ----------------------------------------------------------------------------

@pytest.mark.parametrize('name', DIALOGS)
def test_the_login_loop_confirms_and_never_taps_the_x(login, name):
    dialog = frame(name)
    show(login, [dialog, dialog, dialog])
    with pytest.raises(Stop):
        login._handle_app_login()
    assert clicked(login) == ['LOGIN_ASSETS_DOWNLOAD']


@pytest.mark.parametrize('name', DIALOGS)
def test_the_login_loop_taps_nothing_when_the_download_is_not_empty(login, name):
    dialog = with_another_size(frame(name))
    show(login, [dialog, dialog, dialog])
    with pytest.raises(Stop):
        login._handle_app_login()
    assert clicked(login) == []


def test_the_login_loop_still_closes_a_maintenance_notice(login):
    # A screen with the close button and no dialog title: what MAINTENANCE_ANNOUNCE is for.
    notice = painted(frame(DIALOGS[0]), TITLE, 0)
    show(login, [notice])
    with pytest.raises(Stop):
        login._handle_app_login()
    assert clicked(login) == ['MAINTENANCE_ANNOUNCE']


# --- The page check every task starts with -----------------------------------------------------

@pytest.mark.parametrize('name', DIALOGS)
def test_the_page_check_confirms_and_never_taps_the_x(login, name):
    login.device.image = frame(name)
    assert [login.ui_additional() for _ in range(3)] == [True, True, True]
    assert clicked(login) == ['LOGIN_ASSETS_DOWNLOAD']


def test_the_page_check_taps_nothing_when_the_download_is_not_empty(login):
    login.device.image = with_another_size(frame(DIALOGS[1]))
    assert [login.ui_additional() for _ in range(3)] == [True, True, True]
    assert clicked(login) == []


def test_the_page_check_still_closes_a_maintenance_notice(login):
    login.device.image = painted(frame(DIALOGS[0]), TITLE, 0)
    assert login.ui_additional() is True
    assert clicked(login) == ['MAINTENANCE_ANNOUNCE']
