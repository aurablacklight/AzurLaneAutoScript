"""
Device-free checks of the area the oil amount is read from.

On EN a five-digit amount reaches the right edge of OCR_OIL and the sidecar's model drops
digits (22364 read as 24). The EN area is 8 px wider; other servers keep the stock area.

Run from the repository root: python -m pytest ai_sidecar/tests/test_oil_area.py
"""
from types import SimpleNamespace

import numpy as np
import pytest

import module.campaign.campaign_status as campaign_status
from module.campaign.assets import OCR_OIL, OCR_OIL_CHECK
from module.campaign.campaign_status import CampaignStatus

STOCK = tuple(OCR_OIL.area)


@pytest.fixture
def on_server(monkeypatch):
    def use(name):
        monkeypatch.setattr(campaign_status.server, 'server', name)
    return use


def test_en_area_is_eight_pixels_wider_on_the_right(on_server):
    on_server('en')
    x1, y1, x2, y2 = STOCK
    assert CampaignStatus._oil_area() == (x1, y1, x2 + 8, y2)


@pytest.mark.parametrize('name', ['cn', 'jp', 'tw'])
def test_other_servers_keep_the_stock_area(on_server, name):
    on_server(name)
    assert CampaignStatus._oil_area() == STOCK


def test_the_shared_button_is_not_changed(on_server):
    on_server('en')
    CampaignStatus._oil_area()
    assert tuple(OCR_OIL.area) == STOCK


@pytest.mark.parametrize('name, expected_x2', [('en', STOCK[2] + 8), ('cn', STOCK[2]), ('jp', STOCK[2])])
def test_get_oil_reads_from_that_area(on_server, monkeypatch, name, expected_x2):
    # `_get_oil` builds the reader itself; record what it is built with.
    on_server(name)
    built = []

    class RecordingDigit:
        def __init__(self, buttons, **kwargs):
            built.append((buttons, kwargs))

        def ocr(self, image):
            return 22843

    monkeypatch.setattr(campaign_status, 'Digit', RecordingDigit)
    monkeypatch.setattr(campaign_status, 'get_color', lambda image, area: OCR_OIL_CHECK.color)

    status = object.__new__(CampaignStatus)
    status.appear = lambda button, **kwargs: True
    status.device = SimpleNamespace(image=np.zeros((720, 1280, 3), dtype=np.uint8))

    assert status._get_oil() == 22843
    assert len(built) == 1
    area, kwargs = built[0]
    assert area == (STOCK[0], STOCK[1], expected_x2, STOCK[3])
    assert kwargs['name'] == 'OCR_OIL'
