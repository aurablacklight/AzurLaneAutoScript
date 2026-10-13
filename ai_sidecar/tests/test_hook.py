"""
Tests for AISidecarClient hook layer.

These tests run in Python 3.11 but test 3.7-compatible code.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "module", "ai_hook"))

import base64
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hook import AISidecarClient


class MockConfig:
    """Minimal config object to pass to AISidecarClient."""

    def __init__(self, enabled=True, url='http://sidecar:8484'):
        self.AiSidecar_Enabled = enabled
        self.AiSidecar_SidecarUrl = url


# ---------------------------------------------------------------------------
# screenshot_to_base64
# ---------------------------------------------------------------------------

def test_screenshot_to_base64_converts_numpy_array():
    """screenshot_to_base64 converts a numpy array to a valid base64 string."""
    client = AISidecarClient(MockConfig())
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    result = client.screenshot_to_base64(image)
    assert result is not None
    # Should be decodeable base64
    decoded = base64.b64decode(result)
    # PNG magic bytes: \x89PNG
    assert decoded[:4] == b'\x89PNG'


def test_screenshot_to_base64_returns_none_for_none():
    """screenshot_to_base64 returns None when given None."""
    client = AISidecarClient(MockConfig())
    result = client.screenshot_to_base64(None)
    assert result is None


# ---------------------------------------------------------------------------
# notify — success path
# ---------------------------------------------------------------------------

def test_notify_success_returns_response_dict():
    """notify returns the JSON response dict on HTTP 200."""
    client = AISidecarClient(MockConfig())
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'ok': True}

    with patch('hook.requests.post', return_value=mock_response) as mock_post:
        result = client.notify('heartbeat', {'data': 'value'})

    assert result == {'ok': True}
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert call_kwargs[1]['json']['event_type'] == 'heartbeat'


# ---------------------------------------------------------------------------
# notify — failure paths
# ---------------------------------------------------------------------------

def test_notify_when_sidecar_down_returns_none():
    """notify returns None when requests raises a connection error."""
    import requests as req_module
    client = AISidecarClient(MockConfig())

    with patch('hook.requests.post', side_effect=req_module.exceptions.ConnectionError('refused')):
        result = client.notify('heartbeat', {})

    assert result is None


def test_notify_non_200_returns_none():
    """notify returns None for non-200 HTTP responses."""
    client = AISidecarClient(MockConfig())
    mock_response = MagicMock()
    mock_response.status_code = 500

    with patch('hook.requests.post', return_value=mock_response):
        result = client.notify('heartbeat', {})

    assert result is None


# ---------------------------------------------------------------------------
# Debouncing
# ---------------------------------------------------------------------------

def test_debounce_second_cycle_start_within_cooldown_is_skipped():
    """A second cycle_start within the cooldown window returns None without posting."""
    client = AISidecarClient(MockConfig())
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'ok': True}

    with patch('hook.requests.post', return_value=mock_response) as mock_post:
        # First call — should go through
        result1 = client.notify('cycle_start', {})
        # Second call immediately — should be debounced
        result2 = client.notify('cycle_start', {})

    assert result1 == {'ok': True}
    assert result2 is None
    assert mock_post.call_count == 1


def test_no_debounce_on_other_event_types():
    """Other event types are never debounced."""
    client = AISidecarClient(MockConfig())
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'ok': True}

    with patch('hook.requests.post', return_value=mock_response) as mock_post:
        result1 = client.notify('heartbeat', {})
        result2 = client.notify('heartbeat', {})

    assert result1 == {'ok': True}
    assert result2 == {'ok': True}
    assert mock_post.call_count == 2


# ---------------------------------------------------------------------------
# Disabled client
# ---------------------------------------------------------------------------

def test_disabled_client_returns_none():
    """When enabled=False, notify returns None without making any HTTP call."""
    client = AISidecarClient(MockConfig(enabled=False))

    with patch('hook.requests.post') as mock_post:
        result = client.notify('heartbeat', {'data': 'value'})

    assert result is None
    mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# Sidecar address
# ---------------------------------------------------------------------------

def test_url_comes_from_config_when_environment_is_unset(monkeypatch):
    """Outside Docker AI_SIDECAR_URL is unset and AiSidecar_SidecarUrl applies."""
    monkeypatch.delenv('AI_SIDECAR_URL', raising=False)
    client = AISidecarClient(MockConfig(url='http://127.0.0.1:8484'))
    assert client.url == 'http://127.0.0.1:8484'


@pytest.mark.parametrize('value', ['', '   '])
def test_empty_environment_url_is_ignored(monkeypatch, value):
    monkeypatch.setenv('AI_SIDECAR_URL', value)
    client = AISidecarClient(MockConfig(url='http://127.0.0.1:8484'))
    assert client.url == 'http://127.0.0.1:8484'


def test_environment_url_wins_over_config(monkeypatch):
    """docker-compose.yml sets AI_SIDECAR_URL, where 127.0.0.1 is the alas container itself."""
    monkeypatch.setenv('AI_SIDECAR_URL', 'http://sidecar:8484/')
    client = AISidecarClient(MockConfig(url='http://127.0.0.1:8484'))
    assert client.url == 'http://sidecar:8484'

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'ok': True}
    with patch('hook.requests.post', return_value=mock_response) as mock_post:
        client.notify('heartbeat', {})

    assert mock_post.call_args[0][0] == 'http://sidecar:8484/api/events'
