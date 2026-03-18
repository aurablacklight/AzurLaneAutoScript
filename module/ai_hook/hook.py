"""
AI Sidecar hook layer for ALAS.

IMPORTANT: This file runs inside the ALAS container (Python 3.7).
Do NOT use 3.8+ features:
- No walrus operator (:=)
- No typing.Literal
- Use typing.Optional, not X | None
- No positional-only parameters (/)
"""
import base64
import logging
import time

import cv2
import requests

logger = logging.getLogger(__name__)

from typing import Any, Dict, Optional


class AISidecarClient:
    def __init__(self, config):
        self.enabled = getattr(config, 'AiSidecar_Enabled', False)
        self.url = getattr(config, 'AiSidecar_SidecarUrl', 'http://sidecar:8484')
        self._last_cycle_start = 0.0
        self._cooldown = 60  # seconds

    def screenshot_to_base64(self, image):
        # type: (Any) -> Optional[str]
        if image is None:
            return None
        try:
            _, buffer = cv2.imencode('.png', image)
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            logger.warning('Failed to encode screenshot: %s', e)
            return None

    def notify(self, event_type, payload):
        # type: (str, Dict[str, Any]) -> Optional[Dict[str, Any]]
        if not self.enabled:
            return None

        # Debounce cycle_start
        if event_type == 'cycle_start':
            now = time.time()
            if now - self._last_cycle_start < self._cooldown:
                return None
            self._last_cycle_start = now

        try:
            payload['event_type'] = event_type
            resp = requests.post(
                self.url + '/api/events',
                json=payload,
                timeout=(5, 30),
            )
            if resp.status_code == 200:
                return resp.json()
            else:
                logger.warning(
                    'Sidecar returned status %d for event %s',
                    resp.status_code,
                    event_type,
                )
                return None
        except Exception as e:
            logger.warning('Sidecar unreachable for event %s: %s', event_type, e)
            return None
