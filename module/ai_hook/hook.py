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
import os
import time

import cv2
import requests

# ALAS's own logger (module/logger.py names it 'alas'), so that hook failures reach the
# ALAS log file. logging.getLogger(__name__) has no handler there and its warnings were lost.
# Taken by name, not imported from module.logger, so that this file still imports on its own:
# the harness's hook-check and dispatch-check load it without ALAS's logger module.
logger = logging.getLogger('alas')

from typing import Any, Dict, Optional


class AISidecarClient:
    def __init__(self, config):
        self.enabled = getattr(config, 'AiSidecar_Enabled', False)
        self.url = getattr(config, 'AiSidecar_SidecarUrl', 'http://sidecar:8484')
        # AI_SIDECAR_URL wins when it is set. docker-compose.yml sets it, because the
        # default 127.0.0.1 is the alas container itself there, not the sidecar.
        env_url = os.environ.get('AI_SIDECAR_URL', '').strip().rstrip('/')
        if env_url:
            self.url = env_url
        self._last_cycle_start = 0.0
        self._cooldown = 60  # seconds

    def screenshot_to_base64(self, image):
        # type: (Any) -> Optional[str]
        if image is None:
            return None
        try:
            # ALAS keeps frames in RGB order and OpenCV encodes BGR. Without the conversion
            # the sidecar and the model receive the frame with red and blue exchanged.
            if getattr(image, 'ndim', 0) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.png', image)
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            logger.warning(f'Failed to encode screenshot: {e}')
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
                try:
                    return resp.json()
                except ValueError as e:
                    # The sidecar answered; its body is not a directive.
                    logger.warning(
                        f'Sidecar returned a non-JSON body for event {event_type} (status 200): {e}')
                    return None
            else:
                logger.warning(
                    f'Sidecar returned status {resp.status_code} for event {event_type}')
                return None
        except Exception as e:
            logger.warning(f'Sidecar unreachable for event {event_type}: {e}')
            return None
