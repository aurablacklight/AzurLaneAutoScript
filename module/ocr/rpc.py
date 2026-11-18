import base64
import time

import requests

from module.logger import logger
from module.webui.setting import State


def _encode_image(img_fp):
    """Encode numpy array as base64 PNG for HTTP transport.

    Args:
        img_fp (np.ndarray): Image array (grayscale or color).

    Returns:
        str: Base64-encoded PNG string.
    """
    import cv2
    _, buf = cv2.imencode('.png', img_fp)
    return base64.b64encode(buf).decode('ascii')


class ModelProxy:
    base_url = None
    timeout = 10  # seconds, slightly higher than old ZeroRPC 5s to account for model cold-load
    connect_timeout = 3  # seconds to open the connection, same as the health check
    # Keep in step with MAX_OCR_BATCH in the sidecar (crates/gacha-server/src/ocr_common.rs).
    # Larger batches are split client-side into chunks of this size. If the sidecar's limit
    # grows, a stale value here only costs extra round trips. If the sidecar's limit is lower
    # than this, it answers 400 to every chunk above its limit and the request raises.
    max_batch = 16
    # Seconds to sleep before each retry. One list per kind of failure, each counted on its
    # own within a request; a request raises once its list is used up.
    #
    # Sidecar down or starting (connection refused, reset or not opened in time, HTTP 503):
    # 8 attempts, 31.5 s of sleeping, so that a sidecar restart is survived. Worst case
    # before raising:
    #   HTTP 503                        about 32 s (the answer itself is immediate)
    #   connection refused              about 32 s; about 48 s against loopback on Windows,
    #                                   which takes 2 s to report each refusal
    #   connection not opened in time   55.5 s (8 x connect_timeout + 31.5 s)
    outage_delays = (0.5, 1, 2, 4, 8, 8, 8)
    # Sidecar reachable but failing (read timeout, 5xx other than 503, anything else):
    # 3 attempts, as before. Worst case for a read timeout is 30.6 s (3 x timeout + 0.6 s);
    # it does not get the long list because every attempt already waits `timeout`.
    # A request that meets both kinds of failure can draw on both lists; that stays under
    # the two worst cases added together (86 s).
    retry_delays = (0.2, 0.4)

    @classmethod
    def init(cls, address="127.0.0.1:8484"):
        logger.info('Connecting to OCR server at http://%s' % address)
        cls.base_url = 'http://%s/ocr/recognize' % address
        try:
            resp = requests.get('http://%s/health' % address, timeout=3)
            resp.raise_for_status()
            try:
                ocr_available = resp.json().get('ocr_available')
            except Exception:
                # Not the sidecar's health body; nothing to judge by.
                ocr_available = None
            if ocr_available is False:
                logger.warning('OCR server is running but reports that OCR is not available '
                               '(health: ocr_available=false). OCR requests will fail with '
                               'HTTP 503 until it is; check the sidecar log and its models')
            else:
                logger.info('Successfully connected to OCR server (HTTP)')
        except Exception as e:
            # Not fatal and deliberately not latched: the server may simply be slow to
            # start. Requests retry on their own, and there is no local OCR fallback.
            logger.warning('OCR server health check failed (%s: %s), continuing anyway'
                           % (type(e).__name__, e))

    @classmethod
    def close(cls):
        if cls.base_url is not None:
            logger.info('Disconnecting from OCR server')
            cls.base_url = None
            logger.info('Successfully disconnected from OCR server')

    def __init__(self, lang) -> None:
        self.lang = lang

    def _post(self, body):
        """Send a POST to the OCR endpoint and return parsed JSON.

        Transport errors and 5xx are retried, see `outage_delays` and `retry_delays`
        for how long. 4xx is raised immediately because the same request would get
        the same answer.

        Args:
            body (dict): JSON request body.

        Returns:
            dict: Parsed JSON response.

        Raises:
            Exception: On a 4xx, or after the final retry.
        """
        outage_failures = 0
        other_failures = 0
        while 1:
            outage = False
            try:
                resp = requests.post(ModelProxy.base_url, json=body,
                                     timeout=(ModelProxy.connect_timeout, ModelProxy.timeout))
            except requests.exceptions.ConnectionError as e:
                # Refused, reset, or not opened in time: the sidecar is down or starting.
                # A read timeout is not a ConnectionError and takes the branch below.
                error = e
                outage = True
            except Exception as e:
                error = e
            else:
                if resp.status_code == 422:
                    # The sidecar's own failure, not a malformed request: it answers
                    # 422 `recognition_failed` when recognition errors out.
                    raise RuntimeError(
                        'OCR server failed to process the request (HTTP 422). This is a '
                        'server-side failure: it could not decode an image, does not know '
                        'the model name, or its model files are missing; the cause is in '
                        'the sidecar log. Server said: %s' % resp.text[:300])
                if 400 <= resp.status_code < 500:
                    # The request itself is wrong. Surface the server's explanation
                    # rather than a misleading downstream error.
                    raise RuntimeError('OCR server rejected request (HTTP %s): %s'
                                       % (resp.status_code, resp.text[:300]))
                if resp.status_code < 500:
                    return resp.json()
                error = RuntimeError('OCR server error (HTTP %s): %s'
                                     % (resp.status_code, resp.text[:300]))
                # 503 `ocr_unavailable`: the sidecar is up but its OCR engine is not.
                outage = resp.status_code == 503

            if outage:
                delays, failures = ModelProxy.outage_delays, outage_failures
                outage_failures += 1
            else:
                delays, failures = ModelProxy.retry_delays, other_failures
                other_failures += 1
            if failures >= len(delays):
                raise error
            logger.warning('OCR request failed (%s: %s), retry %s/%s in %ss'
                           % (type(error).__name__, error, failures + 1, len(delays),
                              delays[failures]))
            time.sleep(delays[failures])

    @staticmethod
    def _result_texts(data, expected):
        """Take the text of each result, refusing a response of the wrong length.

        Results are matched to images by position, so a missing or extra result
        would put every later reading on the wrong item.

        Args:
            data (dict): Parsed JSON response.
            expected (int): Number of images that were sent.

        Returns:
            list[str]:
        """
        results = data["results"]
        if len(results) != expected:
            raise RuntimeError('OCR server returned %s results for %s images'
                               % (len(results), expected))
        return [r["text"] for r in results]

    def _post_batch(self, img_list, cand_alphabet=None):
        """OCR a list of images, splitting oversized batches.

        An empty list returns an empty list without contacting the server, matching
        cnocr 1.2.2 (`cn_ocr.py`, `if len(img_list) == 0: return []`). ALAS reaches
        this legitimately whenever a grid filter selects nothing.

        Args:
            img_list (list[np.ndarray]):
            cand_alphabet (str): Optional character whitelist.

        Returns:
            list[str]: One result per input image, in input order.
        """
        if not len(img_list):
            return []

        results = []
        for start in range(0, len(img_list), ModelProxy.max_batch):
            chunk = img_list[start:start + ModelProxy.max_batch]
            body = {
                "images": [_encode_image(img) for img in chunk],
                "model": self.lang,
            }
            if cand_alphabet is not None:
                body["cand_alphabet"] = cand_alphabet
            data = self._post(body)
            results.extend(self._result_texts(data, len(chunk)))
        return results

    def _post_single(self, img_fp, cand_alphabet=None):
        """OCR one image and return its text.

        Args:
            img_fp (np.ndarray):
            cand_alphabet (str): Optional character whitelist.

        Returns:
            str: Recognized text.
        """
        body = {"image_b64": _encode_image(img_fp), "model": self.lang}
        if cand_alphabet is not None:
            body["cand_alphabet"] = cand_alphabet
        data = self._post(body)
        return self._result_texts(data, 1)[0]

    def ocr(self, img_fp):
        """
        Args:
            img_fp (np.ndarray):

        Returns:
            list: List of character sequences (matching cnocr format).
        """
        data = self._post_single(img_fp)
        return [data]

    def ocr_for_single_line(self, img_fp):
        """
        Args:
            img_fp (np.ndarray):

        Returns:
            str: Recognized text (iterable of chars for caller to join).
        """
        return self._post_single(img_fp)

    def ocr_for_single_lines(self, img_list):
        """
        Args:
            img_list (list[np.ndarray]):

        Returns:
            list: List of recognized texts (each iterable of chars).
        """
        return self._post_batch(img_list)

    def set_cand_alphabet(self, cand_alphabet: str):
        # No-op: alphabet is now passed per-request in the HTTP body.
        logger.debug("set_cand_alphabet is a no-op with HTTP OCR (alphabet passed per-request)")
        return None

    def atomic_ocr(self, img_fp, cand_alphabet=None):
        """
        Args:
            img_fp (np.ndarray):
            cand_alphabet:

        Returns:
            str: Recognized text (iterable of chars for caller to join).
        """
        return self._post_single(img_fp, cand_alphabet)

    def atomic_ocr_for_single_line(self, img_fp, cand_alphabet=None):
        """
        Args:
            img_fp (np.ndarray):
            cand_alphabet:

        Returns:
            str: Recognized text (iterable of chars for caller to join).
        """
        return self._post_single(img_fp, cand_alphabet)

    def atomic_ocr_for_single_lines(self, img_list, cand_alphabet=None):
        """
        Args:
            img_list (list[np.ndarray]):
            cand_alphabet:

        Returns:
            list: List of recognized texts (each iterable of chars for caller to join).
        """
        return self._post_batch(img_list, cand_alphabet)

    def debug(self, img_list):
        """
        Args:
            img_list (list[np.ndarray]):

        Returns:
            list: Empty list (debug visualization not supported with HTTP OCR server).
        """
        logger.debug("debug() is not supported with HTTP OCR server")
        return []


class ModelProxyFactory:
    def __getattribute__(self, __name: str) -> ModelProxy:
        if __name in ["azur_lane", "cnocr", "jp", "tw", "azur_lane_jp"]:
            if ModelProxy.base_url is None:
                ModelProxy.init(address=State.deploy_config.OcrClientAddress)
            return ModelProxy(lang=__name)
        else:
            return super().__getattribute__(__name)

    def close(self):
        ModelProxy.close()
