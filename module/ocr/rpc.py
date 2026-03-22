import argparse
import base64
import multiprocessing
import pickle

import requests

from module.logger import logger
from module.webui.setting import State

process: multiprocessing.Process = None


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
    online = True
    timeout = 10  # seconds, slightly higher than old ZeroRPC 5s to account for model cold-load

    @classmethod
    def init(cls, address="127.0.0.1:8484"):
        logger.info(f"Connecting to OCR server at http://{address}")
        cls.base_url = f"http://{address}/ocr/recognize"
        try:
            resp = requests.get(f"http://{address}/health", timeout=3)
            resp.raise_for_status()
            logger.info("Successfully connected to OCR server (HTTP)")
        except Exception:
            cls.online = False
            logger.warning("OCR server not running (health check failed)")

    @classmethod
    def close(cls):
        if cls.base_url is not None:
            logger.info('Disconnecting from OCR server')
            cls.base_url = None
            logger.info('Successfully disconnected from OCR server')

    def __init__(self, lang) -> None:
        self.lang = lang

    def _post(self, body):
        """Send a POST request to the OCR endpoint and return parsed JSON.

        Args:
            body (dict): JSON request body.

        Returns:
            dict: Parsed JSON response.

        Raises:
            Exception: On any HTTP or parsing error (caught by callers to trigger fallback).
        """
        resp = requests.post(ModelProxy.base_url, json=body, timeout=ModelProxy.timeout)
        resp.raise_for_status()
        return resp.json()

    def ocr(self, img_fp):
        """
        Args:
            img_fp (np.ndarray):

        Returns:
            list: List of character sequences (matching cnocr format).
        """
        if self.online:
            try:
                body = {"image_b64": _encode_image(img_fp), "model": self.lang}
                data = self._post(body)
                # Return format: list of char-lists (caller joins each with ''.join)
                return [r["text"] for r in data["results"]]
            except Exception:
                self.online = False
        from module.ocr.models import OCR_MODEL
        return OCR_MODEL.__getattribute__(self.lang).ocr(img_fp)

    def ocr_for_single_line(self, img_fp):
        """
        Args:
            img_fp (np.ndarray):

        Returns:
            str: Recognized text (iterable of chars for caller to join).
        """
        if self.online:
            try:
                body = {"image_b64": _encode_image(img_fp), "model": self.lang}
                data = self._post(body)
                return data["results"][0]["text"]
            except Exception:
                self.online = False
        from module.ocr.models import OCR_MODEL
        return OCR_MODEL.__getattribute__(self.lang).ocr_for_single_line(img_fp)

    def ocr_for_single_lines(self, img_list):
        """
        Args:
            img_list (list[np.ndarray]):

        Returns:
            list: List of recognized texts (each iterable of chars).
        """
        if self.online:
            try:
                body = {
                    "images": [_encode_image(img) for img in img_list],
                    "model": self.lang,
                }
                data = self._post(body)
                return [r["text"] for r in data["results"]]
            except Exception:
                self.online = False
        from module.ocr.models import OCR_MODEL
        return OCR_MODEL.__getattribute__(self.lang).ocr_for_single_lines(img_list)

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
        if self.online:
            try:
                body = {"image_b64": _encode_image(img_fp), "model": self.lang}
                if cand_alphabet is not None:
                    body["cand_alphabet"] = cand_alphabet
                data = self._post(body)
                return data["results"][0]["text"]
            except Exception:
                self.online = False
        from module.ocr.models import OCR_MODEL
        return OCR_MODEL.__getattribute__(self.lang).atomic_ocr(img_fp, cand_alphabet)

    def atomic_ocr_for_single_line(self, img_fp, cand_alphabet=None):
        """
        Args:
            img_fp (np.ndarray):
            cand_alphabet:

        Returns:
            str: Recognized text (iterable of chars for caller to join).
        """
        if self.online:
            try:
                body = {"image_b64": _encode_image(img_fp), "model": self.lang}
                if cand_alphabet is not None:
                    body["cand_alphabet"] = cand_alphabet
                data = self._post(body)
                return data["results"][0]["text"]
            except Exception:
                self.online = False
        from module.ocr.models import OCR_MODEL
        return OCR_MODEL.__getattribute__(self.lang).atomic_ocr_for_single_line(img_fp, cand_alphabet)

    def atomic_ocr_for_single_lines(self, img_list, cand_alphabet=None):
        """
        Args:
            img_list (list[np.ndarray]):
            cand_alphabet:

        Returns:
            list: List of recognized texts (each iterable of chars for caller to join).
        """
        if self.online:
            try:
                body = {
                    "images": [_encode_image(img) for img in img_list],
                    "model": self.lang,
                }
                if cand_alphabet is not None:
                    body["cand_alphabet"] = cand_alphabet
                data = self._post(body)
                return [r["text"] for r in data["results"]]
            except Exception:
                self.online = False
        from module.ocr.models import OCR_MODEL
        return OCR_MODEL.__getattribute__(self.lang).atomic_ocr_for_single_lines(img_list, cand_alphabet)

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


# Legacy: ZeroRPC server functions. Not used when UseOcrServer points to Rust sidecar.
# These are still needed if someone runs the old mxnet OCR server locally.

def start_ocr_server(port=22268):
    import zerorpc
    import zmq
    from module.ocr.al_ocr import AlOcr
    from module.ocr.models import OcrModel

    class OCRServer(OcrModel):
        def hello(self):
            return "hello"

        def ocr(self, lang, img_fp):
            img_fp = pickle.loads(img_fp)
            cnocr: AlOcr = self.__getattribute__(lang)
            return cnocr.ocr(img_fp)

        def ocr_for_single_line(self, lang, img_fp):
            img_fp = pickle.loads(img_fp)
            cnocr: AlOcr = self.__getattribute__(lang)
            return cnocr.ocr_for_single_line(img_fp)

        def ocr_for_single_lines(self, lang, img_list):
            img_list = [pickle.loads(img_fp) for img_fp in img_list]
            cnocr: AlOcr = self.__getattribute__(lang)
            return cnocr.ocr_for_single_lines(img_list)

        def set_cand_alphabet(self, lang, cand_alphabet):
            cnocr: AlOcr = self.__getattribute__(lang)
            return cnocr.set_cand_alphabet(cand_alphabet)

        def atomic_ocr(self, lang, img_fp, cand_alphabet):
            img_fp = pickle.loads(img_fp)
            cnocr: AlOcr = self.__getattribute__(lang)
            return cnocr.atomic_ocr(img_fp, cand_alphabet)

        def atomic_ocr_for_single_line(self, lang, img_fp, cand_alphabet):
            img_fp = pickle.loads(img_fp)
            cnocr: AlOcr = self.__getattribute__(lang)
            return cnocr.atomic_ocr_for_single_line(img_fp, cand_alphabet)

        def atomic_ocr_for_single_lines(self, lang, img_list, cand_alphabet):
            img_list = [pickle.loads(img_fp) for img_fp in img_list]
            cnocr: AlOcr = self.__getattribute__(lang)
            return cnocr.atomic_ocr_for_single_lines(img_list, cand_alphabet)

        def debug(self, lang, img_list):
            img_list = [pickle.loads(img_fp) for img_fp in img_list]
            cnocr: AlOcr = self.__getattribute__(lang)
            return cnocr.debug(img_list)

    server = zerorpc.Server(OCRServer())
    try:
        server.bind(f"tcp://*:{port}")
    except zmq.error.ZMQError:
        logger.error(f"Ocr server cannot bind on port {port}")
        return
    logger.info(f"Ocr server listen on port {port}")
    server.run()


def start_ocr_server_process(port=22268):
    global process
    if not alive():
        process = multiprocessing.Process(target=start_ocr_server, args=(port,))
        process.start()


def stop_ocr_server_process():
    global process
    if alive():
        process.kill()
        process = None


def alive() -> bool:
    global process
    if process is not None:
        return process.is_alive()
    else:
        return False


if __name__ == "__main__":
    # Run server
    parser = argparse.ArgumentParser(description="Alas OCR service")
    parser.add_argument(
        "--port",
        type=int,
        help="Port to listen. Default to OcrServerPort in deploy setting",
    )
    args, _ = parser.parse_known_args()
    port = args.port or State.deploy_config.OcrServerPort
    start_ocr_server(port=port)
