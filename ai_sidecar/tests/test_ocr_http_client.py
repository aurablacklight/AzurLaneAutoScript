"""
The OCR HTTP client (module/ocr/rpc.py) run against a stub sidecar.

The stub is an http.server on an ephemeral loopback port, started inside each test. No
real sidecar, device or game is involved. The client's sleeps are recorded instead of
slept, so the retry tests assert on attempts and requested sleeps, not on wall-clock time.

Run from the repository root: python -m pytest ai_sidecar/tests/test_ocr_http_client.py
"""
import base64
import json
import logging
import socket
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest
import requests

import module.ocr.rpc as rpc

# The retry schedules documented next to ModelProxy.outage_delays. They are written out
# here, not read from the client, so that a change to the worst-case wait fails a test.
OUTAGE_SLEEPS = [0.5, 1, 2, 4, 8, 8, 8]
SHORT_SLEEPS = [0.2, 0.4]

# What the sidecar really answers (crates/gacha-server/src/ocr_common.rs).
UNAVAILABLE = (503, {'error': 'ocr_unavailable', 'message': 'OCR engine is not available'})
RECOGNITION_FAILED = (422, {'error': 'recognition_failed',
                            'message': 'OCR processing failed; see server logs for detail'})
BATCH_TOO_LARGE = (400, {'error': 'invalid_request',
                         'message': 'images batch size 16 exceeds maximum of 8'})
DROP = 'drop'  # close the connection without answering
HANG = 'hang'  # accept the request and never answer


class StubSidecar:
    """Stands in for the sidecar's POST /ocr/recognize and GET /health.

    Each POST takes the next entry of `replies`: a (status, body) pair, DROP or HANG.
    When `replies` is empty the stub answers 200 with one result per image, whose text
    names the request and the image's position in it ('r0i0', 'r0i1', ...).
    """

    def __init__(self):
        self.posts = []  # JSON bodies of the POSTs received, in order
        self.replies = []
        self.health = (200, {'status': 'ok', 'ocr_available': True, 'detector_available': True})
        self.released = threading.Event()
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def answer(self, status, body):
                if not isinstance(body, bytes):
                    body = json.dumps(body).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self.answer(*stub.health)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                index = len(stub.posts)
                stub.posts.append(body)
                reply = stub.replies.pop(0) if stub.replies else None
                if reply == DROP:
                    self.close_connection = True
                    return
                if reply == HANG:
                    stub.released.wait(10)
                    self.close_connection = True
                    return
                if reply is None:
                    count = len(body['images']) if 'images' in body else 1
                    results = [{'text': 'r%di%d' % (index, i), 'confidence': 0.9, 'model': body['model']}
                               for i in range(count)]
                    reply = (200, {'results': results, 'duration_ms': 1})
                self.answer(*reply)

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.address = '127.0.0.1:%d' % self.server.server_address[1]
        # The short poll interval is only so that shutdown() returns promptly.
        threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': 0.01},
                         daemon=True).start()

    def close(self):
        self.released.set()
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stub():
    sidecar = StubSidecar()
    yield sidecar
    sidecar.close()


@pytest.fixture
def sleeps(monkeypatch):
    """The sleeps the client asks for, recorded and not slept.

    Only the client's own `time` name is replaced, so nothing else stops sleeping.
    """
    requested = []
    monkeypatch.setattr(rpc, 'time', types.SimpleNamespace(sleep=requested.append))
    return requested


@pytest.fixture
def proxy(stub, sleeps, monkeypatch):
    monkeypatch.setattr(rpc.ModelProxy, 'base_url', 'http://%s/ocr/recognize' % stub.address)
    return rpc.ModelProxy('azur_lane')


@pytest.fixture
def log():
    """Records the client logs, straight from ALAS's logger."""
    records = []

    class Collect(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Collect(level=logging.DEBUG)
    rpc.logger.addHandler(handler)
    yield records
    rpc.logger.removeHandler(handler)


def image(value):
    """A small grayscale image filled with `value`, so each image is recognisable."""
    return np.full((4, 6), value, dtype=np.uint8)


def sent_value(image_b64):
    """Decode an image as the sidecar would and return the value it was filled with."""
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(image_b64), np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded.shape == (4, 6)
    assert (decoded == decoded[0, 0]).all()
    return int(decoded[0, 0])


def dead_address():
    """A loopback address nothing listens on."""
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return '127.0.0.1:%d' % sock.getsockname()[1]


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------

def test_single_image_request_body(stub, proxy):
    assert proxy.atomic_ocr_for_single_line(image(7), '0123456789') == 'r0i0'

    assert len(stub.posts) == 1
    body = stub.posts[0]
    assert sorted(body) == ['cand_alphabet', 'image_b64', 'model']
    assert body['model'] == 'azur_lane'
    assert body['cand_alphabet'] == '0123456789'
    assert sent_value(body['image_b64']) == 7


def test_single_image_methods_return_cnocr_shapes(stub, proxy):
    assert proxy.ocr(image(1)) == ['r0i0']
    assert proxy.ocr_for_single_line(image(2)) == 'r1i0'
    assert proxy.atomic_ocr(image(3), 'abc') == 'r2i0'

    assert [sorted(body) for body in stub.posts] == [
        ['image_b64', 'model'], ['image_b64', 'model'], ['cand_alphabet', 'image_b64', 'model']]


def test_batch_request_body(stub, proxy):
    texts = proxy.atomic_ocr_for_single_lines([image(1), image(2), image(3)], '0123456789')

    assert texts == ['r0i0', 'r0i1', 'r0i2']
    assert len(stub.posts) == 1
    body = stub.posts[0]
    assert sorted(body) == ['cand_alphabet', 'images', 'model']
    assert body['model'] == 'azur_lane'
    assert body['cand_alphabet'] == '0123456789'
    assert [sent_value(b64) for b64 in body['images']] == [1, 2, 3]


def test_model_name_is_the_proxy_language(stub, sleeps, monkeypatch):
    monkeypatch.setattr(rpc.ModelProxy, 'base_url', 'http://%s/ocr/recognize' % stub.address)

    rpc.ModelProxy('jp').ocr_for_single_lines([image(1)])

    assert stub.posts[0]['model'] == 'jp'


def test_seventeen_images_are_split_into_sixteen_and_one(stub, proxy):
    texts = proxy.ocr_for_single_lines([image(i) for i in range(17)])

    assert [len(body['images']) for body in stub.posts] == [16, 1]
    assert [sent_value(b64) for body in stub.posts for b64 in body['images']] == list(range(17))
    # One text per image, in input order, across both requests.
    assert texts == ['r0i%d' % i for i in range(16)] + ['r1i0']


def test_empty_list_makes_no_request(stub, proxy):
    assert proxy.ocr_for_single_lines([]) == []
    assert proxy.atomic_ocr_for_single_lines([], '0123456789') == []

    assert stub.posts == []


def test_cand_alphabet_is_omitted_when_none(stub, proxy):
    proxy.atomic_ocr_for_single_lines([image(1)], None)
    proxy.atomic_ocr_for_single_line(image(1), None)
    proxy.ocr_for_single_lines([image(1)])

    assert [sorted(body) for body in stub.posts] == [
        ['images', 'model'], ['image_b64', 'model'], ['images', 'model']]


def test_empty_cand_alphabet_is_still_sent(stub, proxy):
    proxy.atomic_ocr_for_single_lines([image(1)], '')

    assert stub.posts[0]['cand_alphabet'] == ''


# ---------------------------------------------------------------------------
# Sidecar down or starting: long retry budget
# ---------------------------------------------------------------------------

def test_503_that_recovers_returns_the_result(stub, proxy, sleeps):
    # Five 503s is more than the old three attempts could survive.
    stub.replies = [UNAVAILABLE] * 5

    texts = proxy.atomic_ocr_for_single_lines([image(1), image(2)], '0123456789')

    assert texts == ['r5i0', 'r5i1']
    assert len(stub.posts) == 6
    assert sleeps == OUTAGE_SLEEPS[:5]
    # Every attempt carried the same request.
    assert all(body == stub.posts[0] for body in stub.posts)


def test_503_that_never_recovers_raises_within_the_bound(stub, proxy, sleeps):
    stub.replies = [UNAVAILABLE] * 20

    with pytest.raises(RuntimeError) as error:
        proxy.ocr_for_single_lines([image(1)])

    assert 'HTTP 503' in str(error.value)
    assert 'ocr_unavailable' in str(error.value)
    assert len(stub.posts) == 8
    assert sleeps == OUTAGE_SLEEPS
    assert 20 <= sum(sleeps) <= 60


def test_dropped_connection_that_recovers_returns_the_result(stub, proxy, sleeps):
    stub.replies = [DROP] * 4

    assert proxy.ocr_for_single_line(image(1)) == 'r4i0'

    assert len(stub.posts) == 5
    assert sleeps == OUTAGE_SLEEPS[:4]


def test_connection_refused_that_never_recovers_raises_within_the_bound(sleeps, monkeypatch):
    """Nothing listens on the port, as when the sidecar is stopped.

    Windows takes 2 s to report a refused loopback connection, which would make eight
    attempts take 16 s. The connect timeout is shortened so each attempt ends at once:
    there it ends as a connect timeout, elsewhere as the refusal. Both are a
    requests ConnectionError and take the same path in the client.
    """
    address = dead_address()
    attempts = []
    real_post = requests.post

    def counting_post(url, **kwargs):
        attempts.append(url)
        return real_post(url, **kwargs)

    monkeypatch.setattr(rpc.ModelProxy, 'base_url', 'http://%s/ocr/recognize' % address)
    monkeypatch.setattr(rpc.ModelProxy, 'connect_timeout', 0.05)
    monkeypatch.setattr(rpc.requests, 'post', counting_post)

    with pytest.raises(requests.exceptions.ConnectionError):
        rpc.ModelProxy('azur_lane').ocr_for_single_lines([image(1)])

    assert attempts == ['http://%s/ocr/recognize' % address] * 8
    assert sleeps == OUTAGE_SLEEPS
    # The documented bound: eight connection attempts and 31.5 s of sleeping.
    assert sum(sleeps) == 31.5


# ---------------------------------------------------------------------------
# Sidecar reachable but failing: short retry budget
# ---------------------------------------------------------------------------

def test_read_timeout_keeps_the_short_budget(stub, proxy, sleeps, monkeypatch):
    # The server accepted the request and is slow. Each attempt already waits the
    # full timeout, so this must not get the long outage budget.
    monkeypatch.setattr(rpc.ModelProxy, 'timeout', 0.2)
    stub.replies = [HANG] * 20

    with pytest.raises(requests.exceptions.ReadTimeout):
        proxy.ocr_for_single_lines([image(1)])

    assert len(stub.posts) == 3
    assert sleeps == SHORT_SLEEPS


def test_other_5xx_keeps_the_short_budget(stub, proxy, sleeps):
    stub.replies = [(500, {'error': 'internal', 'message': 'boom'})] * 20

    with pytest.raises(RuntimeError) as error:
        proxy.ocr_for_single_lines([image(1)])

    assert 'HTTP 500' in str(error.value)
    assert len(stub.posts) == 3
    assert sleeps == SHORT_SLEEPS


def test_500_that_recovers_returns_the_result(stub, proxy, sleeps):
    stub.replies = [(500, {'error': 'internal', 'message': 'boom'})]

    assert proxy.ocr_for_single_lines([image(1)]) == ['r1i0']
    assert sleeps == SHORT_SLEEPS[:1]


# ---------------------------------------------------------------------------
# Answers that must raise at once
# ---------------------------------------------------------------------------

def test_400_raises_at_once(stub, proxy, sleeps):
    stub.replies = [BATCH_TOO_LARGE]

    with pytest.raises(RuntimeError) as error:
        proxy.ocr_for_single_lines([image(1)])

    assert len(stub.posts) == 1
    assert sleeps == []
    assert 'rejected request (HTTP 400)' in str(error.value)
    assert 'images batch size 16 exceeds maximum of 8' in str(error.value)


def test_422_raises_at_once_and_is_described_as_a_server_failure(stub, proxy, sleeps):
    stub.replies = [RECOGNITION_FAILED]

    with pytest.raises(RuntimeError) as error:
        proxy.ocr_for_single_lines([image(1)])

    assert len(stub.posts) == 1
    assert sleeps == []
    message = str(error.value)
    assert 'HTTP 422' in message
    assert 'server-side failure' in message
    # It is not blamed on the request, and the server's own words are kept.
    assert 'rejected request' not in message
    assert 'recognition_failed' in message
    assert 'OCR processing failed; see server logs for detail' in message


def test_non_json_200_raises(stub, proxy, sleeps):
    stub.replies = [(200, b'<html>not json</html>')]

    with pytest.raises(ValueError):
        proxy.ocr_for_single_lines([image(1)])

    assert len(stub.posts) == 1
    assert sleeps == []


# ---------------------------------------------------------------------------
# One result per image
# ---------------------------------------------------------------------------

def result_list(count):
    return (200, {'results': [{'text': '111', 'confidence': 0.9, 'model': 'azur_lane'}] * count,
                  'duration_ms': 1})


def test_fewer_results_than_images_raises(stub, proxy, sleeps):
    stub.replies = [result_list(2)]

    with pytest.raises(RuntimeError) as error:
        proxy.atomic_ocr_for_single_lines([image(1), image(2), image(3)], '0123456789')

    assert 'returned 2 results for 3 images' in str(error.value)
    assert len(stub.posts) == 1
    assert sleeps == []


def test_more_results_than_images_raises(stub, proxy):
    stub.replies = [result_list(4)]

    with pytest.raises(RuntimeError) as error:
        proxy.ocr_for_single_lines([image(1), image(2), image(3)])

    assert 'returned 4 results for 3 images' in str(error.value)


def test_result_count_is_checked_for_every_chunk(stub, proxy):
    # 17 images: the first chunk of 16 is answered correctly, the second chunk of 1 is not.
    stub.replies = [result_list(16), result_list(2)]

    with pytest.raises(RuntimeError) as error:
        proxy.ocr_for_single_lines([image(i) for i in range(17)])

    assert 'returned 2 results for 1 images' in str(error.value)


def test_two_results_for_a_single_image_raises(stub, proxy):
    stub.replies = [result_list(2)]

    with pytest.raises(RuntimeError) as error:
        proxy.ocr_for_single_line(image(1))

    assert 'returned 2 results for 1 images' in str(error.value)


# ---------------------------------------------------------------------------
# Low-confidence readings are logged and returned unchanged
# ---------------------------------------------------------------------------

def reading(text, confidence=None):
    result = {'text': text, 'model': 'azur_lane'}
    if confidence is not None:
        result['confidence'] = confidence
    return (200, {'results': [result], 'duration_ms': 1})


def low_confidence_messages(log):
    return [record.getMessage() for record in log
            if record.levelname == 'WARNING' and 'low confidence' in record.getMessage()]


def test_low_confidence_whitelisted_reading_is_logged_and_kept(stub, proxy, log):
    stub.replies = [reading('24', 0.27), reading('24', 0.9)]

    unsure = proxy.atomic_ocr_for_single_lines([image(1)], '0123456789')
    sure = proxy.atomic_ocr_for_single_lines([image(1)], '0123456789')

    assert unsure == sure
    assert low_confidence_messages(log) == ["OCR low confidence 0.27 for '24'"]


def test_low_confidence_is_logged_for_a_single_image_too(stub, proxy, log):
    stub.replies = [reading('2283', 0.4)]

    proxy.atomic_ocr_for_single_line(image(1), '0123456789')

    assert low_confidence_messages(log) == ["OCR low confidence 0.40 for '2283'"]


@pytest.mark.parametrize('reply, alphabet', [
    (reading('22843', 0.5), '0123456789'),      # at the line, not under it
    (reading('22843', 0.94), '0123456789'),
    (reading('', 0.0), '0123456789'),           # nothing was read; there is nothing to doubt
    (reading('22843'), '0123456789'),           # a server that sends no confidence
    (reading('22843', True), '0123456789'),     # not a number
])
def test_other_whitelisted_readings_are_not_logged(stub, proxy, log, reply, alphabet):
    stub.replies = [reply]

    proxy.atomic_ocr_for_single_lines([image(1)], alphabet)

    assert low_confidence_messages(log) == []


def test_free_text_is_not_logged(stub, proxy, log):
    # No whitelist: on free text the confidence does not separate right from wrong.
    stub.replies = [reading('Urgent Mission', 0.2)]

    proxy.ocr_for_single_lines([image(1)])

    assert low_confidence_messages(log) == []


# ---------------------------------------------------------------------------
# init and the health check
# ---------------------------------------------------------------------------

@pytest.fixture
def restore_base_url(monkeypatch):
    # init() assigns base_url on the class; put the previous value back afterwards.
    monkeypatch.setattr(rpc.ModelProxy, 'base_url', None)


def messages(log, level):
    return [record.getMessage() for record in log if record.levelno == level]


def test_init_logs_connected_when_ocr_is_available(stub, log, restore_base_url):
    rpc.ModelProxy.init(stub.address)

    assert rpc.ModelProxy.base_url == 'http://%s/ocr/recognize' % stub.address
    assert any('Successfully connected' in text for text in messages(log, logging.INFO))
    assert messages(log, logging.WARNING) == []


def test_init_warns_when_health_reports_ocr_unavailable(stub, log, restore_base_url):
    stub.health = (200, {'status': 'ok', 'ocr_available': False, 'detector_available': False})

    rpc.ModelProxy.init(stub.address)

    warnings = messages(log, logging.WARNING)
    assert len(warnings) == 1
    assert 'OCR is not available' in warnings[0]
    assert not any('Successfully connected' in text for text in messages(log, logging.INFO))
    # Still not fatal and not latched: requests go to this address.
    assert rpc.ModelProxy.base_url == 'http://%s/ocr/recognize' % stub.address


@pytest.mark.parametrize('health', [
    (200, {'status': 'ok'}),
    (200, b'ok'),
    (200, ['ok']),
], ids=['no ocr_available field', 'not json', 'not an object'])
def test_init_logs_connected_when_health_does_not_say(stub, log, restore_base_url, health):
    stub.health = health

    rpc.ModelProxy.init(stub.address)

    assert any('Successfully connected' in text for text in messages(log, logging.INFO))
    assert messages(log, logging.WARNING) == []


def test_init_warns_and_continues_when_health_fails(stub, log, restore_base_url):
    stub.health = (500, {'error': 'internal'})

    rpc.ModelProxy.init(stub.address)

    warnings = messages(log, logging.WARNING)
    assert len(warnings) == 1
    assert 'health check failed' in warnings[0]
    assert rpc.ModelProxy.base_url == 'http://%s/ocr/recognize' % stub.address
