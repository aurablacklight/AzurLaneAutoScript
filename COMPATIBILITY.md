# Multi-Architecture Docker Compatibility

## Containers

`docker-compose.yml` starts two containers:

| Container  | Service name | Port  | Purpose |
|------------|--------------|-------|---------|
| ALAS       | `alas`       | 22267 | Main application and web UI |
| AI Sidecar | `sidecar`    | 8484  | Python sidecar built from `./ai_sidecar` (`GET /health`, `POST /api/events`) |

A third service, `ocr` (cnocr/mxnet over zerorpc on port 22268), is commented out and kept
for reference only. See [Legacy cnocr server](#legacy-cnocr-server).

## OCR

With the default settings ALAS does not run OCR itself. `module/ocr/rpc.py` sends every OCR
call over HTTP to the Rust sidecar `gacha-server` from
[rust-gacha-ocr-sidecar](https://github.com/aurablacklight/rust-gacha-ocr-sidecar), which
listens on port 8484 by default:

```
ALAS  --HTTP/JSON-->  gacha-server
                       GET  /health          checked once when the client connects
                       POST /ocr/recognize   one image, or a batch of up to 16
```

`gacha-server` also answers `GET /health` and `POST /api/events`, the two endpoints of the
Python sidecar, so one process serves both the AI hook and OCR.

The connection is configured in `config/deploy.yaml`:

- `UseOcrServer: true` -- send OCR to the HTTP server. With `false`, every ALAS instance
  loads cnocr and mxnet in-process, which only works where both are installed (x86 only).
- `OcrClientAddress` -- `host:port` of `gacha-server`, without `http://`.
  - ALAS and `gacha-server` on the same host: `127.0.0.1:8484`
  - ALAS in Docker Compose: `host.docker.internal:8484`. The `alas` service maps that name
    to the host with `extra_hosts`.
  - ALAS in a container started with `--net=host` (`deploy/docker/Docker-run.sh`):
    `127.0.0.1:8484`
- `StartOcrServer: false` -- this only controls the legacy zerorpc server and is ignored
  while `UseOcrServer` is `true`.

Every shipped template sets `UseOcrServer: true`. `config/deploy.template-docker*.yaml` set
`OcrClientAddress: host.docker.internal:8484`, the others `127.0.0.1:8484`. An existing
`config/deploy.yaml` keeps the values it already has; change them by hand to move an install
from the legacy path to the sidecar.

There is no local fallback. If the server is unreachable or answers with a 5xx, the request
is tried 3 times and then raises; a 4xx raises immediately.

### Known gap in Docker Compose

The `sidecar` service is still the Python sidecar. It has no `/ocr/recognize`, so with the
compose file as shipped every OCR call fails with HTTP 404. It also publishes host port
8484, the port `gacha-server` uses.

Until the compose file runs `gacha-server` (a container image is tracked in
[rust-gacha-ocr-sidecar#50](https://github.com/aurablacklight/rust-gacha-ocr-sidecar/issues/50)),
run `gacha-server` on the host in place of the `sidecar` service, so that it is what listens
on host port 8484. Two things to check:

- `gacha-server` binds `127.0.0.1` by default (`server.host` in its `config.toml`). On Linux
  a container reaches the host through the Docker bridge, not through loopback, so the
  server has to listen on an address the bridge can reach. The API has no authentication;
  do not expose it beyond the Docker network.
- The `alas` service waits for the `sidecar` service to be healthy. Removing the service
  means removing that `depends_on` entry too.

## Architecture Behavior by Host

`deploy/docker/Dockerfile` has no platform pin and installs no mxnet, so the `alas`
container runs natively on amd64 and arm64. `gacha-server` builds natively for both.
Nothing in the default setup needs x86 emulation.

## Legacy cnocr server

Before the Rust sidecar, OCR ran in a separate x86 container because `mxnet 1.6.0` has no
arm64 build, and ALAS talked to it over zerorpc on port 22268 (`OcrClientAddress: ocr:22268`).
The server side is still in the tree:

- `deploy/docker/Dockerfile.ocr`, `deploy/docker/requirements-ocr.txt` -- the x86-pinned
  image, started with `python -m module.ocr.rpc`
- `start_ocr_server()` in `module/ocr/rpc.py` -- the zerorpc server
- `StartOcrServer` / `OcrServerPort` -- start that server from the web UI

The client side is gone: `ModelProxy` in `module/ocr/rpc.py` speaks HTTP only, so this fork
cannot use the zerorpc server, and uncommenting the `ocr` service does not bring the old
setup back. The web UI therefore starts the zerorpc server only when `StartOcrServer` is
`true` and `UseOcrServer` is `false`.

## Files

- `deploy/docker/Dockerfile` -- main ALAS container (multi-arch, no platform pin)
- `deploy/docker/requirements.txt` -- Python deps for ALAS (no mxnet)
- `docker-compose.yml` -- the `alas` and `sidecar` services
- `deploy/template` -- source of every `config/deploy.template*.yaml`; regenerate them with
  `ConfigGenerator.generate_deploy_template()` in `module/config/config_updater.py`
- `config/deploy.template-docker.yaml` -- deploy config for Docker Compose
- `module/ocr/rpc.py` -- HTTP OCR client, and the legacy zerorpc server
- `ai_sidecar/` -- the Python sidecar
- `deploy/docker/Dockerfile.ocr`, `deploy/docker/requirements-ocr.txt` -- legacy OCR server
