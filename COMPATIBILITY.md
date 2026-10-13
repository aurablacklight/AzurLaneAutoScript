# Multi-Architecture Docker Compatibility

## Containers

`docker-compose.yml` starts two containers:

| Container | Service name | Port  | Purpose |
|-----------|--------------|-------|---------|
| ALAS      | `alas`       | 22267 | Main application and web UI |
| Sidecar   | `sidecar`    | 8484  | `gacha-server`: OCR and the AI hook |

`gacha-server` is the Rust sidecar from
[rust-gacha-ocr-sidecar](https://github.com/aurablacklight/rust-gacha-ocr-sidecar). One
process answers everything ALAS sends:

```
ALAS  --HTTP/JSON-->  gacha-server
                       GET  /health          checked once when the OCR client connects
                       POST /ocr/recognize   one image, or a batch of up to 16
                       POST /api/events      the AI hook
```

`deploy/docker/Dockerfile` has no platform pin and installs no mxnet, and the sidecar image
is published for `linux/amd64` and `linux/arm64`. Both containers run natively on either
architecture; nothing needs x86 emulation.

## Setting up the sidecar service

The `sidecar` service runs the published image
`ghcr.io/aurablacklight/rust-gacha-ocr-sidecar:main`. It holds the server binary and nothing
else, so four things have to be in place before `docker compose up`:

1. **Registry login.** The image is a private package. Run `docker login ghcr.io` with a
   token that can read packages, or the pull is refused.
2. **AI provider key.** Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`.
   `OPENROUTER_API_KEY` is accepted too and wins if both are set. The server exits at
   startup without a key, and the `alas` service then never starts, because it waits for a
   healthy sidecar.
3. **OCR models.** `./models/ocr` is mounted read-only at `/app/models/ocr`. Set
   `GACHA_OCR_MODELS` in `.env` to use another directory. `scripts/download_models.sh` in
   the sidecar repository downloads the models into this layout:

   ```
   models/ocr/det/pp-ocrv5_mobile_det.onnx
   models/ocr/chinese/rec.onnx, dict.txt
   models/ocr/english/rec.onnx, dict.txt
   ```

   Without the models the server still starts and is healthy, `GET /health` reports
   `ocr_available: false`, and every OCR call answers 503.
4. **Sidecar config.** `deploy/docker/gacha-server.toml` is mounted read-only at
   `/app/config.toml`. It sets `server.host = "0.0.0.0"`, which the container needs: the
   server's default `127.0.0.1` is the sidecar container's own loopback and the `alas`
   container cannot reach it. The `[ai]` block holds the endpoint and the model. Set
   `GACHA_SERVER_CONFIG` in `.env` to mount another file.

Decision history is kept in the named volume `sidecar-state`, mounted at `/app/state`.

The compose file defines no health check for the service. The image has no `curl` and
brings its own check, which probes `GET /health` on port 8484; `depends_on` in the `alas`
service waits for it.

Port 8484 is published on the host's loopback only (`127.0.0.1:8484:8484`), because the
sidecar has no authentication. The `alas` container does not use the published port.

The Python sidecar in `./ai_sidecar` is no longer part of `docker-compose.yml`. It has no
`/ocr/recognize`. The directory stays in the tree: its `tests/` hold the device-free tests
of this fork.

## How ALAS reaches the sidecar

ALAS has two settings for the sidecar, one for OCR and one for the AI hook.

| Used for | Setting | Default | Under Docker Compose |
|----------|---------|---------|----------------------|
| OCR      | `OcrClientAddress` in `config/deploy.yaml`, `host:port` without `http://` | `127.0.0.1:8484` | `sidecar:8484` |
| AI hook  | `AiSidecar_SidecarUrl` in `module/config/config_generated.py` | `http://127.0.0.1:8484` | `http://sidecar:8484`, from the environment variable `AI_SIDECAR_URL` |

Inside the `alas` container `127.0.0.1` is the container itself, so both have to name the
`sidecar` service instead.

- **OCR.** `config/deploy.template-docker.yaml` and `config/deploy.template-docker-cn.yaml`
  set `OcrClientAddress: sidecar:8484`. Copy one of them to `config/deploy.yaml` before the
  first start. If `config/deploy.yaml` does not exist, ALAS creates it from `deploy/template`
  with `127.0.0.1:8484`, which does not work in the container.
- **AI hook.** `docker-compose.yml` sets `AI_SIDECAR_URL=http://sidecar:8484` on the `alas`
  service. `module/ai_hook/hook.py` uses that variable in place of `AiSidecar_SidecarUrl`
  whenever it is set. Outside Docker the variable is unset and the default applies.

`UseOcrServer` in `config/deploy.yaml` selects the OCR path:

- `true` (every shipped template) -- send OCR to the HTTP server at `OcrClientAddress`.
- `false` -- every ALAS instance loads cnocr and mxnet in-process, which only works where
  both are installed (x86 only). The `alas` image has no mxnet.

There is no local fallback. If the server is unreachable or answers with a 5xx, the request
is tried 3 times and then raises; a 4xx raises immediately.

Outside Docker Compose:

- ALAS and `gacha-server` on the same host: `127.0.0.1:8484`, the default.
- ALAS in a container started with `--net=host` (`deploy/docker/Docker-run.sh`):
  `127.0.0.1:8484`.

## Existing installs

An existing `config/deploy.yaml` is not migrated. It keeps the values it already has, and
they have to be changed by hand to move an install to the sidecar:

- An install that says `UseOcrServer: false` keeps running cnocr in-process.
- A Docker install made before the sidecar says `OcrClientAddress: ocr:22268`. Change it to
  `sidecar:8484`; the `ocr` service no longer exists.

The legacy cnocr server (cnocr/mxnet over zerorpc on port 22268) has been removed, together
with its `ocr` compose service, `deploy/docker/Dockerfile.ocr`, and the deploy settings
`StartOcrServer` and `OcrServerPort`. Nothing reads those two keys any more. Lines for them
in an existing `config/deploy.yaml` are ignored and disappear the next time ALAS writes the
file.

## Files

- `docker-compose.yml` -- the `alas` and `sidecar` services
- `deploy/docker/Dockerfile` -- main ALAS container (multi-arch, no platform pin)
- `deploy/docker/requirements.txt` -- Python deps for ALAS (no mxnet)
- `deploy/docker/gacha-server.toml` -- config mounted into the `sidecar` service
- `.env.example` -- provider key and optional paths for the `sidecar` service
- `deploy/template` -- source of every `config/deploy.template*.yaml`; regenerate them with
  `ConfigGenerator.generate_deploy_template()` in `module/config/config_updater.py`
- `config/deploy.template-docker.yaml` -- deploy config for Docker Compose
- `module/ocr/rpc.py` -- HTTP OCR client
- `module/ai_hook/hook.py` -- AI hook client
- `ai_sidecar/` -- the earlier Python sidecar and the device-free tests
