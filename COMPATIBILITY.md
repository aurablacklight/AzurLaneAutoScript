# Multi-Architecture Docker Compatibility

## Three-Container Architecture

The Docker deployment uses three containers:

| Container | Service name | Port  | Purpose |
|-----------|-------------|-------|---------|
| ALAS      | `alas`      | 22267 | Main application and web UI |
| OCR Server| `ocr`       | 22268 | Optical character recognition via cnocr/mxnet |
| AI Sidecar| `sidecar`   | 8484  | AI-powered gameplay assistance |

## Why OCR is a Separate Container

The OCR engine depends on `mxnet 1.6.0`, which only has x86 (amd64) builds. On Apple
Silicon (arm64) Macs, mxnet crashes under Python when loaded in-process. Splitting OCR
into its own container lets us:

- Run the OCR server under x86 emulation (Rosetta 2 / QEMU) in an isolated container.
- Run the main ALAS container natively on arm64 for full performance.
- Keep everything working identically on amd64 (x86) hosts where all containers run
  natively.

## Architecture Behavior by Host

### amd64 (x86_64) Hosts
All three containers run natively. No emulation overhead.

### arm64 (Apple Silicon) Hosts
- **ALAS** (`alas`) -- runs natively on arm64.
- **OCR Server** (`ocr`) -- runs under x86 emulation (`--platform=linux/amd64`).
  Docker Desktop on macOS uses Rosetta 2 for this automatically.
- **AI Sidecar** (`sidecar`) -- runs natively on arm64.

## RPC Architecture

ALAS communicates with the OCR server over [zerorpc](https://www.zerorpc.io/) (TCP):

```
ALAS container  --zerorpc-->  OCR container
 (port 22267)                  (port 22268)
```

The connection is configured in the deploy template:

- `UseOcrServer: true` -- tells ALAS to use the remote OCR server instead of loading
  mxnet in-process.
- `OcrClientAddress: ocr:22268` -- the Docker Compose service name `ocr` resolves to the
  OCR container's IP on the shared network.
- `StartOcrServer: false` -- the OCR server is started by Docker Compose, not by the
  ALAS GUI.

The OCR server loads cnocr models from `./bin/cnocr_models/` (mounted via the source
volume) and exposes them over zerorpc. If the OCR server is unreachable, ALAS falls back
to loading mxnet locally (which will fail on arm64 -- so the OCR container must be
running).

## Files

- `deploy/docker/Dockerfile` -- main ALAS container (multi-arch, no platform pin)
- `deploy/docker/Dockerfile.ocr` -- OCR server container (pinned to linux/amd64)
- `deploy/docker/requirements.txt` -- Python deps for ALAS (no mxnet)
- `deploy/docker/requirements-ocr.txt` -- Python deps for OCR server (mxnet + zerorpc)
- `docker-compose.yml` -- orchestrates all three containers
- `config/deploy.template-docker.yaml` -- deploy config with `UseOcrServer: true`
- `module/ocr/rpc.py` -- OCR RPC server and client implementation
