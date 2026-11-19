# This fork: the sidecar, Docker, and what differs from upstream

This is a fork of [LmeSzinc/AzurLaneAutoScript](https://github.com/LmeSzinc/AzurLaneAutoScript).
The branch it runs is `recover/ai-hook-2026-07-16`, which is also the repository's default
branch; `master` is an older state and is not maintained. The fork adds an AI hook in the
scheduler and sends OCR to a separate server, the Rust sidecar `gacha-server`. The first half of
this page is the Docker setup; [What the fork changes in ALAS](#what-the-fork-changes-in-alas)
describes the behaviour.

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

   Without the model files the server still starts and is healthy. Docker creates the
   mounted directory if it is missing, so the server sees a models directory, `GET /health`
   reports `ocr_available: true`, and every OCR call answers 422 `recognition_failed`; ALAS
   then stops on its first reading with "OCR server failed to process the request". Only a
   server whose models directory does not exist at all reports `ocr_available: false` and
   answers 503.
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

The Python sidecar that used to live in `./ai_sidecar` is gone: nothing ran it after the Rust
sidecar replaced it, and it had no `/ocr/recognize`. The directory stays for its `tests/`,
which hold the device-free tests of this fork.

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

There is no local fallback. What the OCR client does when a request fails is under
[OCR](#ocr) below.

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

## What the fork changes in ALAS

### The AI hook

`alas.py` reports four events to the sidecar (`POST /api/events`, through
`module/ai_hook/hook.py`) and applies the directive that comes back.

| Event | When | What is sent |
|---|---|---|
| `cycle_start` | Each pass of the scheduler loop, after the next task is chosen; at most once in 60 seconds | A fresh screenshot and the pending and waiting task lists |
| `unknown_state` | A task ends in `GameStuckError` or `GameTooManyClickError` | The last frame the task saw, the click history, the task, the error, the task's failure count |
| `task_failed` | A task returns failure and was not deferred | The same stored frame, the task, the failure count |
| `task_complete` | A task returns success | The task and its duration; the answer is not used |

| Directive | What ALAS does |
|---|---|
| `continue`, anything unknown, no answer | Nothing. ALAS goes on as upstream would |
| `skip` | Defers the task: sets its `NextRun` to now plus the requested delay, kept between 15 minutes and one minute under the task's own interval ceiling, and checks the value was saved. The task is never disabled. After a skip on `unknown_state` the game is restarted only if the screen is not a page ALAS recognises; on a recognised page the next task simply starts. A skip does not clear the task's failure count: a third failure in a row still ends the scheduler run, as upstream does |
| `reprioritize` | Only at `cycle_start`. Ranks the tasks that are already due in the order given, for that one selection. `NextRun` is not changed, and once a task is picked ALAS's own priority applies again, so a higher-priority task can still pre-empt it |
| `pause` | At `cycle_start`: ends this scheduler run and sends the push notification. On a task event it is ignored until the task has failed twice in a row |

The hook never stops ALAS: if the sidecar is down, slow or answers nonsense, the event is
dropped, a warning goes to the ALAS log, and ALAS carries on. One event waits at most 5 seconds
to connect and 30 seconds for an answer.

The hook is on by default and has no switch in the GUI. `AiSidecar_Enabled` and
`AiSidecar_SidecarUrl` exist only in `module/config/config_generated.py`, kept there by hand:
running ALAS's config generator removes them, which turns the hook off without a message. The
environment variable `AI_SIDECAR_URL` overrides the URL.

Known gaps, left as they are:

- A deferral does not count as a failure, so a task that sticks every time can be deferred again
  and again with only an info line in the log. One stuck task is also reported twice, as
  `unknown_state` and then `task_failed`.
- When the next task is more than two hours away, ALAS closes the game for the wait whatever
  `WhenTaskQueueEmpty` says. Unless that setting is `close_game`, it then starts the waited task
  before the Restart it scheduled, so that task fails once with the game not running.

### OCR

`module/ocr/rpc.py` sends every reading to the sidecar (`POST /ocr/recognize`).

- **A sidecar that is down or restarting** (connection refused or reset, or HTTP 503) is retried
  eight times over about 32 seconds, 48 on Windows loopback. A read timeout or another 5xx is
  tried three times. A 4xx raises at once. When the retries run out the request raises and
  ALAS stops.
- **One result per image.** A response with a different number of results raises.
- **No reading is rejected for low confidence.** A reading with a character whitelist (a number,
  a counter, a timer) that the sidecar scores under 0.5 is logged as `OCR low confidence` and
  used as it is.
- **Oil on EN** is read from an area 8 px wider than upstream's. A five-digit amount reaches
  the edge of upstream's area and loses a digit, which could end a campaign on a false "Reach
  oil limit".
- **Research project names.** The sidecar's models cannot read the D of the font on the
  project cards: it comes back as `0`, as `1`, or not at all, and a C is sometimes lost the
  same way. The other letters are read. `module/research/project.py` therefore:
  - treats a leading `0` as upstream treats `1` and `I`, as a letter that was not read, and
    names it D or C from the project number;
  - gives a name with no letter D, C, Q, G and E in that order and takes the first that is a
    project of the card's series (upstream tries Q, G, E).

  Five pairs of projects differ by that letter only: S4 `063-UL` (D, H), S4 `068-UL` (D, Q),
  S4 `368-MI` (D, Q), S6 `339-MI` (D, H) and S8 `387-MI` (D, H). D is taken, and a warning
  names the other project. A name that lost both its letter and its ending (`-774-M`) stays
  invalid, as upstream leaves it.
- **What the research log says.** A name read correctly logs nothing. Otherwise one line says
  how it was matched: `is revised to` (a repaired letter or ending), `has no genre letter,
  taken as`, `the C was read as D`, or `matched ... by genre and number, its ending was not
  read`. `Invalid research` gives what the OCR returned and the cause: `no series N project has
  that name`, `the series was not read`, or `no name was read`. When the retries on one screen
  run out, `Research names still invalid:` lists the cards left out of the selection.
- **`cnocr` on the CN server** is served by the sidecar's English model and cannot read Chinese.
  EN is not affected.

### Starting the game

Since version 9.3.23 the EN client opens with a "Download Game Assets" dialog. Its red X exits
the game, and ALAS's image of a maintenance notice's close button (`MAINTENANCE_ANNOUNCE`)
matches that X: upstream taps it, the game closes, and the Restart task fails every time.

- `handle_assets_download` in `module/handler/info_handler.py` looks for the dialog first, in
  the login loop and in the page check every task starts with, and the X is never tapped.
- **Nothing to download** ("Download basic assets (0B)"): ALAS taps Confirm, which leads to the
  title screen, and logs `Download Game Assets dialog with nothing to download (0B), confirm`.
- **Any other size, or a size that was not read:** ALAS taps nothing, because which assets to
  download is the player's choice. It logs a warning that says so, waits on the dialog until
  its own stuck check ends the task, and after three failed restarts the scheduler stops.
  Choose in the game, then start ALAS again.
- The two images are `assets/cn/handler/LOGIN_ASSETS_DOWNLOAD*.png`, taken from the EN client.
  They sit in the `cn` folder because ALAS generates a button only from a file there; the
  dialog was not seen on the other servers, where the text differs and nothing will match.

### Updates

`deploy/template` and every `config/deploy.template*.yaml` name this fork and its branch and set
`AutoUpdate: false`. With `AutoUpdate: true` ALAS runs `git reset --hard` to
`Repository`/`Branch` at startup and at `AutoRestartTime`; pointed at upstream, that replaces
the fork's code in the checkout. An existing `config/deploy.yaml` keeps whatever it says: check
those three settings in it by hand.

### Tests

The device-free tests of everything above are in `ai_sidecar/tests`. They need ALAS's Python
dependencies and pytest, and run from the repository root:

```
python -m pytest ai_sidecar/tests
```

Running them creates `config/deploy.yaml` and `log/` in the working directory; both are
ignored by git.

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
- `alas.py` -- where the hook's events are raised and its directives applied
- `ai_sidecar/tests/` -- the device-free tests
