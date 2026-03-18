# AI Sidecar for ALAS — Design Spec

## Overview

Add AI-driven task prioritization to AzurLaneAutoScript (ALAS) via an event-driven sidecar service. The AI sidecar runs alongside ALAS in Docker, receives game state events, consults an OpenAI-compatible AI provider, and returns task directives.

This is the **Option A foundation** of a hybrid AI integration (Option C). Option A focuses solely on strategic task prioritization. The architecture is designed to later support vision-based fallback and real-time AI intervention (Option C) without rearchitecting.

## Goals

- AI decides task execution order based on game context (event deadlines, resource state, daily resets)
- Pluggable AI provider via OpenAI-compatible API (OpenAI, Claude via proxy, Ollama, LM Studio, vLLM, etc.)
- Zero impact to ALAS when the sidecar is unavailable — graceful fallback to default behavior
- Dockerized for portability between macOS dev and home lab hosting
- Minimal changes to the existing ALAS codebase

## Non-Goals (for this milestone)

- Fleet composition decisions
- Resource management (oil/cube spending)
- Vision-based game control (tap/swipe from AI)
- Multi-turn AI reasoning for stuck states
- Local model hosting (works if provider is OpenAI-compatible, but not a design focus)

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Docker Compose                                  │
│                                                  │
│  ┌──────────────┐       ┌────────────────────┐  │
│  │  ALAS         │       │  AI Sidecar        │  │
│  │  (existing)   │──────▶│                    │  │
│  │              │  HTTP  │  - Event receiver  │  │
│  │  + thin hook │◀──────│  - State model     │  │
│  │    layer     │       │  - AI client       │  │
│  └──────┬───────┘       │  - Directive engine │  │
│         │               └────────┬───────────┘  │
│         ▼                        ▼               │
│   Android Emulator      OpenAI-compatible API    │
│   (host machine)        (Claude/OpenAI/Ollama)   │
└─────────────────────────────────────────────────┘
```

Two Docker containers communicating over internal Docker networking via REST.

---

## Event Triggers (ALAS → Sidecar)

The hook layer in ALAS fires an HTTP POST to the sidecar at specific moments. All calls are fire-and-wait with a short timeout; if the sidecar is unreachable, ALAS proceeds with default behavior.

### Events

| Event | When it fires | Payload |
|---|---|---|
| `cycle_start` | Beginning of a new scheduler loop (debounced — at most once per 60s) | Full task list (enabled/disabled, next_run times), screenshot (base64) |
| `task_complete` | A task finishes successfully | Task name, duration (elapsed seconds) |
| `task_failed` | Any individual task failure (count 1+, before the hard exit at 3) | Task name, error type, failure count |
| `unknown_state` | `GameStuckError` or `GameTooManyClickError` caught in `alas.py` `run()` | Screenshot (base64), last 15 click history |

### Directive Responses

The sidecar responds with a JSON directive:

```json
{"action": "continue"}
```
No changes, proceed as normal.

```json
{"action": "reprioritize", "task_order": ["Event_A", "Daily", "Commission", ...]}
```
Reorder the task queue. Only enabled tasks may appear.

```json
{"action": "skip", "task": "Exercise"}
```
Skip a specific task this cycle.

```json
{"action": "pause", "reason": "Event ends in 30 minutes — manual intervention recommended"}
```
Stop automation, surface reason to user via ALAS web GUI.

---

## Sidecar Internals

### Tech Stack

- Python 3.11+
- FastAPI + Uvicorn (single worker — see State Model for rationale)
- `openai` Python SDK (for OpenAI-compatible API calls)
- Pydantic for request/response validation

### Components

#### 1. Event Receiver (`/api/events`)

- POST endpoint accepting event payloads
- Validates against Pydantic models
- Updates the state model
- Decides whether to consult the AI (configurable per event type via `consult_on`)
- Returns directive to ALAS

#### 2. Health Check (`GET /health`)

- Returns `{"status": "ok"}` for Docker health checks and ALAS connectivity verification
- Used by Docker Compose `healthcheck` and optionally by the hook layer on startup

#### 3. State Model

In-memory game state built incrementally from events:

- Current task queue and enabled/disabled states
- Task completion history (last N completions with timestamps)
- Task failure history
- Resource levels (oil, coins, cubes — from OCR data in payloads)
- Event timers (active events, end dates)
- Last screenshot

Persisted to `/data/state.json` via atomic writes (write to temp file, then `os.rename`) on every update so state survives container restarts. Sidecar must run with a single Uvicorn worker to avoid concurrent write issues.

#### 4. AI Client

- Uses the `openai` Python SDK with configurable `base_url`, `api_key`, and `model`
- Constructs messages:
  - **System prompt**: game context, available tasks, decision criteria, output format (JSON schema)
  - **User message**: current state model (structured text) + event details + screenshot (base64 image) + time context
- Parses AI response as JSON directive
- If AI returns unparseable response or is unreachable: returns `{"action": "continue"}`
- Configurable timeout (default 30s) and retry count (default 1)

#### 5. Directive Engine

- Validates AI-generated directives:
  - `reprioritize`: all tasks must exist and be enabled
  - `skip`: task must exist
  - `pause`: reason must be non-empty
- Invalid directives fall back to `{"action": "continue"}`
- Logs all AI decisions (event, state snapshot, AI response, final directive) to `/data/decisions.log`
- Log rotation: 10MB max file size, keep last 5 rotated files (Python `RotatingFileHandler`)

### Configuration (`config.yaml`)

```yaml
ai:
  base_url: "https://api.openai.com/v1"
  api_key: "${AI_API_KEY}"
  model: "gpt-4o"
  timeout: 30
  max_retries: 1

sidecar:
  port: 8484
  consult_on:
    - task_failed
    - cycle_start
    - unknown_state
  cycle_start_cooldown: 60  # seconds — debounce AI calls on cycle_start
  state_file: "/data/state.json"
  decision_log: "/data/decisions.log"
```

`consult_on` values are validated against known event types at startup — unknown event types cause a fast failure with a clear error message.

`task_complete` is received and updates state but does not trigger an AI call by default (configurable by adding it to `consult_on`).

---

## ALAS Hook Layer

### New File: `module/ai_hook/hook.py`

`AISidecarClient` class:

- `__init__(config)`: reads sidecar URL from ALAS config
- `notify(event_type: str, payload: dict) -> Optional[Directive]`: POSTs event to sidecar, returns parsed directive or `None`
- `screenshot_to_base64(image: np.ndarray) -> str`: converts `self.device.image` (numpy ndarray) to base64 via `cv2.imencode('.png', image)` + `base64.b64encode`
- All network calls wrapped in try/except with short timeout (5s connect, 30s read)
- Debounce logic: tracks `last_consulted` timestamp, skips `cycle_start` AI calls if within cooldown period
- Includes latest screenshot as base64 (via `screenshot_to_base64`) when `event_type` is `cycle_start` or `unknown_state`

**Python 3.7 compatibility note:** The hook layer runs inside the ALAS container which uses Python 3.7. The hook code must avoid 3.8+ features (no walrus operator, no `typing.Literal`, use `typing.Optional` instead of `X | None`, etc.). Only standard library + `requests` for HTTP calls.

### New Config Section in ALAS

```
[AiSidecar]
Enabled = True
SidecarUrl = http://sidecar:8484
```

When `Enabled = False`, the hook layer is a no-op.

### Changes to `alas.py` (~40 lines)

All hooks are placed in `alas.py` only — not in `device.py`. The sidecar client is stored on the `AzurLaneAutoScript` instance so it's accessible in the main loop and exception handlers without needing to pass it into `Device`.

```python
# In loop() setup
from module.ai_hook.hook import AISidecarClient
sidecar = AISidecarClient(self.config) if self.config.AiSidecar_Enabled else None

# Before get_next_task() — cycle_start
# Note: self.device may not be initialized on first iteration (it's a cached_property).
# Take a fresh screenshot if device is available; send None otherwise.
# Note: on the first iteration, pending_task and waiting_task may be empty lists
# because get_next_task() has not been called yet. The sidecar handles empty
# task lists gracefully (treats as "initial state, no opinion yet").
if sidecar:
    screenshot = None
    try:
        _ = self.device  # ensure device is initialized
        self.device.screenshot()
        screenshot = sidecar.screenshot_to_base64(self.device.image)
    except Exception:
        pass  # device not ready yet, send event without screenshot
    directive = sidecar.notify("cycle_start", {
        "pending_tasks": [str(t) for t in self.config.pending_task],
        "waiting_tasks": [str(t) for t in self.config.waiting_task],
        "screenshot": screenshot
    })
    if directive:
        self._apply_directive(directive)

# After successful task execution (run() returns True)
if sidecar:
    sidecar.notify("task_complete", {
        "task": task,
        "success": True,
        "duration": elapsed_seconds
    })

# After task failure (run() returns False), in the else branch (~line 577)
# Note: the exception is caught inside run(), so we use failure_record for context
if sidecar:
    directive = sidecar.notify("task_failed", {
        "task": task,
        "error": "task_returned_failure",
        "count": deep_get(self.failure_record, keys=task, default=0)
    })
    if directive:
        self._apply_directive(directive)

# In except (GameStuckError, GameTooManyClickError) handler (~line 77 in run())
# Note: _apply_directive is safe here — task_delay only modifies next_run timestamps
# which are re-read on the next get_next_task() call. It does not affect the current
# exception handling flow. After this, run() returns False and loop() continues normally.
if sidecar:
    directive = sidecar.notify("unknown_state", {
        "screenshot": sidecar.screenshot_to_base64(self.device.image),
        "click_history": [str(c) for c in self.device.click_record]
    })
    if directive:
        self._apply_directive(directive)
```

### New Method: `_apply_directive(directive)` on `AzurLaneAutoScript`

Translates directives into scheduler actions:

- **`reprioritize`**: Manipulates `next_run` times on individual tasks via `self.config.task_delay(target=datetime, task=name)`. The first task gets `target = now + 1s`, second gets `now + 2s`, etc. Using the `target` parameter (which accepts a `datetime` directly) rather than `minute` avoids fragile fractional-minute arithmetic. All targets are slightly in the future so they land in `waiting_task` (sorted by `next_run`) rather than `pending_task` (sorted by `SCHEDULER_PRIORITY`, which would override the AI's order). The scheduler naturally picks the waiting task with the earliest `next_run`. **Performance note:** calling `task_delay` N times triggers N config saves. For large task lists (ALAS has 40+ tasks), consider batching by manipulating `self.config.modified` directly and calling `self.config.update()` once.
- **`skip`**: Calls `self.config.task_delay(task=task_name, minute=1440)` to push the task 24 hours into the future (past the current cycle).
- **`pause`**: Sets `self.stop_event.set()`, which is the existing mechanism for GUI-triggered stops. The `loop()` method's `while 1` checks this event and breaks cleanly. A log message with `directive.reason` is emitted so the user can see why automation stopped. **Guard:** `stop_event` is `None` when ALAS runs in standalone mode (not from the web GUI). The implementation must check `if self.stop_event is not None` before calling `.set()`. When `stop_event` is `None`, fall back to logging a warning ("pause requested but not supported in standalone mode") and returning `{"action": "continue"}`. **Note:** `task_stop()` is NOT used here — it raises `TaskEnd` which would either be swallowed by `run()` or propagate uncaught from `loop()`. `stop_event` is the correct mechanism for halting the scheduler.

This avoids creating new methods on `AzurLaneConfig` — it uses existing APIs (`task_delay`, `stop_event`) to achieve the desired effect.

**Runtime config changes:** The `AISidecarClient` is created at the start of `loop()`. If ALAS config is reloaded at runtime (via `del_cached_property`), the sidecar client should be re-created. This is handled by checking `self.config.AiSidecar_Enabled` on each loop iteration rather than caching the decision once.

### Key Principle

The hook layer is a satellite. Removing `module/ai_hook/` and the ~40 lines in `alas.py` restores ALAS to its original behavior. No existing logic is modified — only augmented with optional calls.

---

## AI Prompt Strategy

### System Prompt (stored as editable template file)

Contents:
- Brief Azur Lane gameplay description (resource types, daily reset, event mechanics)
- Full list of automatable tasks with short descriptions
- Decision criteria and heuristics:
  - Time-limited events take priority over repeatable content
  - Daily tasks should complete before daily reset
  - Failed tasks may indicate a bug — deprioritize rather than retry endlessly
  - Resource thresholds (if provided) inform stop decisions
- Output format: strict JSON matching the directive schema
- Constraints: only reference enabled tasks, never invent task names

### User Message (constructed per event)

- Structured game state (text): task queue, completion history, failures, resources, timers
- Event details: what just happened and why the AI is being consulted
- Screenshot: base64-encoded, when available
- Time context: current server time, time until daily reset, active event deadlines

### Response Parsing

- AI instructed to respond with JSON only (no markdown, no explanation)
- Parsed against Pydantic directive model
- Unparseable responses → `{"action": "continue"}`

Prompt templates stored in `ai_sidecar/prompts/` as editable `.txt` files.

---

## Docker Compose

```yaml
services:
  alas:
    build:
      context: .
      dockerfile: deploy/docker/Dockerfile
    ports:
      - "22267:22267"
    volumes:
      - ./config:/app/AzurLaneAutoScript/config
      - alas-data:/app/data
    environment:
      - AI_SIDECAR_URL=http://sidecar:8484
      - EMULATOR_SERIAL=host.docker.internal:5555
    extra_hosts:
      - "host.docker.internal:host-gateway"  # required for Linux Docker
    depends_on:
      sidecar:
        condition: service_healthy

  sidecar:
    build:
      context: ./ai_sidecar
      dockerfile: Dockerfile
    ports:
      - "8484:8484"
    volumes:
      - sidecar-data:/data
      - ./ai_sidecar/prompts:/app/prompts
    environment:
      - AI_API_KEY=${AI_API_KEY}
      - AI_BASE_URL=${AI_BASE_URL:-https://api.openai.com/v1}
      - AI_MODEL=${AI_MODEL:-gpt-4o}
    env_file:
      - .env
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8484/health"]
      interval: 10s
      timeout: 5s
      retries: 3

volumes:
  alas-data:
  sidecar-data:
```

Notable:
- No `network_mode: host` — uses Docker internal networking with explicit port mappings
- AI credentials via `.env` file, never in image
- Prompt templates mounted as volume for easy iteration
- Sidecar port exposed for development debugging
- Health check ensures sidecar is ready before ALAS starts
- Sidecar health endpoint includes API version: `{"status": "ok", "api_version": "1.0"}` — the hook layer checks compatibility on startup and logs a warning if versions don't match

### Migration from Existing `docker-compose.yml`

The existing `docker-compose.yml` uses `network_mode: host`. This new compose file replaces it with Docker internal networking. This is a **breaking change**:

1. **Emulator connectivity**: Previously, ALAS reached the emulator at `localhost:5555`. Now it must use `host.docker.internal:5555`. Users with custom ADB serial configs (e.g., `127.0.0.1:62001` for LDPlayer) must update to `host.docker.internal:<port>`.
2. **Web GUI**: Previously exposed via host networking automatically. Now requires explicit `ports: ["22267:22267"]` (already in the compose file above).
3. **Platform support**:
   - macOS Docker Desktop: `host.docker.internal` works natively
   - Linux Docker 20.10+: `extra_hosts` with `host-gateway` (included above) provides equivalent functionality
   - Linux Docker <20.10: not supported — users must upgrade Docker or continue using a `network_mode: host` override
4. **Opting out of sidecar**: Users who want the Docker networking fix without the AI sidecar can remove the `sidecar` service and `depends_on` from the compose file. The hook layer is a no-op when `AiSidecar.Enabled = False`.

---

## ALAS Dockerfile Compatibility

The existing `deploy/docker/Dockerfile` uses Python 3.7. The hook layer (`module/ai_hook/`) must be written to be Python 3.7-compatible since it runs inside the ALAS container. The only additional dependency is `requests` (likely already available, but to be verified; fallback: `urllib.request` from stdlib).

The sidecar has its own Dockerfile with Python 3.11+ — no version conflict.

---

## Option C Upgrade Path

The architecture is designed so that Option C (hybrid AI with vision fallback) layers on naturally:

1. **`unknown_state` event** already sends screenshots — extend the sidecar to do multi-turn vision reasoning for stuck states
2. **New directive types** (e.g., `{"action": "tap", "x": 640, "y": 360}`) added to the directive schema
3. **Expanded hook in `alas.py`** to accept and execute vision-based action directives
4. **Periodic oversight loop** added to the sidecar — polls ALAS for screenshots on a timer, can proactively intervene
5. **AI tool-use** — give the AI function-calling capability to request specific game info

None of these require rearchitecting — they extend the existing event/directive protocol.

---

## File Structure (new files)

```
AzurLaneAutoScript/
├── ai_sidecar/                    # New: sidecar service
│   ├── Dockerfile
│   ├── requirements.txt           # fastapi, uvicorn, openai, pydantic
│   ├── config.yaml
│   ├── main.py                    # FastAPI app entry point
│   ├── models.py                  # Pydantic models (events, directives, state)
│   ├── state.py                   # State model management
│   ├── ai_client.py               # OpenAI-compatible API client
│   ├── directive_engine.py        # Directive validation and logging
│   └── prompts/
│       ├── system.txt             # System prompt template
│       └── user_template.txt      # User message template
├── module/
│   └── ai_hook/                   # New: ALAS hook layer (Python 3.7 compatible)
│       ├── __init__.py
│       └── hook.py                # AISidecarClient
├── docker-compose.yml             # Updated: replaces existing, adds sidecar service
└── .env.example                   # New: template for AI credentials
```

Changes to existing files:
- `alas.py`: ~40 lines added (hook calls in main loop + exception handlers, `_apply_directive` method)
- `module/config/config.py`: new `AiSidecar` config section (Enabled, SidecarUrl)
- `deploy/docker/Dockerfile`: add `requests` to pip install if not already present
