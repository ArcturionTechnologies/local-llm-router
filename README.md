# local-llm-router

A local-first LLM router for Apple Silicon. It sends each request to an on-device
model first (MLX, LM Studio, or any OpenAI-compatible local server), **escalates to a
bigger tier only when the task needs it**, and **backs off when the Mac is running hot
or busy**. A rate guard, a single-heavy-model lock and a budget tracker keep the
whole thing from eating your machine, your subscription limits or your API bill.

Pure standard library. Python 3.11+. macOS first (Apple Silicon), works on Linux with a
reduced thermal probe.

> Built by Robert Lingoes with AI coding agents (Claude Code / Codex); Robert owns the architecture, requirements and review.

## The problem

If you run scheduled agents, heartbeats and batch jobs, most of what you ask an LLM is
small: label this, pull these fields out, is this worth a bigger model? Sending all of it
to a frontier API is slow, expensive and burns rate limits that your hard problems need.
Sending all of it to local models is cheap, but a laptop running a 30B model while you
also compile, video-call and run Docker will swap, throttle and stall.

So you need a router that is honest about both constraints:

* **Pick the smallest tier that can do the job**, and let the caller escalate when it can't.
* **Treat the machine as a shared resource.** Don't start inference when the Mac is already
  hot, low on memory, unplugged and nearly flat, or already holding another big model.
* **Never block the caller.** Every failure mode (hot host, busy lock, server down, rate
  limit, empty answer) falls through to the next tier with a recorded reason.

## How it works

```
caller ──► pick_tier(task_class, complexity, stakes, bulk, context)
                │            (or an explicit tier=..., or tier_call(complexity=...))
                ▼
   ┌─────────────────────────── LOCAL · free · private ────────────────────────────┐
   │  fast ──► small ──► medium ──► coder ──► heavy          vision               │
   │  4B       8B        14B        14B code   30B MoE         VL-7B               │
   │  light    standard  medium     coder      heavy ◄── model lock               │
   └───────────────┬─────────────────────────────────────────────────────────────┘
                   │  hot host · busy lock · server down · rejected answer
                   ▼  (each tier lists its own `fallbacks`)
   ┌──────────────────────────── CLOUD · escalation ─────────────────────────────┐
   │  free:  groq ──► cerebras ──► gemini (1M ctx)                                │
   │  paid:  haiku ──► sonnet ──► opus   (strongest, for irreversible work)       │
   └──────────────────────────────────────────────────────────────────────────────┘

for every tier tried:   thermal gate ─► [model lock] ─► HTTP ─► telemetry + budget
```

A call walks a **chain**: the starting tier followed by its `fallbacks`, depth-first and
de-duplicated. For `heavy` the default chain is `heavy → medium → small → groq → cerebras → gemini`:
step *down* locally under pressure, then *up* to the cloud. Every attempt, including the
refused ones, is logged with why it was skipped.

Escalation happens three ways:

1. **By task signature**: `pick_tier()` encodes hard rules (irreversible → strongest tier,
   > 100k tokens → long-context tier, > 20-item bulk loops never touch paid tiers) and
   local-first defaults (classify → `fast`, code → `coder`, reasoning → `medium`).
2. **By complexity**: `tier_call(prompt, complexity="auto")` asks a small local model to rate the
   task `trivial | routine | hard`, then starts at the tier you mapped to that rating
   (default: `fast / medium / sonnet`, and `opus` for `critical`).
3. **By answer quality**: pass `accept=lambda text: ...`; if the tier's answer is rejected the
   call moves up the chain instead of returning it.

## The thermal / heartbeat gate

Local inference is only free if the machine has headroom. The thermal gate classifies the
host before every local call:

| Level | Trigger (all thresholds configurable) |
|---|---|
| **red** | CPU load > 80% of cores · free memory < 8% · on battery below 15% · macOS CPU speed limit < 90 |
| **yellow** | CPU load > 50% of cores · free memory < 15% |
| **unknown** | load or thermal sensor unreadable (never treated as green) |
| **green** | everything else |

Each tier has a **profile** that decides what yellow/unknown mean *for that model's size*:

| Profile | Red | Yellow | Unknown sensors |
|---|---|---|---|
| `light` (4B) | block | allow | allow |
| `standard` (8B) | block | allow | block |
| `medium` (14B) | block | **downgrade** to a lighter tier | downgrade |
| `coder` / `vision` | block | block | block |
| `heavy` (30B MoE) | block | block | block |
| `none` (cloud) | allow | allow | allow |

"Block" and "downgrade" both mean *skip this tier and try the next fallback*; the difference is
recorded so you can see "the host was busy" versus "the host was on fire". Heavy tiers can also
**demote on memory pressure** (free RAM below `min_free_gb_heavy`, or a configured process pattern
such as another coding-agent session is running).

The built-in probe reads `os.getloadavg`, `memory_pressure`/`vm_stat`, and `pmset` (battery and
`CPU_Speed_Limit`) on macOS. If you would rather bring your own logic, set `thermal.command` to a
script that is called as `<command> <profile>` and exits `0` (go), `1` (lighter tier please) or `2`
(blocked); see `examples/thermal-gate-script.sh`. A missing script fails open, a hung one fails
closed. Set `LLM_ROUTER_BYPASS_THERMAL=1` for lean jobs that must never be gated.

**The heartbeat gate** applies the same idea to *scheduled* work. A cron/launchd job that wakes an
LLM every few minutes wastes tokens when nothing changed. Each heartbeat declares what it watches
(files, directories, command output) in `~/.config/local-llm-router/heartbeat/signals/<id>.json`:

```bash
llm-router heartbeat should-fire inbox || exit 0     # exit 0 = fire, 1 = skip
```

It fires when the fingerprint of those signals changes, or when a sanity floor (default 6 h) elapses
so silent failures don't hide; with no manifest it skips (fail closed); and with `"skip_when_hot": true`
it also skips while the thermal gate is red, **without consuming the change**, so the work happens on
the next cool tick. See `examples/heartbeat-signals/inbox.json`.

## Other guard rails

| Component | What it does |
|---|---|
| **Model lock** (`ModelLock`) | `flock`-based single-holder lock with holder metadata (`llm-router lock status`). Tiers marked `exclusive = true` hold it around each call; if another process holds it the tier answers *busy* and the router falls through. Servers can use it directly to refuse to load a second heavy model. Crashed holders are reaped automatically. |
| **Rate guard** (`RateGuard`) | Call before launching a paid agent session (`claude -p ...` etc.). Counts concurrent sessions and rolling 1 h / 5 h / 7 d spawns against ceilings, scans spawn logs for rate-limit text to start a cool-down, honours a kill-switch file. Exit code 0 ok / 1 throttle / 2 block. |
| **Budget tracker** (`Budget`) | Prices paid usage in USD against a monthly envelope, tracks free-tier daily token quotas, and reports **displacement**: what share of calls stayed local. Optional enforcement refuses a paid tier once the envelope is spent. |
| **Local-first policy** (`LocalFirstGate`) | Warn once, then block, routine work sent to paid APIs in one session. Also a Claude Code `PreToolUse` hook that spots `claude -p` / SDK calls in shell commands. |

## Install

```bash
git clone https://github.com/ArcturionTechnologies/local-llm-router
cd local-llm-router
pip install -e .          # no dependencies; adds the `llm-router` command
llm-router tiers          # see the built-in tier table
```

## Quickstart

### A. LM Studio (one server, easiest)

1. In LM Studio load a model (e.g. an MLX Qwen3 8B) and start the local server
   (Developer tab → *Start Server*, default `http://localhost:1234`; or `lms server start`).
2. Point every local tier at it:

```bash
export LLM_ROUTER_LOCAL_BASE_URL=http://127.0.0.1:1234/v1
export LLM_ROUTER_LOCAL_MODEL=qwen3-8b            # the identifier LM Studio shows
llm-router health
llm-router chat "Say hi in five words" --tier small -v
llm-router classify "AAPL beats EPS by 8%" earnings,macro,rumor,noise
```

Or use the ready-made config, which also routes every role to LM Studio and keeps the cloud tiers
as escalation targets: `export LLM_ROUTER_CONFIG=examples/lmstudio.toml`.

### B. `mlx_lm.server` (one process per model)

```bash
pip install mlx-lm
mlx_lm.server --model mlx-community/Qwen3-4B-Instruct-2507-4bit --port 8770 &   # fast
mlx_lm.server --model mlx-community/Qwen3-8B-4bit               --port 8765 &   # small
mlx_lm.server --model mlx-community/Qwen3-14B-4bit              --port 8766 &   # medium
llm-router health          # UP fast / small / medium, DOWN coder / heavy / vision
llm-router pick --task-class classify --complexity simple     # -> fast
llm-router thermal --profile medium ; echo $?                  # 0 go / 1 lighter / 2 blocked
```

Those ports and model ids are the built-in defaults (see `examples/mlx-multi-server.toml`).

### C. Add the cloud tiers (optional escalation)

Cloud tiers read keys from the environment only; nothing is ever stored in a config file.

```bash
export GROQ_API_KEY=...  CEREBRAS_API_KEY=...  GEMINI_API_KEY=...   # free tiers
export ANTHROPIC_API_KEY=...                                           # haiku / sonnet / opus
llm-router tier-call "Design a retry policy for this queue" --complexity hard -v
```

A tier whose key is missing is skipped with `MissingCredentials` in the attempt trail, not an error.

### Python

```python
from local_llm_router import Router, classify, extract_json, summarize, pick_tier

label = classify("AAPL beats EPS", ["earnings", "macro", "rumor", "noise"])
data = extract_json(invoice_text, {"vendor": "string", "amount": "number"})

router = Router()                                   # config file + environment
c = router.complete("Plan a 5-step migration", task_class="reason", complexity="complex")
print(c.text, c.tier, c.escalated, [(a.tier, a.error) for a in c.attempts])

# Start small, escalate when the answer is not good enough:
c = router.complete(prompt, tier="small", accept=lambda t: t.strip().endswith("}"))

# Let a local model rate the task, then start at the matching tier:
c = router.tier_call(prompt, complexity="auto")

# Never fall through; just tell me if this tier can't serve it:
router.complete(prompt, tier="vision", fallback="raise")
```

`fallback` is `"cascade"` (default), `"raise"` (one tier, re-raise its error) or `"skip"` (return an
empty completion instead of raising).

## CLI

```
llm-router chat PROMPT [--tier T] [--system S] [--fallback cascade|raise|skip] [-v]
llm-router classify TEXT a,b,c        llm-router summarize TEXT [WORDS]
llm-router extract TEXT '{"f":"type"}' llm-router route TEXT a,b,c
llm-router escalate PROMPT            # YES / NO: is this worth a bigger tier?
llm-router tier-call PROMPT [--complexity auto|trivial|routine|hard|critical]
llm-router pick [--task-class ..] [--complexity ..] [--stakes ..] [--bulk N] [--context-tokens N]
llm-router tiers | health
llm-router thermal [--profile P] [--json]        # exit 0 go / 1 lighter tier / 2 blocked
llm-router config init|show|path
llm-router budget status|snapshot|displacement|record TIER IN OUT
llm-router lock status|force-release
llm-router guard check|log-spawn CALLER TASK     # exit 0 ok / 1 throttle / 2 block
llm-router heartbeat should-fire|status|reset ID | discover   # should-fire: exit 0 fire / 1 skip
llm-router local-first status|test|clear|hook
```

Exit codes for model commands: `0` success, `1` bad arguments or config, `2` no tier could answer
(so shell callers can decide whether to escalate themselves).

## Configuration reference

Search order: `$LLM_ROUTER_CONFIG` → `$LLM_ROUTER_HOME/config.toml` → `config.json`
(`$LLM_ROUTER_HOME` defaults to `~/.config/local-llm-router`). Unknown keys are an error.
`llm-router config init` writes the full effective defaults as a starting point. Runtime state
(telemetry, budget, locks, rate-guard and heartbeat state) lives in `<home>/state/`, or
`$LLM_ROUTER_STATE_DIR`.

### Environment variables

| Variable | Effect |
|---|---|
| `LLM_ROUTER_HOME` | config + state root (default `~/.config/local-llm-router`) |
| `LLM_ROUTER_CONFIG` | explicit config file |
| `LLM_ROUTER_STATE_DIR` | move state elsewhere |
| `LLM_ROUTER_LOCAL_BASE_URL`, `LLM_ROUTER_LOCAL_MODEL` | retarget **every local tier** at one server/model |
| `LLM_ROUTER_URL_<TIER>`, `LLM_ROUTER_MODEL_<TIER>` | retarget one tier (`LLM_ROUTER_URL_HEAVY`) |
| `LLM_ROUTER_BYPASS_THERMAL=1` | skip the thermal gate (lean/cron jobs) |
| `LLM_ROUTER_CALLER`, `LLM_ROUTER_RUN_ID` | labels written to telemetry |
| `GROQ_API_KEY`, `CEREBRAS_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY` | cloud keys (names configurable per tier) |

### `[tiers.<name>]`

| Key | Default | Meaning |
|---|---|---|
| `kind` | `local` | `local` or `cloud` |
| `protocol` | `openai` | `openai` (`/chat/completions`) or `anthropic` (`/v1/messages`) |
| `base_url` | required | OpenAI style: ends in `/v1`. Anthropic: API origin |
| `model` | `""` | model id sent in the request |
| `api_key_env` | `""` | env var holding the key (cloud tiers) |
| `timeout` | `30.0` | seconds; applies to the requested tier, fallbacks use their own |
| `thermal` | by kind | profile: `none` `light` `standard` `medium` `coder` `vision` `heavy` (cloud → `none`) |
| `exclusive` | `false` | hold the model lock around each call |
| `server` | tier name | holder name written to the lock metadata |
| `demote_on_pressure` | `false` | skip when free RAM is low / pressure process runs |
| `fallbacks` | `[]` | ordered tier names to try next |
| `cost` | by kind | `local` `free` `paid` (budget class) |
| `price_in_per_mtok`, `price_out_per_mtok` | `0` | USD per million tokens (paid tiers; **estimates, edit them**) |
| `daily_token_limit` | `0` | free-tier daily quota for budget warnings (0 = unknown) |
| `context_window` | `32768` | informational |
| `enabled` | `true` | disabled tiers are skipped in chains |
| `extra_payload` | `{}` | extra JSON merged into the request body (e.g. `enable_thinking`) |

Built-in tiers: `fast` `small` `medium` `coder` `heavy` `vision` (local, ports 8770 / 8765 / 8766 /
8772 / 8768 / 8769) and `groq` `cerebras` `gemini` (free cloud) and `haiku` `sonnet` `opus` (paid).
You can override any field of a built-in tier, or define new ones.

### `[roles]` and `[complexity_tiers]`

`roles` maps what `pick_tier()` needs to tier names: `fast`, `helper` (used by classify / extract /
summarize), `small`, `medium`, `coder`, `heavy`, `vision`, `bulk_simple`, `bulk_complex`,
`long_context`, `strong`, `strongest`. `complexity_tiers` maps `trivial | routine | hard | critical`
to the starting tier used by `tier_call`. Entries merge, so you can remap one.

### Other sections

| Section | Keys (defaults) |
|---|---|
| `[routing]` | `bulk_threshold = 20`, `long_context_tokens = 100000` (both "strictly greater than") |
| `[thermal]` | `enabled = true`, `command = ""`, `bypass_env`, `cpu_heavy_ratio = 0.8`, `cpu_moderate_ratio = 0.5`, `mem_low_pct = 15`, `mem_critical_pct = 8`, `battery_low_pct = 15`, `throttle_speed_limit = 90`, `min_free_gb_heavy = 22`, `pressure_process_pattern = ""`, `pressure_ignore = []` |
| `[budget]` | `paid_monthly_usd = 100`, `warn_pct = 80`, `block_pct = 100`, `enforce = false` |
| `[rate_guard]` | `max_concurrent = 3`, `max_1h = 5`, `max_5h = 8`, `max_week = 50`, `cooldown_min = 60`, `yellow_pct = 0.8`, `red_pct = 0.95`, `process_pattern = "claude -p"`, `rate_limit_regex`, `spawn_log_dir`, `registry_path`, `kill_flags` |
| `[local_first]` | `paid_providers`, `exempt_task_classes`, `shell_tools`, `paid_markers` |
| `[telemetry]` | `enabled = true`, `path = ""` (default `<state>/telemetry.jsonl`) |
| `[heartbeat]` | `signals_dir`, `state_dir`, `tasks_dir`, `sanity_fire_hours = 6` |

### Server contract

Any server that answers `POST {base_url}/chat/completions` in the OpenAI shape works; liveness is
`GET {base_url}/models`. Servers that implement the model lock can answer HTTP 503
`{"error": "local-pool-busy", "holder": "...", "holder_pid": 123}` and the router treats it as
"busy, healthy" and falls through (a plain 503 is a real fault).

### Telemetry

One JSON line per attempt in `telemetry.jsonl`: tier, model, cost class, task, latency, tokens
(server-reported when available, else a 4-chars-per-token estimate), `ok`, `error`, `fallback`,
`caller`, `run_id`. `llm-router budget displacement` aggregates it.

## Tests

```bash
pip install -e '.[dev]'
python -m pytest            # or: python -m unittest discover -s tests
```

No real model is ever loaded and no real API is ever called. The tests run the whole stack against
a mock OpenAI/Anthropic-compatible `http.server` on `127.0.0.1` and fake host readings
(`tests/helpers.py`); the thermal gate, model lock (including a real second process), rate guard,
heartbeat gate, budget, config and CLI are all covered.

## Honest limitations

* The built-in thermal probe is a heuristic built on `os.getloadavg`, `memory_pressure` and
  `pmset`; it does not read die temperatures. On Linux there is no thermal signal at all (only
  load and `MemAvailable`). Use `thermal.command` if you want something smarter.
* The model lock serialises *heavy calls routed through this package* (and any server that
  adopts it). It cannot see a model you loaded by hand in another app: check
  `llm-router lock status` and unload it first.
* Token counts are server-reported when the server provides `usage`, otherwise estimated;
  paid-tier dollar figures use the per-million prices in your config, which are estimates.
* The `anthropic` protocol carries text only; image inputs use the OpenAI `image_url` form.
* The default local model ids and ports are suggestions, not requirements.

## Layout

```
local_llm_router/
  config.py      defaults, TOML/JSON loading, env overrides      client.py     HTTP for openai + anthropic
  router.py      pick_tier, chains, escalation, helpers          thermal.py    probe + gate + RAM demotion
  model_lock.py  single-heavy-model flock                        heartbeat.py  change-detection gate
  rate_guard.py  session ceilings, cool-down, kill switch        budget.py     envelopes + displacement
  local_first.py policy gate + PreToolUse hook                   telemetry.py  JSONL log
  cli.py         llm-router
examples/        lmstudio.toml · mlx-multi-server.toml · thermal-gate-script.sh · heartbeat-signals/
docs/            tier-routing.md
tests/           mock-server test suite
```

## License

MIT, © 2026 Robert Lingoes. See [LICENSE](LICENSE).
