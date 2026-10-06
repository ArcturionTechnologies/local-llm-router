# Tier routing

How the default tier table is meant to be used, and why. Everything here is a *default*;
tiers, roles and thresholds are all overridable in `config.toml` (see the README).

## Principle

**Lowest sufficient tier. Escalate only on signal** (complexity, stakes, a prior tier
failing, or an answer that was not good enough). A frontier model reasoning over a
five-label classification is slow, burns context and starves the work that needs it.

## Default tier table

| Tier | Where | Typical latency | Cost | Good for |
|---|---|---|---|---|
| `fast` | local, Qwen3-4B | ~200 ms | free | snap classify / route, short summaries |
| `small` | local, Qwen3-8B | ~500 ms | free | label sets up to ~8, JSON extraction, yes/no, simple summaries |
| `medium` | local, Qwen3-14B | 1-2 s | free | reasoning chains, draft outlines, review passes |
| `coder` | local, Qwen2.5-Coder-14B | 1-2 s | free | routine code generation / refactors (RAM-light) |
| `heavy` | local, Qwen3-30B-A3B MoE | 2-4 s | free | explicit deep-local reasoning; ~18 GB resident, exclusive |
| `vision` | local, Qwen2.5-VL-7B | ~1 s | free | screenshots, OCR, image questions |
| `groq` | cloud (free tier) | ~250 ms | free | mid-size summaries, bulk classify, fast fallback |
| `cerebras` | cloud (free tier) | ~600 ms | free | bulk jobs, larger summaries, agentic loops |
| `gemini` | cloud (free tier) | ~1 s | free | whole-repo / full-log reads, > 100k-token contexts |
| `haiku` | cloud, paid | ~1 s | $ | fast frontier-quality writes and edits |
| `sonnet` | cloud, paid | ~3 s | $$ | the workhorse: high-stakes code, complex reasoning |
| `opus` | cloud, paid | ~5 s | $$$ | irreversible or expensive decisions, final review |

Latencies are rough, observed on an M-series Mac with 4-bit MLX models. Prices live in the
config and are estimates.

## Decision tree (what `pick_tier()` does)

```
Task arrives.
├─ stakes = irreversible? ─────────────────────────────→ strongest tier (opus). Stop.
├─ context > long_context_tokens (100k)? ──────────────→ long-context tier (gemini).
├─ bulk loop (> bulk_threshold items, default 20)? ────→ never paid:
│     trivial/simple → groq        moderate/complex → cerebras
├─ vision / ocr / screenshot ──────────────────────────→ vision
├─ classify / route, trivial-simple ───────────────────→ fast
├─ extract trivial-simple, summarize simple ───────────→ small
├─ code-gen / edit / refactor ─────────────────────────→ coder   (stakes=high → sonnet)
├─ reason / analyze ───────────────────────────────────→ small if simple,
│                                                        medium otherwise (stakes=high → sonnet)
├─ review ─────────────────────────────────────────────→ medium  (stakes=high → opus)
├─ complex ────────────────────────────────────────────→ medium  (stakes=high → opus)
└─ everything else ────────────────────────────────────→ fast if stakes=low, else sonnet
```

Hard rules are checked first and in that order, so a bulk irreversible job still goes to the
strongest tier. Boundaries are strict: exactly 20 items is *not* bulk, exactly 100,000 tokens
is *not* long-context (both pinned by tests).

Signature vocabulary: `task_class` is one of `classify | extract | summarize | route | reason |
code-gen | edit | refactor | analyze | review | vision | other`; `complexity` is `trivial | simple |
moderate | complex`; `stakes` is `low | medium | high | irreversible`.

## Fallthrough

Each tier lists `fallbacks`. When a tier cannot serve a request the router moves to the next one
in the chain and records why:

| Attempt result | Meaning |
|---|---|
| `ThermalBlocked` | host red, or profile says no at yellow / unknown sensors |
| `ThermalDowngrade` | a lighter tier was requested (medium at yellow, RAM pressure on heavy) |
| `TierBusy` | model lock held by another process / server answered `local-pool-busy` |
| `RateLimited` | provider answered 429/402; that tier is skipped for 10 minutes |
| `MissingCredentials` | no API key in the tier's env var |
| `BudgetExceeded` | `budget.enforce` is on and the envelope is spent |
| `TierError` | server unreachable, HTTP error, malformed reply |
| `rejected` | the caller's `accept()` refused the answer (this is escalation by quality) |

If the whole chain is exhausted `AllTiersFailed` carries the full attempt trail. Callers that
would rather degrade than raise pass `fallback="skip"`.

## Running next to LM Studio

LM Studio and `mlx_lm.server` both load weights into the same unified memory. Two copies of an
18 GB MoE will swap a 48 GB machine to a standstill, so:

1. `llm-router lock status` before loading a big model in LM Studio. If a tier you marked
   `exclusive` holds the lock, wait for the call to finish.
2. The lock only covers calls routed through this package (and servers that adopt `ModelLock`);
   it cannot see a model you loaded by hand. Unload it before routing heavy work.
3. Small models (4B-8B, about 2-5 GB) are fine to keep resident next to anything.
4. For a single-server setup prefer `examples/lmstudio.toml`: one tier, thermal-gated, with cloud
   fallbacks.

## Skip the router entirely when

* the work involves money, production deploys, external sends or anything irreversible: call
  the strongest tier directly and say why;
* the content must not leave the machine: use `fallback="raise"` on a local tier so it can never
  escalate to a cloud provider;
* a human explicitly asked for a specific model.
