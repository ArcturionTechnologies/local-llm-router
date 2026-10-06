"""Command line: ``llm-router <command>`` or ``python -m local_llm_router <command>``.

Exit codes: 0 success, 1 bad arguments, 2 the router could not produce an
answer (so shell callers can decide whether to escalate themselves). The
``thermal``, ``guard``, ``heartbeat should-fire`` and ``local-first`` gates use
their own 0/1/2 conventions, documented in ``--help``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Optional, Sequence

from . import __version__
from .budget import Budget
from .config import dump_toml, find_config_file, home_dir, load_config
from .errors import ConfigError, RouterError
from .heartbeat import HeartbeatGate
from .local_first import LocalFirstGate
from .model_lock import ModelLock
from .rate_guard import RateGuard
from .router import Router, pick_tier
from .thermal import ThermalGate, probe


def _p() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="llm-router",
        description="Local-first LLM router for Apple Silicon: local tiers first, "
                    "escalate when needed, back off when the Mac runs hot.")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    ap.add_argument("--config", help="path to config.toml / config.json (default: $LLM_ROUTER_CONFIG or "
                                     "~/.config/local-llm-router/config.toml)")
    sub = ap.add_subparsers(dest="cmd", metavar="<command>")

    c = sub.add_parser("chat", help="single-turn chat through the tier chain")
    c.add_argument("prompt")
    c.add_argument("--tier")
    c.add_argument("--system")
    c.add_argument("--max-tokens", type=int, default=400)
    c.add_argument("--fallback", choices=["cascade", "raise", "skip"], default="cascade")
    c.add_argument("-v", "--verbose", action="store_true", help="print which tier answered")

    c = sub.add_parser("classify", help="pick one label: classify TEXT label_a,label_b,...")
    c.add_argument("text"); c.add_argument("labels")
    c = sub.add_parser("summarize", help="summarize TEXT [MAX_WORDS]")
    c.add_argument("text"); c.add_argument("max_words", nargs="?", type=int, default=40)
    c = sub.add_parser("extract", help="extract JSON: extract TEXT '{\"field\": \"type\"}'")
    c.add_argument("text"); c.add_argument("schema")
    c = sub.add_parser("route", help="route TEXT option_a,option_b,...")
    c.add_argument("text"); c.add_argument("options")
    c = sub.add_parser("escalate", help="should this go to a bigger tier? prints YES or NO")
    c.add_argument("prompt")
    c = sub.add_parser("tier-call", help="complexity-based escalation (trivial/routine/hard/critical/auto)")
    c.add_argument("prompt")
    c.add_argument("--complexity", default="auto",
                   choices=["trivial", "routine", "hard", "critical", "auto"])
    c.add_argument("-v", "--verbose", action="store_true")

    c = sub.add_parser("pick", help="which tier would this task use?")
    c.add_argument("--task-class", default="other"); c.add_argument("--complexity", default="simple")
    c.add_argument("--stakes", default="low"); c.add_argument("--bulk", type=int, default=1)
    c.add_argument("--context-tokens", type=int, default=0)

    sub.add_parser("tiers", help="list configured tiers and their fallback chains")
    sub.add_parser("health", help="probe every tier (local: GET /models; cloud: key present)")

    c = sub.add_parser("thermal", help="thermal gate verdict; exit 0 go / 1 lighter tier / 2 blocked")
    c.add_argument("--profile", default="standard",
                   choices=["none", "light", "standard", "medium", "coder", "vision", "heavy"])
    c.add_argument("--json", action="store_true")

    c = sub.add_parser("config", help="init | show | path")
    c.add_argument("action", choices=["init", "show", "path"])
    c.add_argument("--force", action="store_true")

    c = sub.add_parser("budget", help="status | snapshot | displacement | record TIER TOKENS_IN TOKENS_OUT")
    c.add_argument("action", choices=["status", "snapshot", "displacement", "record"])
    c.add_argument("args", nargs="*")

    c = sub.add_parser("lock", help="model lock: status | force-release")
    c.add_argument("action", choices=["status", "force-release"])

    c = sub.add_parser("guard", help="session rate guard: check (exit 0 ok/1 throttle/2 block) | log-spawn CALLER TASK")
    c.add_argument("action", choices=["check", "log-spawn"])
    c.add_argument("args", nargs="*")

    c = sub.add_parser("heartbeat", help="should-fire ID (exit 0 fire/1 skip) | status ID | reset ID | discover")
    c.add_argument("action", choices=["should-fire", "status", "reset", "discover"])
    c.add_argument("id", nargs="?")

    c = sub.add_parser("local-first", help="policy gate: status | test | clear | hook")
    c.add_argument("action", choices=["status", "test", "clear", "hook"])
    return ap


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = _p()
    args = ap.parse_args(argv)
    if not args.cmd:
        ap.print_help()
        return 0
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        _err(f"[config-error] {e}")
        return 1
    try:
        return _dispatch(args, cfg)
    except RouterError as e:
        _err(f"[error] {e}")
        return 2
    except (IndexError, ValueError, json.JSONDecodeError) as e:
        _err(f"[bad-args] {e}")
        return 1


def _dispatch(a: argparse.Namespace, cfg) -> int:
    cmd = a.cmd

    if cmd == "config":
        if a.action == "path":
            print(find_config_file() or (home_dir() / "config.toml"))
        elif a.action == "show":
            print(json.dumps(cfg.to_dict(), indent=2))
        else:
            target = home_dir() / "config.toml"
            if target.exists() and not a.force:
                _err(f"{target} already exists (use --force to overwrite)")
                return 1
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("# local-llm-router -- effective defaults. Edit freely; unknown keys are rejected.\n"
                              + dump_toml(cfg.to_dict()))
            print(f"wrote {target}")
        return 0

    if cmd == "thermal":
        gate = ThermalGate(cfg.thermal)
        d = gate.check(a.profile)
        if a.json:
            print(json.dumps({"profile": a.profile, **asdict(d), "snapshot": asdict(probe())}, indent=2))
        else:
            print(f"{d.action.upper()} ({d.level}): {d.reason}")
        return d.exit_code

    if cmd == "pick":
        tier, why = pick_tier(cfg, a.task_class, a.complexity, a.stakes, a.bulk, a.context_tokens)
        print(f"{tier}\t{why}")
        return 0

    if cmd == "tiers":
        for t in cfg.tiers.values():
            chain = " -> ".join([t.name, *t.fallbacks]) if t.fallbacks else t.name
            flag = "" if t.enabled else " (disabled)"
            print(f"{t.name:10s} {t.kind:5s} {t.cost_class:5s} {t.profile:8s} {t.model}{flag}\n"
                  f"{'':10s} {t.base_url}   chain: {chain}")
        return 0

    if cmd == "budget":
        b = Budget(cfg)
        if a.action == "status":
            b.status()
        elif a.action == "snapshot":
            b.save(b.load()); print(f"snapshot written to {b.path}")
        elif a.action == "displacement":
            b.displacement_report()
        else:
            tier, tin, tout = a.args[0], int(a.args[1]), int(a.args[2])
            b.record(tier, tin, tout)
            b.status()
        return 0

    if cmd == "lock":
        lock = ModelLock(cfg.lock_dir)
        if a.action == "status":
            h = lock.holder()
            print(json.dumps(h, indent=2) if h else "(lock not held)")
        else:
            lock.force_release()
            print("meta + lock files cleared")
        return 0

    if cmd == "guard":
        g = RateGuard(cfg)
        if a.action == "check":
            rec = g.check()
            print(json.dumps(rec, indent=2))
            return g.exit_code(rec)
        g.log_spawn(a.args[0], a.args[1])
        print("ok")
        return 0

    if cmd == "heartbeat":
        hb = HeartbeatGate(cfg)
        if a.action == "discover":
            for it in hb.discover():
                print(f"  {'ok     ' if it['has_manifest'] else 'MISSING'} {it['hbid']}")
            return 0
        if not a.id:
            raise ValueError("heartbeat id required")
        if a.action == "should-fire":
            return 0 if hb.should_fire(a.id) else 1
        if a.action == "status":
            print(json.dumps(hb.status(a.id), indent=2, default=str))
        else:
            hb.reset(a.id); print("ok")
        return 0

    if cmd == "local-first":
        gate = LocalFirstGate(cfg)
        if a.action == "hook":
            gate.hook()
        elif a.action == "status":
            print(gate.status())
        elif a.action == "clear":
            gate.clear(); print("violations cleared")
        else:
            for p, tc, cx, sk in [("local-qwen3-8b", "summarize", "simple", "low"),
                                  ("sonnet", "classify", "simple", "low"),
                                  ("sonnet", "code-gen", "complex", "low"),
                                  ("opus", "review", "complex", "irreversible"),
                                  ("haiku", "summarize", "trivial", "low")]:
                d, r = gate.gate(p, tc, cx, sk)
                print(f"  {p:16s} {tc:11s} {cx:8s} {sk:13s} -> {d:5s} | {r}")
        return 0

    # -- commands that call a model ----------------------------------------
    router = Router(cfg)
    if cmd == "health":
        results = router.health()
        for name, ok in results.items():
            print(f"{'UP  ' if ok else 'DOWN'} {name}")
        return 0 if any(results.values()) else 2
    if cmd == "chat":
        comp = router.complete(a.prompt, system=a.system, tier=a.tier, max_tokens=a.max_tokens,
                               fallback=a.fallback)
        print(comp.text)
        if a.verbose:
            _err(f"[tier={comp.tier} model={comp.model} {comp.latency_ms}ms "
                 f"attempts={[(x.tier, x.error or 'ok') for x in comp.attempts]}]")
        return 0
    if cmd == "classify":
        print(router.classify(a.text, [x.strip() for x in a.labels.split(",") if x.strip()]))
    elif cmd == "summarize":
        print(router.summarize(a.text, a.max_words))
    elif cmd == "extract":
        print(json.dumps(router.extract_json(a.text, json.loads(a.schema)), indent=2))
    elif cmd == "route":
        print(router.route(a.text, [x.strip() for x in a.options.split(",") if x.strip()]))
    elif cmd == "escalate":
        print("YES" if router.should_escalate(a.prompt) else "NO")
    elif cmd == "tier-call":
        comp = router.tier_call(a.prompt, complexity=a.complexity)
        print(comp.text)
        if a.verbose:
            _err(f"[tier={comp.tier} model={comp.model} {comp.latency_ms}ms]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
