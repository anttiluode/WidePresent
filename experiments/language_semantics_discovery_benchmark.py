"""Language-agent temporal-semantics discovery benchmark.

Unlike language_tool_validity_benchmark.py, this track does NOT tell the model
which temporal coordinate governs each source. Source names are arbitrary:
alpha, beta, gamma.

The model receives audited past episodes and must infer the validity semantics.

Experience conditions
---------------------
narrow
    alpha/beta examples occur at ~1 second/message. World age and event distance
    are confounded, so seconds-vs-events is not identifiable from the examples.

wide
    audit examples deliberately decouple elapsed seconds from message count.
    alpha becomes identifiable as wall-time-yoked and beta as structure-yoked.

Within either experience condition, each final case has three paired prompt
conditions compatible with language_tool_validity_benchmark.py run-openai:

raw
    timestamped audit transcripts + timestamped final history.

age_plane
    same audits/final history plus deterministic ages/counts. No semantic rule or
    action recommendation.

resolver
    strongest boring attacker: the true source validity contract is supplied
    explicitly, but the final action is still left to the model.

The final hidden cases are identical for --experience narrow and wide when seed
and generation arguments match.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

SOURCES = ("alpha", "beta", "gamma")
CONDS = ("raw", "age_plane", "resolver")
TTL_S = 8.0
TTL_N = 8
R_GOOD = 1.0
R_BAD = -1.5
R_REFRESH = 0.55

REGIMES = {
    "iid": (0.95, 1.05, 1.0),
    "dense": (0.25, 0.45, 1.0),
    "sparse": (1.80, 2.40, 1.0),
    "long_delay": (0.95, 1.05, 6.0),
    "dense_long": (0.25, 0.45, 6.0),
}

VALUES = {
    "alpha": ("blue-17", "amber-42", "silver-09", "green-31"),
    "beta": ("Project Alder", "Project Birch", "Project Cedar", "Project Delta"),
    "gamma": ("HELD", "CONFIRMED", "PAID", "CHECKED-IN"),
}

TOOL = {
    "type": "function",
    "name": "refresh_source",
    "description": "Fetch a fresh current value for an arbitrary cached source when you infer the cache is no longer valid.",
    "parameters": {
        "type": "object",
        "properties": {
            "source": {"type": "string", "enum": list(SOURCES)},
            "key": {"type": "string"},
        },
        "required": ["source", "key"],
        "additionalProperties": False,
    },
    "strict": True,
}

DISCOVERY_INTRO = """You are a cache-aware tool agent. The sources alpha, beta, and gamma have different stable validity semantics, but their rules are NOT stated. Infer those semantics from the audited past episodes below.

Each audit shows a cached source record, intervening neutral conversation messages, a decision point, and whether the cached record was still valid at that decision. Source labels are arbitrary; do not assume semantics from their names.

For the final user request:
- if the cached evidence is still valid under the pattern supported by the audits, answer directly from the cached value and do not call the tool;
- if it is no longer valid, call refresh_source for the correct source/key and do not guess.

The final user request itself is not an intervening message."""

ORACLE_CONTRACT = """The source semantics are:
- ALPHA: valid through 8.0 seconds after observed_at. Use observed_at, not received_at.
- BETA: valid through 8 intervening conversation messages after the cache record was received. Wall-clock seconds do not expire it.
- GAMMA: valid until an explicit source-change notification after the cache; age alone does not expire it.
"""


def iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def readjl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def writejl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def validity(source: str, world_age: float, n_messages: int, invalidation: bool) -> bool:
    if source == "alpha":
        return world_age <= TTL_S + 1e-9
    if source == "beta":
        return n_messages <= TTL_N
    return not invalidation


def reward(action: str, oracle: str) -> float:
    if action == "refresh":
        return R_REFRESH
    return R_GOOD if oracle == "reuse" else R_BAD


def audit_specs(experience: str) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if experience == "narrow":
        # alpha and beta are exactly confounded: world_age == event_count.
        core = [(4, 1.0), (8, 1.0), (9, 1.0), (12, 1.0)]
    elif experience == "wide":
        # Two examples explicitly oppose the seconds and event-count rules.
        core = [(12, 0.30), (4, 2.50), (6, 1.0), (10, 1.0)]
    else:
        raise ValueError(experience)

    for source in ("alpha", "beta"):
        for n, gap in core:
            specs.append(
                {
                    "source": source,
                    "n_messages": n,
                    "gap": gap,
                    "observed_delay": 0.0,
                    "invalidation": False,
                }
            )

    gamma = [
        (3, 0.4, False),
        (12, 1.8, True),
        (14, 0.3, False),
        (4, 2.4, True),
    ]
    for n, gap, invalid in gamma:
        specs.append(
            {
                "source": "gamma",
                "n_messages": n,
                "gap": gap,
                "observed_delay": 1.0,
                "invalidation": invalid,
            }
        )
    return specs


def render_audit(spec: dict[str, Any], index: int, *, include_plane: bool) -> str:
    source = spec["source"]
    n = spec["n_messages"]
    gap = spec["gap"]
    invalid = spec["invalidation"]

    received = datetime(2026, 1, 3, 9, 0, tzinfo=timezone.utc) + timedelta(minutes=10 * index)
    observed = received - timedelta(seconds=spec["observed_delay"])
    decision = received + timedelta(seconds=n * gap)
    world_age = (decision - observed).total_seconds()
    is_valid = validity(source, world_age, n, invalid)

    lines = [
        f"AUDIT {index + 1} — source={source}",
        f"[{iso(received)}] cached record received; observed_at={iso(observed)}.",
    ]
    for j in range(n):
        t = received + timedelta(seconds=(j + 1) * gap)
        if source == "gamma" and invalid and j == max(0, n // 2 - 1):
            lines.append(
                f"[{iso(t)}] SOURCE-CHANGE notification: gamma changed after the cached record."
            )
        else:
            lines.append(f"[{iso(t)}] neutral conversation message {j + 1}.")
    lines.append(f"[{iso(decision)}] audit decision point.")
    if include_plane:
        lines.append(
            "DERIVED TEMPORAL STATE: "
            f"world_age={world_age:.3f}s; arrival_age={n * gap:.3f}s; "
            f"intervening_messages={n}; explicit_invalidation={'yes' if invalid else 'no'}."
        )
    lines.append(
        "AUDIT RESULT: cached record was "
        + ("STILL VALID." if is_valid else "STALE.")
    )
    return "\n".join(lines)


def candidate_fit(experience: str) -> dict[str, dict[str, float]]:
    specs = audit_specs(experience)
    out: dict[str, dict[str, float]] = {}
    for source in SOURCES:
        subset = [s for s in specs if s["source"] == source]
        truth = []
        pred_seconds = []
        pred_events = []
        pred_invalidation = []
        for s in subset:
            world_age = s["observed_delay"] + s["n_messages"] * s["gap"]
            truth.append(validity(source, world_age, s["n_messages"], s["invalidation"]))
            pred_seconds.append(world_age <= TTL_S + 1e-9)
            pred_events.append(s["n_messages"] <= TTL_N)
            pred_invalidation.append(not s["invalidation"])
        out[source] = {
            "seconds": float(np.mean(np.asarray(pred_seconds) == np.asarray(truth))),
            "events": float(np.mean(np.asarray(pred_events) == np.asarray(truth))),
            "invalidation": float(np.mean(np.asarray(pred_invalidation) == np.asarray(truth))),
        }
    return out


def audit_block(experience: str, *, include_plane: bool) -> str:
    rendered = [
        render_audit(spec, i, include_plane=include_plane)
        for i, spec in enumerate(audit_specs(experience))
    ]
    return "\n\n".join(rendered)


def final_payload(source: str, index: int, rng: random.Random) -> tuple[str, str, str]:
    value = rng.choice(VALUES[source])
    if source == "alpha":
        key = f"A-{100 + index % 900:03d}"
        final = f"What is the current value for alpha key {key}? Reuse the cache only if it is still valid."
    elif source == "beta":
        key = f"B-{100 + index % 900:03d}"
        final = f"What does beta key {key} currently resolve to? Reuse the cache only if it is still valid."
    else:
        key = f"G-{100 + index % 900:03d}"
        final = f"What is the current state for gamma key {key}? Reuse the cache only if it is still valid."
    return key, value, final


def make_final_case(seed: int, index: int, regime: str) -> dict[str, Any]:
    # Crucially independent of experience mode.
    ridx = list(REGIMES).index(regime)
    rng = np.random.default_rng(seed + 100003 * index + 71 * ridx)
    pr = random.Random(seed + 900001 * index + 53 * ridx)

    source = SOURCES[index % len(SOURCES)]
    key, value, final = final_payload(source, index, pr)

    n = int(rng.integers(0, 15))
    lo, hi, delay_scale = REGIMES[regime]
    gap = float(rng.uniform(lo, hi))
    arrival_age = (n + 1) * gap
    delay = min(float(rng.exponential(delay_scale)), 12.0)

    received = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc) + timedelta(
        days=index // 120, minutes=3 * (index % 120)
    )
    observed = received - timedelta(seconds=delay)
    decision = received + timedelta(seconds=arrival_age)

    invalidation = False
    invalidation_slot: int | None = None
    if source == "gamma" and n:
        invalidation = bool(rng.random() < 0.40)
        if invalidation:
            invalidation_slot = int(rng.integers(n))

    history = [
        {
            "role": "assistant",
            "time": iso(received),
            "content": (
                f"Cached source={source} key={key} value={value}. "
                f"observed_at={iso(observed)}; received_at={iso(received)}."
            ),
        }
    ]
    for j in range(n):
        t = received + timedelta(seconds=(j + 1) * gap)
        role = "user" if j % 2 == 0 else "assistant"
        if source == "gamma" and invalidation and invalidation_slot == j:
            content = (
                f"SOURCE-CHANGE notification: gamma key {key} changed after the cached record. "
                "The cached value is no longer current."
            )
        else:
            content = (
                f"Neutral conversation message {j + 1}; marker {int(rng.integers(100,999))}. "
                "No cached source is updated."
            )
        history.append({"role": role, "time": iso(t), "content": content})

    history.append({"role": "user", "time": iso(decision), "content": final})

    world_age = (decision - observed).total_seconds()
    valid = validity(source, world_age, n, invalidation)

    return {
        "case_id": f"{regime}-{index:05d}",
        "base_case_id": f"{regime}-{index:05d}",
        "regime": regime,
        "source": source,
        "key": key,
        "cached_value": value,
        "observed_at": iso(observed),
        "received_at": iso(received),
        "decision_at": iso(decision),
        "valid_age_seconds": world_age,
        "arrival_age_seconds": arrival_age,
        "intervening_messages": n,
        "explicit_invalidation": invalidation,
        "oracle_action": "reuse" if valid else "refresh",
        "history": history,
    }


def temporal_extra(case: dict[str, Any]) -> str:
    return (
        "\n\nFINAL TEMPORAL RUNTIME STATE (arithmetic only; no semantic rule or recommendation):\n"
        f"- source: {case['source']}\n"
        f"- world age: {case['valid_age_seconds']:.3f} seconds\n"
        f"- arrival age: {case['arrival_age_seconds']:.3f} seconds\n"
        f"- intervening conversation messages: {case['intervening_messages']}\n"
        f"- explicit invalidation: {'yes' if case['explicit_invalidation'] else 'no'}"
    )


def system_prompt(experience: str, condition: str, case: dict[str, Any]) -> str:
    if condition == "raw":
        return (
            DISCOVERY_INTRO
            + "\n\nPAST AUDITS:\n\n"
            + audit_block(experience, include_plane=False)
        )
    if condition == "age_plane":
        return (
            DISCOVERY_INTRO
            + "\n\nPAST AUDITS WITH DETERMINISTIC TEMPORAL ARITHMETIC:\n\n"
            + audit_block(experience, include_plane=True)
            + temporal_extra(case)
        )
    if condition == "resolver":
        return (
            DISCOVERY_INTRO
            + "\n\nPAST AUDITS:\n\n"
            + audit_block(experience, include_plane=False)
            + "\n\nEXTERNALLY RESOLVED VALIDITY CONTRACT:\n"
            + ORACLE_CONTRACT
        )
    raise ValueError(condition)


def render(case: dict[str, Any], experience: str, condition: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt(experience, condition, case)}]
    messages.extend(
        {"role": m["role"], "content": f"[{m['time']}] {m['content']}"}
        for m in case["history"]
    )
    return messages


def generate(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    index = 0
    for regime in REGIMES:
        for _ in range(args.per_regime):
            case = make_final_case(args.seed, index, regime)
            case["experience"] = args.experience
            case["case_id"] = f"{args.experience}::{case['base_case_id']}"
            case["conditions"] = {
                c: {"messages": render(case, args.experience, c), "tools": [TOOL]}
                for c in CONDS
            }
            rows.append(case)
            index += 1

    writejl(args.output, rows)
    print(
        f"wrote {len(rows)} {args.experience}-experience cases; "
        f"{len(rows) * len(CONDS)} model decisions"
    )


def sanity(args: argparse.Namespace) -> None:
    fits = candidate_fit(args.experience)
    print(f"experience={args.experience}")
    for source in SOURCES:
        s = fits[source]
        print(
            f"  {source}: seconds={s['seconds']:.3f} "
            f"events={s['events']:.3f} invalidation={s['invalidation']:.3f}"
        )

    if args.experience == "narrow":
        assert fits["alpha"]["seconds"] == 1.0
        assert fits["alpha"]["events"] == 1.0
        assert fits["beta"]["seconds"] == 1.0
        assert fits["beta"]["events"] == 1.0
        print("  alpha/beta seconds-vs-events ambiguity assertion passed")
    else:
        assert fits["alpha"]["seconds"] == 1.0
        assert fits["alpha"]["events"] < 1.0
        assert fits["beta"]["events"] == 1.0
        assert fits["beta"]["seconds"] < 1.0
        assert fits["gamma"]["invalidation"] == 1.0
        print("  wide-experience semantic identifiability assertions passed")

    if args.input:
        rows = readjl(args.input)
        assert all(r["experience"] == args.experience for r in rows)
        for r in rows:
            tail = r["conditions"]["raw"]["messages"][1:]
            assert r["conditions"]["age_plane"]["messages"][1:] == tail
            assert r["conditions"]["resolver"]["messages"][1:] == tail
            assert (
                r["conditions"]["raw"]["tools"]
                == r["conditions"]["age_plane"]["tools"]
                == r["conditions"]["resolver"]["tools"]
            )
        print(f"  pairing checks passed for {len(rows)} generated cases")


def compare(args: argparse.Namespace) -> None:
    """Compare two generated case files and assert identical final hidden cases."""
    left = {r["base_case_id"]: r for r in readjl(args.left)}
    right = {r["base_case_id"]: r for r in readjl(args.right)}
    assert left.keys() == right.keys()

    fields = (
        "regime",
        "source",
        "key",
        "cached_value",
        "observed_at",
        "received_at",
        "decision_at",
        "valid_age_seconds",
        "arrival_age_seconds",
        "intervening_messages",
        "explicit_invalidation",
        "oracle_action",
        "history",
    )
    for key in left:
        for field in fields:
            assert left[key][field] == right[key][field], (key, field)
    print(f"final-case identity passed for {len(left)} narrow/wide matched cases")


def compare_responses(args: argparse.Namespace) -> None:
    narrow_cases = readjl(args.narrow_cases)
    wide_cases = readjl(args.wide_cases)
    narrow_by_id = {c["case_id"]: c for c in narrow_cases}
    wide_by_id = {c["case_id"]: c for c in wide_cases}
    narrow_base = {c["base_case_id"]: c for c in narrow_cases}
    wide_base = {c["base_case_id"]: c for c in wide_cases}
    assert narrow_base.keys() == wide_base.keys()

    narrow_resp = readjl(args.narrow_responses)
    wide_resp = readjl(args.wide_responses)

    nr: dict[tuple[str, str], dict[str, Any]] = {}
    wr: dict[tuple[str, str], dict[str, Any]] = {}
    for row in narrow_resp:
        case = narrow_by_id.get(row.get("case_id"))
        if case:
            nr[(case["base_case_id"], row["condition"])] = row
    for row in wide_resp:
        case = wide_by_id.get(row.get("case_id"))
        if case:
            wr[(case["base_case_id"], row["condition"])] = row

    for condition in CONDS:
        records = []
        for base_id in sorted(narrow_base):
            key = (base_id, condition)
            if key not in nr or key not in wr:
                continue
            nrow, wrow = nr[key], wr[key]
            na, wa = nrow.get("action"), wrow.get("action")
            if na not in ("reuse", "refresh") or wa not in ("reuse", "refresh"):
                continue
            oracle = narrow_base[base_id]["oracle_action"]
            assert oracle == wide_base[base_id]["oracle_action"]
            records.append(
                {
                    "base_id": base_id,
                    "source": narrow_base[base_id]["source"],
                    "regime": narrow_base[base_id]["regime"],
                    "oracle": oracle,
                    "narrow": na,
                    "wide": wa,
                    "n_ok": na == oracle,
                    "w_ok": wa == oracle,
                }
            )

        if not records:
            continue

        n_acc = float(np.mean([r["n_ok"] for r in records]))
        w_acc = float(np.mean([r["w_ok"] for r in records]))
        n_util = float(np.mean([reward(r["narrow"], r["oracle"]) for r in records]))
        w_util = float(np.mean([reward(r["wide"], r["oracle"]) for r in records]))
        helpful = sum((not r["n_ok"]) and r["w_ok"] for r in records)
        harmful = sum(r["n_ok"] and (not r["w_ok"]) for r in records)
        changed = sum(r["narrow"] != r["wide"] for r in records)

        print(f"\n{condition}: matched={len(records)}")
        print(
            f"  narrow_accuracy={n_acc:.3f} wide_accuracy={w_acc:.3f} "
            f"delta={w_acc - n_acc:+.3f}"
        )
        print(
            f"  narrow_utility={n_util:.3f} wide_utility={w_util:.3f} "
            f"delta={w_util - n_util:+.3f}"
        )
        print(
            f"  action_changed={changed} helpful_flips={helpful} harmful_flips={harmful}"
        )

        for source in SOURCES:
            z = [r for r in records if r["source"] == source]
            if z:
                print(
                    f"    source={source}: narrow={np.mean([r['n_ok'] for r in z]):.3f} "
                    f"wide={np.mean([r['w_ok'] for r in z]):.3f} "
                    f"helpful={sum((not r['n_ok']) and r['w_ok'] for r in z)} "
                    f"harmful={sum(r['n_ok'] and (not r['w_ok']) for r in z)}"
                )
        for regime in REGIMES:
            z = [r for r in records if r["regime"] == regime]
            if z:
                print(
                    f"    regime={regime:>10s}: narrow={np.mean([r['n_ok'] for r in z]):.3f} "
                    f"wide={np.mean([r['w_ok'] for r in z]):.3f}"
                )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--output", type=Path, required=True)
    g.add_argument("--experience", choices=("narrow", "wide"), required=True)
    g.add_argument("--per-regime", type=int, default=30)
    g.add_argument("--seed", type=int, default=777)
    g.set_defaults(func=generate)

    s = sub.add_parser("sanity")
    s.add_argument("--experience", choices=("narrow", "wide"), required=True)
    s.add_argument("--input", type=Path)
    s.set_defaults(func=sanity)

    c = sub.add_parser("compare")
    c.add_argument("--left", type=Path, required=True)
    c.add_argument("--right", type=Path, required=True)
    c.set_defaults(func=compare)

    cr = sub.add_parser("compare-responses")
    cr.add_argument("--narrow-cases", type=Path, required=True)
    cr.add_argument("--narrow-responses", type=Path, required=True)
    cr.add_argument("--wide-cases", type=Path, required=True)
    cr.add_argument("--wide-responses", type=Path, required=True)
    cr.set_defaults(func=compare_responses)

    return p


if __name__ == "__main__":
    args = parser().parse_args()
    args.func(args)
