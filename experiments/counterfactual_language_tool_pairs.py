"""Counterfactual minimal-pair attack for the language/tool validity benchmark.

This suite asks a stronger question than aggregate accuracy:

    Does the agent change its reuse/refresh decision when the *causal* temporal
    coordinate crosses the validity boundary, and remain invariant when a
    non-causal temporal coordinate changes?

The generated JSONL is compatible with:
    language_tool_validity_benchmark.py run-openai

Then score paired responses with this script.

Pair families
-------------
weather_flip
    Same message content/count/arrival age; only observed_at changes so world
    age crosses 8 seconds.

weather_invariant
    Same world age (both valid or both stale); intervening message count changes
    sharply. Action must stay the same.

discourse_flip
    Nearly identical wall time; intervening conversation messages cross 8 -> 9.

discourse_invariant
    Same event distance; elapsed wall time changes sharply. Action must stay the
    same.

reservation_flip
    Same time/count; one neutral message is replaced by an explicit state-change
    notification.

reservation_invariant
    Same invalidation state; wall time and event count change sharply. Action
    must stay the same.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

SOURCES = ("weather", "discourse", "reservation")
CONDS = ("raw", "age_plane", "resolver")
TTL_S = 8.0
TTL_N = 8
R_GOOD = 1.0
R_BAD = -1.5
R_REFRESH = 0.55

PLACES = ("Northport", "Lakeview", "Riverton", "Hillcrest", "Westhaven")
WEATHER = ("light rain, 9 C", "clear, 14 C", "overcast, 11 C", "windy, 8 C", "drizzle, 7 C")
PROJECTS = ("Project Alder", "Project Birch", "Project Cedar", "Project Delta", "Project Elm")
STATES = ("CONFIRMED", "HELD", "PAID", "CHECKED-IN")

TOOL = {
    "type": "function",
    "name": "refresh_source",
    "description": "Fetch a fresh current value only when the cached evidence is invalid under the contract.",
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

CONTRACT = """You are a cache-aware tool agent. For the final user request, either answer directly from the cached evidence or call refresh_source. Follow this validity contract exactly:
- WEATHER: valid through 8.0 seconds after observed_at. Use observed_at, not received_at.
- DISCOURSE: valid through 8 intervening conversation messages after the cache record was received. Wall-clock seconds do not expire it.
- RESERVATION: does not expire with time/message count; invalid only after an explicit reservation-change notification after the cache.
If valid, answer directly and do not call the tool. If invalid, call refresh_source and do not guess. The final user message is the decision point and is not an intervening message."""


def iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def writejl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def readjl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def reward(action: str, oracle: str) -> float:
    if action == "refresh":
        return R_REFRESH
    return R_GOOD if oracle == "reuse" else R_BAD


def source_payload(source: str, pair_index: int, rng: random.Random) -> tuple[str, str, str]:
    if source == "weather":
        key = rng.choice(PLACES)
        value = rng.choice(WEATHER)
        final = f"What is the weather in {key} now? Use cache if valid; otherwise refresh it."
    elif source == "discourse":
        key = "current_draft_reference"
        value = rng.choice(PROJECTS)
        final = "What does 'the draft' refer to now? Use the cached resolution if valid; otherwise refresh it."
    else:
        key = f"R-{2000 + pair_index % 7000:04d}"
        value = rng.choice(STATES)
        final = f"What is the current status of reservation {key}? Use cache if valid; otherwise refresh it."
    return key, value, final


def cache_text(source: str, key: str, value: str, observed: datetime, received: datetime) -> str:
    if source == "weather":
        return (
            f"Cached weather record for {key}: {value}. "
            f"observed_at={iso(observed)}; received_at={iso(received)}."
        )
    if source == "discourse":
        return (
            f"Cached conversation-focus resolution: 'the draft' refers to {value}. "
            f"observed_at={iso(observed)}; received_at={iso(received)}."
        )
    return (
        f"Cached reservation record: {key} status is {value}. "
        f"observed_at={iso(observed)}; received_at={iso(received)}."
    )


def extra(case: dict[str, Any], condition: str) -> str:
    if condition == "raw":
        return ""
    if condition == "age_plane":
        return (
            "\n\nTEMPORAL RUNTIME STATE (arithmetic only; no recommendation):\n"
            f"- source: {case['source']}\n"
            f"- current decision time: {case['decision_at']}\n"
            f"- valid/world age: {case['valid_age_seconds']:.3f} seconds\n"
            f"- arrival/knowledge age: {case['arrival_age_seconds']:.3f} seconds\n"
            f"- intervening conversation messages: {case['intervening_messages']}\n"
            f"- explicit invalidation after cache: {'yes' if case['explicit_invalidation'] else 'no'}\n"
            "Apply the contract yourself."
        )
    return (
        "\n\nDETERMINISTIC VALIDITY RESOLVER:\n"
        f"- source: {case['source']}\n"
        f"- cache_valid_under_contract: {'yes' if case['oracle_action'] == 'reuse' else 'no'}\n"
        f"- recommended action: {case['oracle_action'].upper()}\n"
        "The resolver applied the contract mechanically."
    )


def render(case: dict[str, Any], condition: str) -> list[dict[str, str]]:
    out = [{"role": "system", "content": CONTRACT + extra(case, condition)}]
    out.extend(
        {"role": m["role"], "content": f"[{m['time']}] {m['content']}"}
        for m in case["history"]
    )
    return out


def build_case(
    *,
    pair_index: int,
    pair_family: str,
    variant: str,
    expected_relation: str,
    source: str,
    key: str,
    value: str,
    final: str,
    received: datetime,
    observed_delay: float,
    arrival_age: float,
    n_messages: int,
    invalidation: bool,
    invalidation_slot: int | None,
    marker_seed: int,
) -> dict[str, Any]:
    observed = received - timedelta(seconds=observed_delay)
    decision = received + timedelta(seconds=arrival_age)

    pr = random.Random(marker_seed)
    history: list[dict[str, str]] = [
        {
            "role": "assistant",
            "time": iso(received),
            "content": cache_text(source, key, value, observed, received),
        }
    ]

    # Place neutral messages evenly between cache receipt and decision.
    for j in range(n_messages):
        t = received + timedelta(seconds=arrival_age * (j + 1) / (n_messages + 1))
        role = "user" if j % 2 == 0 else "assistant"
        marker = pr.randint(100, 999)
        if source == "reservation" and invalidation and invalidation_slot == j:
            text = (
                f"Reservation-change notification: {key} changed externally after the cached record. "
                "The cached reservation status is no longer current."
            )
        elif role == "user":
            text = (
                f"Side note {j + 1}: record neutral marker {marker}. "
                "This does not update any cached source."
            )
        else:
            text = f"Marker {marker} recorded. No cached source was updated."
        history.append({"role": role, "time": iso(t), "content": text})

    history.append({"role": "user", "time": iso(decision), "content": final})

    world_age = arrival_age + observed_delay
    if source == "weather":
        valid = world_age <= TTL_S + 1e-9
    elif source == "discourse":
        valid = n_messages <= TTL_N
    else:
        valid = not invalidation

    case = {
        "case_id": f"cf-{pair_family}-{pair_index:05d}-{variant}",
        "pair_id": f"cf-{pair_family}-{pair_index:05d}",
        "pair_family": pair_family,
        "pair_variant": variant,
        "expected_pair_relation": expected_relation,
        "regime": f"counterfactual_{pair_family}",
        "source": source,
        "key": key,
        "cached_value": value,
        "observed_at": iso(observed),
        "received_at": iso(received),
        "decision_at": iso(decision),
        "valid_age_seconds": world_age,
        "arrival_age_seconds": arrival_age,
        "intervening_messages": n_messages,
        "explicit_invalidation": invalidation,
        "oracle_action": "reuse" if valid else "refresh",
        "history": history,
    }
    case["conditions"] = {
        condition: {"messages": render(case, condition), "tools": [TOOL]}
        for condition in CONDS
    }
    return case


def make_pair(seed: int, pair_index: int, family: str) -> list[dict[str, Any]]:
    rng = random.Random(seed + 10007 * pair_index + 97 * list(PAIR_FAMILIES).index(family))
    source = family.split("_", 1)[0]
    key, value, final = source_payload(source, pair_index, rng)
    received = datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc) + timedelta(minutes=3 * pair_index)
    marker_seed = seed + 30011 * pair_index

    if family == "weather_flip":
        # Same received time, decision time, event count and neutral content.
        # Only observed_at moves, crossing 8 seconds of WORLD age.
        common = dict(
            pair_index=pair_index, pair_family=family, expected_relation="flip",
            source=source, key=key, value=value, final=final, received=received,
            arrival_age=4.0, n_messages=6, invalidation=False, invalidation_slot=None,
            marker_seed=marker_seed,
        )
        return [
            build_case(variant="A", observed_delay=3.5, **common),  # 7.5 s -> reuse
            build_case(variant="B", observed_delay=4.5, **common),  # 8.5 s -> refresh
        ]

    if family == "weather_invariant":
        # Same world/arrival age, but event count changes drastically.
        world_age = 7.0 if rng.random() < 0.5 else 10.0
        arrival_age = 4.0
        delay = world_age - arrival_age
        common = dict(
            pair_index=pair_index, pair_family=family, expected_relation="same",
            source=source, key=key, value=value, final=final, received=received,
            observed_delay=delay, arrival_age=arrival_age, invalidation=False,
            invalidation_slot=None, marker_seed=marker_seed,
        )
        return [
            build_case(variant="A", n_messages=2, **common),
            build_case(variant="B", n_messages=14, **common),
        ]

    if family == "discourse_flip":
        # Same total wall time and source timestamps; only structural distance crosses 8 -> 9.
        common = dict(
            pair_index=pair_index, pair_family=family, expected_relation="flip",
            source=source, key=key, value=value, final=final, received=received,
            observed_delay=1.0, arrival_age=4.0, invalidation=False,
            invalidation_slot=None, marker_seed=marker_seed,
        )
        return [
            build_case(variant="A", n_messages=8, **common),  # valid
            build_case(variant="B", n_messages=9, **common),  # stale
        ]

    if family == "discourse_invariant":
        # Same structural distance; wall time changes sharply.
        n = 6 if rng.random() < 0.5 else 11
        common = dict(
            pair_index=pair_index, pair_family=family, expected_relation="same",
            source=source, key=key, value=value, final=final, received=received,
            observed_delay=1.0, n_messages=n, invalidation=False,
            invalidation_slot=None, marker_seed=marker_seed,
        )
        return [
            build_case(variant="A", arrival_age=3.0, **common),
            build_case(variant="B", arrival_age=24.0, **common),
        ]

    if family == "reservation_flip":
        # Same timing/count. One neutral slot becomes an explicit invalidation.
        common = dict(
            pair_index=pair_index, pair_family=family, expected_relation="flip",
            source=source, key=key, value=value, final=final, received=received,
            observed_delay=2.0, arrival_age=6.0, n_messages=6,
            invalidation_slot=3, marker_seed=marker_seed,
        )
        return [
            build_case(variant="A", invalidation=False, **common),
            build_case(variant="B", invalidation=True, **common),
        ]

    if family == "reservation_invariant":
        # Same invalidation state; both physical age and event count change sharply.
        invalid = rng.random() < 0.5
        common = dict(
            pair_index=pair_index, pair_family=family, expected_relation="same",
            source=source, key=key, value=value, final=final, received=received,
            observed_delay=2.0, invalidation=invalid,
            invalidation_slot=1 if invalid else None, marker_seed=marker_seed,
        )
        return [
            build_case(variant="A", arrival_age=3.0, n_messages=2, **common),
            build_case(variant="B", arrival_age=24.0, n_messages=14, **common),
        ]

    raise ValueError(family)


PAIR_FAMILIES = (
    "weather_flip",
    "weather_invariant",
    "discourse_flip",
    "discourse_invariant",
    "reservation_flip",
    "reservation_invariant",
)


def generate(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    index = 0
    for family in PAIR_FAMILIES:
        for _ in range(args.pairs_per_family):
            rows.extend(make_pair(args.seed, index, family))
            index += 1

    # Strong generator assertions.
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[row["pair_id"]].append(row)

    for pair_id, pair in by_pair.items():
        assert len(pair) == 2
        a, b = sorted(pair, key=lambda x: x["pair_variant"])
        relation = a["expected_pair_relation"]
        assert relation == b["expected_pair_relation"]
        if relation == "flip":
            assert a["oracle_action"] != b["oracle_action"], pair_id
            assert a["oracle_action"] == "reuse" and b["oracle_action"] == "refresh", pair_id
        else:
            assert a["oracle_action"] == b["oracle_action"], pair_id

    writejl(args.output, rows)
    print(
        f"wrote {len(rows)} cases = {len(by_pair)} counterfactual pairs; "
        f"{len(rows) * len(CONDS)} model decisions"
    )


def heuristic_action(case: dict[str, Any], policy: str) -> str:
    inv = case["explicit_invalidation"]
    if policy == "arrival":
        valid = case["arrival_age_seconds"] <= TTL_S and not inv
    elif policy == "world":
        valid = case["valid_age_seconds"] <= TTL_S and not inv
    elif policy == "position":
        valid = case["intervening_messages"] <= TTL_N and not inv
    elif policy == "invalidation":
        valid = not inv
    elif policy == "resolver":
        return case["oracle_action"]
    else:
        raise ValueError(policy)
    return "reuse" if valid else "refresh"


def heuristic_diagnostic(rows: list[dict[str, Any]]) -> None:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[row["pair_id"]].append(row)

    print("\nheuristic counterfactual relation accuracy")
    policies = ("arrival", "world", "position", "invalidation", "resolver")
    for policy in policies:
        family_scores: dict[str, list[bool]] = defaultdict(list)
        both_scores: list[bool] = []
        for pair in by_pair.values():
            a, b = sorted(pair, key=lambda x: x["pair_variant"])
            aa, ab = heuristic_action(a, policy), heuristic_action(b, policy)
            relation = "flip" if aa != ab else "same"
            family_scores[a["pair_family"]].append(
                relation == a["expected_pair_relation"]
            )
            both_scores.append(
                aa == a["oracle_action"] and ab == b["oracle_action"]
            )
        parts = " ".join(
            f"{family.replace('_invariant','_inv').replace('_flip','_flip')}="
            f"{np.mean(family_scores[family]):.2f}"
            for family in PAIR_FAMILIES
        )
        print(
            f"  {policy:>12s}: relation_mean="
            f"{np.mean([x for z in family_scores.values() for x in z]):.3f} "
            f"both_oracle={np.mean(both_scores):.3f}  {parts}"
        )


def sanity(args: argparse.Namespace) -> None:
    rows = readjl(args.input)
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[row["pair_id"]].append(row)

    print(f"cases={len(rows)} pairs={len(by_pair)}")
    for family in PAIR_FAMILIES:
        pairs = [
            sorted(pair, key=lambda x: x["pair_variant"])
            for pair in by_pair.values()
            if pair[0]["pair_family"] == family
        ]
        assert pairs
        correct_relation = 0
        for a, b in pairs:
            expected = a["expected_pair_relation"]
            got = "flip" if a["oracle_action"] != b["oracle_action"] else "same"
            correct_relation += int(got == expected)

            # Resolver must differ only in the direction implied by the oracle.
            assert a["conditions"]["raw"]["tools"] == b["conditions"]["raw"]["tools"]
            for case in (a, b):
                raw_tail = case["conditions"]["raw"]["messages"][1:]
                assert case["conditions"]["age_plane"]["messages"][1:] == raw_tail
                assert case["conditions"]["resolver"]["messages"][1:] == raw_tail

        print(
            f"{family:>23s}: n={len(pairs)} "
            f"relation_assert={correct_relation / len(pairs):.3f} "
            f"A={pairs[0][0]['oracle_action']} B={pairs[0][1]['oracle_action']}"
        )

    # Explicit source-isolation checks on representative pairs.
    reps = {family: next(
        sorted(pair, key=lambda x: x["pair_variant"])
        for pair in by_pair.values() if pair[0]["pair_family"] == family
    ) for family in PAIR_FAMILIES}

    a, b = reps["weather_flip"]
    assert a["intervening_messages"] == b["intervening_messages"]
    assert math.isclose(a["arrival_age_seconds"], b["arrival_age_seconds"])
    assert a["history"][1:] == b["history"][1:]  # only cache observed_at text differs

    a, b = reps["discourse_flip"]
    assert math.isclose(a["arrival_age_seconds"], b["arrival_age_seconds"])
    assert a["intervening_messages"] == 8 and b["intervening_messages"] == 9

    a, b = reps["reservation_flip"]
    assert math.isclose(a["valid_age_seconds"], b["valid_age_seconds"])
    assert a["intervening_messages"] == b["intervening_messages"]
    assert not a["explicit_invalidation"] and b["explicit_invalidation"]

    print("source-specific minimality checks passed")
    heuristic_diagnostic(rows)


def parse_tool_args(row: dict[str, Any], case: dict[str, Any]) -> bool:
    for call in row.get("tool_calls") or []:
        if call.get("name") != "refresh_source":
            continue
        try:
            args = json.loads(call.get("arguments")) if isinstance(call.get("arguments"), str) else (call.get("arguments") or {})
        except Exception:
            continue
        if args.get("source") == case["source"] and str(args.get("key")) == str(case["key"]):
            return True
    return False


def task_success(row: dict[str, Any], case: dict[str, Any]) -> bool:
    if case["oracle_action"] == "refresh":
        return row.get("action") == "refresh" and parse_tool_args(row, case)
    return (
        row.get("action") == "reuse"
        and str(case["cached_value"]).casefold() in (row.get("output_text") or "").casefold()
    )


def score(args: argparse.Namespace) -> None:
    cases = {row["case_id"]: row for row in readjl(args.cases)}
    responses = [row for row in readjl(args.responses) if row.get("case_id") in cases]
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in responses:
        by_condition[row["condition"]].append(row)

    for condition in sorted(by_condition):
        rows = by_condition[condition]
        indexed = {row["case_id"]: row for row in rows}

        case_ok = []
        task_ok = []
        utils = []
        for row in rows:
            case = cases[row["case_id"]]
            action = row.get("action")
            if action not in ("reuse", "refresh"):
                continue
            case_ok.append(action == case["oracle_action"])
            task_ok.append(task_success(row, case))
            utils.append(reward(action, case["oracle_action"]))

        pair_stats = []
        for pair_id in sorted({c["pair_id"] for c in cases.values()}):
            pair_cases = sorted(
                [c for c in cases.values() if c["pair_id"] == pair_id],
                key=lambda x: x["pair_variant"],
            )
            if len(pair_cases) != 2:
                continue
            a, b = pair_cases
            if a["case_id"] not in indexed or b["case_id"] not in indexed:
                continue
            ra, rb = indexed[a["case_id"]], indexed[b["case_id"]]
            aa, ab = ra.get("action"), rb.get("action")
            if aa not in ("reuse", "refresh") or ab not in ("reuse", "refresh"):
                continue
            relation = "flip" if aa != ab else "same"
            expected = a["expected_pair_relation"]
            pair_stats.append({
                "family": a["pair_family"],
                "expected": expected,
                "relation_ok": relation == expected,
                "both_oracle": aa == a["oracle_action"] and ab == b["oracle_action"],
                "spurious_flip": expected == "same" and relation == "flip",
                "missed_flip": expected == "flip" and relation == "same",
                "both_task": task_success(ra, a) and task_success(rb, b),
            })

        print(f"\n{condition}: cases={len(rows)}")
        if case_ok:
            print(
                f"  action_agreement={np.mean(case_ok):.3f} "
                f"task_success={np.mean(task_ok):.3f} utility={np.mean(utils):.3f}"
            )
        if pair_stats:
            print(
                f"  pair_relation={np.mean([p['relation_ok'] for p in pair_stats]):.3f} "
                f"both_oracle={np.mean([p['both_oracle'] for p in pair_stats]):.3f} "
                f"both_task={np.mean([p['both_task'] for p in pair_stats]):.3f}"
            )
            flip = [p for p in pair_stats if p["expected"] == "flip"]
            same = [p for p in pair_stats if p["expected"] == "same"]
            print(
                f"  causal_flip_success={1.0 - np.mean([p['missed_flip'] for p in flip]):.3f} "
                f"spurious_flip_rate={np.mean([p['spurious_flip'] for p in same]):.3f}"
            )
            for family in PAIR_FAMILIES:
                z = [p for p in pair_stats if p["family"] == family]
                if z:
                    print(
                        f"    {family:>23s}: relation={np.mean([p['relation_ok'] for p in z]):.3f} "
                        f"both_oracle={np.mean([p['both_oracle'] for p in z]):.3f}"
                    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--output", type=Path, required=True)
    g.add_argument("--pairs-per-family", type=int, default=40)
    g.add_argument("--seed", type=int, default=123)
    g.set_defaults(func=generate)

    s = sub.add_parser("sanity")
    s.add_argument("--input", type=Path, required=True)
    s.set_defaults(func=sanity)

    sc = sub.add_parser("score")
    sc.add_argument("--cases", type=Path, required=True)
    sc.add_argument("--responses", type=Path, required=True)
    sc.set_defaults(func=score)

    return p


if __name__ == "__main__":
    args = parser().parse_args()
    args.func(args)
