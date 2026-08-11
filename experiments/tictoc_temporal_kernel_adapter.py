"""Prepare a TicToc condition with derived temporal runtime state.

This adapter intentionally does NOT vendor TicToc data. Point it at a local copy
of the official dataset/repository.

Fair comparison intended by this script:

B) official TicToc timestamp condition
   python eval_from_api.py ... --use_time_stamp

C) temporal-kernel condition
   python this_script.py --input ... --output derived.json
   python eval_from_api.py --data derived.json ...   # NO --use_time_stamp

The kernel condition receives no human labels, no inferred staleness threshold,
and no action recommendation. It only converts the timestamps already present
in the sample into clock-derived relative ages plus the decision time.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from temporal_kernel import derive_temporal_state, select_variant_time


def number_of_variants(sample: dict[str, Any]) -> int:
    lengths: set[int] = set()
    for msg in sample.get("history", []):
        value = msg.get("time")
        if isinstance(value, list):
            lengths.add(len(value))
    for key in ("call_tool_output", "no_call_tool_output"):
        obj = sample.get(key)
        if isinstance(obj, dict) and isinstance(obj.get("time"), list):
            lengths.add(len(obj["time"]))
    if not lengths:
        return 1
    if len(lengths) != 1:
        raise ValueError(f"inconsistent time-variant counts in sample {sample.get('id')}: {lengths}")
    return next(iter(lengths))


def scalarize_time_fields(sample: dict[str, Any], level: int) -> dict[str, Any]:
    out = copy.deepcopy(sample)
    for msg in out.get("history", []):
        if "time" in msg:
            msg["time"] = select_variant_time(msg["time"], level)
    for key in ("call_tool_output", "no_call_tool_output"):
        obj = out.get(key)
        if isinstance(obj, dict) and "time" in obj:
            obj["time"] = select_variant_time(obj["time"], level)
    return out


def timestamp_visible_in_official_openai_condition(msg: dict[str, Any]) -> bool:
    """Mirror the official OpenAI handler's timestamp-exposure rule.

    The handler prefixes timestamps only when content is non-null and the
    message is not itself an assistant tool-call envelope.
    """
    return msg.get("content") is not None and "tool_calls" not in msg


def render_fair_kernel_block(history: list[dict[str, Any]], level: int = 0) -> str:
    state = derive_temporal_state(history, level=level)
    visible_indices = {
        i for i, msg in enumerate(history) if timestamp_visible_in_official_openai_condition(msg)
    }

    lines = [
        "TEMPORAL RUNTIME STATE (computed from the same clock information as the timestamp condition):",
        f"- current decision time: {state.now.isoformat(timespec='milliseconds').replace('+00:00', 'Z')}",
        f"- elapsed since conversation start: {state.conversation_age_seconds:.3f} seconds",
        "- elapsed age of timestamp-visible prior messages at the current decision:",
    ]
    for m in state.messages:
        if m.index not in visible_indices:
            continue
        name = f", name={m.name}" if m.name else ""
        lines.append(f"  - message[{m.index}] role={m.role}{name}: {m.age_seconds:.3f} seconds ago")
    lines.append(
        "Timing facts only: no freshness threshold, tool-use recommendation, or human preference label is provided."
    )
    return "\n".join(lines)


def inject_kernel_state(sample: dict[str, Any], level: int = 0) -> dict[str, Any]:
    out = scalarize_time_fields(sample, level)
    history = out.get("history", [])
    if not history:
        raise ValueError("sample has empty history")

    block = render_fair_kernel_block(history, level=0)  # already scalarized
    system = next((msg for msg in history if msg.get("role") == "system"), None)
    if system is None:
        history.insert(0, {"role": "system", "content": block, "time": history[0]["time"]})
    else:
        original = system.get("content") or ""
        system["content"] = original.rstrip() + "\n\n" + block

    out["widepresent_temporal_condition"] = {
        "kind": "derived_kernel_v1",
        "time_variant": level,
        "label_blind": True,
    }
    return out


def prepare(data: list[dict[str, Any]], expand_variants: bool = False) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for sample in data:
        nvariants = number_of_variants(sample)
        levels = range(nvariants) if expand_variants else range(1)
        for level in levels:
            item = inject_kernel_state(sample, level)
            if expand_variants and nvariants > 1:
                original_id = str(item.get("id", "sample"))
                item["id"] = f"{original_id}::time_variant_{level + 1}"
            prepared.append(item)
    return prepared


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Local TicToc JSON file")
    p.add_argument("--output", required=True, help="Output JSON with derived temporal state")
    p.add_argument(
        "--expand-variants",
        action="store_true",
        help="Expand raw scenario files whose final timestamps contain three alternatives",
    )
    args = p.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]

    prepared = prepare(data, expand_variants=args.expand_variants)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(prepared, f, indent=2, ensure_ascii=False)

    print(f"wrote {len(prepared)} samples to {args.output}")
    print("Run TicToc inference on this file WITHOUT --use_time_stamp.")
    print("Compare against the same underlying samples WITH --use_time_stamp.")


if __name__ == "__main__":
    main()
