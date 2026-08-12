"""Measure real probe costs for the context re-entry question.

This script does not decide what your probes *mean*. It times read-only commands and
asks whether conditional branch measurement is cheaper than eagerly measuring every
possible branch on every resume.

Example (replace commands with read-only commands that make sense in your repo):

    python experiments/reentry_command_costs.py \
      --route "git status --porcelain=v1" \
      --branch "code=python -m pytest -q" \
      --branch "docs=python -m mkdocs build --strict" \
      --branch "remote=gh run view --json status" \
      --weight "code=0.6" --weight "docs=0.2" --weight "remote=0.2"

Only use commands safe to execute repeatedly. Commands may be expensive.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import statistics
import subprocess
import time
from typing import Iterable


@dataclass(frozen=True)
class Timing:
    name: str
    command: str
    samples_s: tuple[float, ...]

    @property
    def median_s(self) -> float:
        return statistics.median(self.samples_s)

    @property
    def mean_s(self) -> float:
        return statistics.mean(self.samples_s)


def parse_name_value(items: Iterable[str], option: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"{option} expects NAME=VALUE, got: {item!r}")
        name, value = item.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name or not value:
            raise SystemExit(f"{option} expects non-empty NAME=VALUE, got: {item!r}")
        if name in out:
            raise SystemExit(f"duplicate {option} name: {name}")
        out[name] = value
    return out


def parse_weights(items: Iterable[str]) -> dict[str, float]:
    raw = parse_name_value(items, "--weight")
    weights = {name: float(value) for name, value in raw.items()}
    if any(value < 0 for value in weights.values()):
        raise SystemExit("weights must be non-negative")
    total = sum(weights.values())
    if total <= 0:
        raise SystemExit("weights must sum to a positive value")
    return {name: value / total for name, value in weights.items()}


def run_once(command: str, cwd: str | None) -> float:
    start = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=cwd,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(
            f"command returned {proc.returncode}: {command}\n"
            "Use only commands that are expected to succeed in the calibration state."
        )
    return elapsed


def time_command(name: str, command: str, repeats: int, cwd: str | None) -> Timing:
    samples = tuple(run_once(command, cwd) for _ in range(repeats))
    return Timing(name=name, command=command, samples_s=samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--route",
        required=True,
        help="read-only command for the cheap routing/recorded-state probe",
    )
    parser.add_argument(
        "--branch",
        action="append",
        default=[],
        metavar="NAME=COMMAND",
        help="branch-specific read-only measurement command; repeat for multiple branches",
    )
    parser.add_argument(
        "--weight",
        action="append",
        default=[],
        metavar="NAME=PROBABILITY",
        help="expected branch frequency; normalized automatically",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cwd", default=None)
    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    branch_commands = parse_name_value(args.branch, "--branch")
    if not branch_commands:
        raise SystemExit("provide at least one --branch NAME=COMMAND")

    if args.weight:
        weights = parse_weights(args.weight)
        missing = set(branch_commands) - set(weights)
        extra = set(weights) - set(branch_commands)
        if missing or extra:
            raise SystemExit(
                f"weights must match branch names exactly; missing={sorted(missing)} "
                f"extra={sorted(extra)}"
            )
    else:
        equal = 1.0 / len(branch_commands)
        weights = {name: equal for name in branch_commands}

    print("WidePresent: real command cost calibration")
    print("WARNING: commands are executed repeatedly. Use read-only commands.")
    print()

    route = time_command("route", args.route, args.repeats, args.cwd)
    branches = {
        name: time_command(name, command, args.repeats, args.cwd)
        for name, command in branch_commands.items()
    }

    print("measured command costs (median seconds)")
    print(f"  {'route':>18s}: {route.median_s:10.6f}  {args.route}")
    for name, timing in branches.items():
        print(f"  {name:>18s}: {timing.median_s:10.6f}  {timing.command}")
    print()

    eager_cost = route.median_s + sum(t.median_s for t in branches.values())
    reentry_cost = route.median_s + sum(
        weights[name] * branches[name].median_s for name in branches
    )
    saved = eager_cost - reentry_cost
    fraction = saved / eager_cost if eager_cost > 0 else 0.0

    print("branch weights")
    for name, weight in weights.items():
        print(f"  {name:>18s}: {weight:.3f}")
    print()

    print("cost comparison")
    print(f"  eager verified snapshot : {eager_cost:.6f} s/resume")
    print(f"  conditional re-entry    : {reentry_cost:.6f} s/resume")
    print(f"  expected saving         : {saved:.6f} s/resume")
    print(f"  fraction saved          : {fraction:.3%}")
    print()

    most_expensive = max(branches.values(), key=lambda t: t.median_s)
    p = weights[most_expensive.name]
    avoid_rate = 1.0 - p
    print("diagnostic")
    print(
        f"  most expensive branch: {most_expensive.name} "
        f"({most_expensive.median_s:.6f}s), expected need rate={p:.3f}"
    )
    print(f"  policy avoids that measurement on ~{avoid_rate:.1%} of resumptions")
    print()
    print("Interpretation:")
    print("  * if the measured saving is tiny, prefer a boring eager snapshot;")
    print("  * if expensive measurements are skipped often, a re-entry policy has a real")
    print("    wall-clock/tool-cost reason to exist;")
    print("  * this calibrates cost only. A real interruption benchmark must still measure")
    print("    recovery correctness, duplicate work, stale assumptions, and maintenance cost.")


if __name__ == "__main__":
    main()
