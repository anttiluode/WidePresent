"""Heterogeneous-cost context re-entry gate.

This attacks the first re-entry toy's weakest assumption: every live probe cost one unit.

The new world distinguishes cheap *recorded* state from expensive *measured* state.
A cheap route probe says which validation branch is relevant. Only that branch's
measurement is executed.

This is a known-answer cost model, not a real benchmark. The assigned costs are
deliberately visible and replaceable. Use ``reentry_command_costs.py`` to time actual
commands on a local repository.

Run:
    python experiments/context_reentry_cost_gate.py
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from statistics import mean


class Action(str, Enum):
    PUBLISH_CODE = "publish_code"
    FIX_CODE = "fix_code"
    PUBLISH_DOCS = "publish_docs"
    FIX_DOCS = "fix_docs"
    WAIT_REMOTE = "wait_remote"
    FINISH_REMOTE = "finish_remote"


@dataclass(frozen=True)
class World:
    name: str
    route: str
    outcome: str
    correct_action: Action


# Two equally likely outcomes per route. All six worlds have the same completed history.
WORLDS = (
    World("code_pass", "code", "pass", Action.PUBLISH_CODE),
    World("code_fail", "code", "fail", Action.FIX_CODE),
    World("docs_pass", "docs", "pass", Action.PUBLISH_DOCS),
    World("docs_fail", "docs", "fail", Action.FIX_DOCS),
    World("remote_running", "remote", "running", Action.WAIT_REMOTE),
    World("remote_idle", "remote", "idle", Action.FINISH_REMOTE),
)

COMPLETED_TRANSCRIPT_VIEW = {
    "task": "resume_and_finish_patch",
    "last_completed_message": "work interrupted after patch preparation",
    "completed_tool_calls": 9,
}

# Cost units are intentionally arbitrary. Think of them as relative wall-clock/tool cost.
# route: cheap recorded metadata/diff classification
# code_validation: e.g. test suite
# docs_validation: e.g. docs build/link check
# remote_status: e.g. remote process/API status
PROBE_COST = {
    "route": 1.0,
    "code_validation": 20.0,
    "docs_validation": 8.0,
    "remote_status": 2.0,
}


def branch_probe(route: str) -> str:
    return {
        "code": "code_validation",
        "docs": "docs_validation",
        "remote": "remote_status",
    }[route]


def reentry_cost(world: World) -> float:
    """Cheap route read, then only the measurement relevant to that branch."""
    return PROBE_COST["route"] + PROBE_COST[branch_probe(world.route)]


def eager_snapshot_cost(_: World) -> float:
    """Measure every potentially useful live variable, whether relevant or not."""
    return sum(PROBE_COST.values())


def outcome_to_action(route: str, outcome: str) -> Action:
    table = {
        ("code", "pass"): Action.PUBLISH_CODE,
        ("code", "fail"): Action.FIX_CODE,
        ("docs", "pass"): Action.PUBLISH_DOCS,
        ("docs", "fail"): Action.FIX_DOCS,
        ("remote", "running"): Action.WAIT_REMOTE,
        ("remote", "idle"): Action.FINISH_REMOTE,
    }
    return table[(route, outcome)]


def main() -> None:
    print("WidePresent: heterogeneous-cost context re-entry gate")
    print()
    print("completed transcript view (IDENTICAL in all worlds):")
    for key, value in COMPLETED_TRANSCRIPT_VIEW.items():
        print(f"  {key}: {value}")
    print()

    print("probe costs (assigned cost units; NOT measurements):")
    for name, cost in PROBE_COST.items():
        print(f"  {name:>16s}: {cost:5.1f}")
    print()

    reentry_costs = []
    snapshot_costs = []
    for world in WORLDS:
        action = outcome_to_action(world.route, world.outcome)
        assert action == world.correct_action
        rc = reentry_cost(world)
        sc = eager_snapshot_cost(world)
        reentry_costs.append(rc)
        snapshot_costs.append(sc)
        print(f"{world.name:>16s}")
        print(f"  cheap route       : {world.route}")
        print(f"  selected measure  : {branch_probe(world.route)}")
        print(f"  measured outcome  : {world.outcome}")
        print(f"  recovered action  : {action.value}")
        print(f"  re-entry cost     : {rc:.1f}")
        print(f"  eager snapshot    : {sc:.1f}")
        print()

    avg_reentry = mean(reentry_costs)
    avg_snapshot = mean(snapshot_costs)
    saving = avg_snapshot - avg_reentry
    saving_fraction = saving / avg_snapshot

    transcript_only_accuracy = 1.0 / len(WORLDS)
    print("known-answer balanced summary")
    print(f"  transcript-only accuracy        : {transcript_only_accuracy:.3f}")
    print("  re-entry accuracy               : 1.000")
    print("  eager verified snapshot accuracy: 1.000")
    print(f"  mean re-entry cost              : {avg_reentry:.3f}")
    print(f"  mean eager snapshot cost        : {avg_snapshot:.3f}")
    print(f"  mean cost saved                 : {saving:.3f}")
    print(f"  fraction of snapshot cost saved : {saving_fraction:.3%}")
    print()

    # With balanced routes, the exact expected cost is:
    expected = PROBE_COST["route"] + mean(
        [PROBE_COST["code_validation"], PROBE_COST["docs_validation"], PROBE_COST["remote_status"]]
    )
    assert abs(avg_reentry - expected) < 1e-12
    assert avg_reentry < avg_snapshot
    assert abs(transcript_only_accuracy - 1.0 / 6.0) < 1e-12

    print("Interpretation:")
    print("  * a full 'live snapshot' is not free when some fields only exist after measurement;")
    print("  * the route probe is useful because it prevents irrelevant expensive measurements;")
    print("  * the value is cost-sensitive active diagnosis, not the word 're-entry';")
    print("  * assigned costs prove only the arithmetic; real command timings must decide whether")
    print("    this matters in an actual repository or agent runtime.")


if __name__ == "__main__":
    main()
