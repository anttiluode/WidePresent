"""Known-answer aliasing gate for unfinished asynchronous work.

The completed transcript can be identical while optimal action differs because one
runtime has a refresh already in flight and the other does not.

This is not a machine-learning benchmark. It is an observation-sufficiency check:
if two hidden runtime states map to the same model observation but require different
optimal actions, no policy that sees only that observation can be optimal in both.

Run:
    python experiments/inflight_state_aliasing.py
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Action(str, Enum):
    WAIT = "wait"
    LAUNCH = "launch"
    USE_CACHE = "use_cache"


@dataclass(frozen=True)
class RuntimeCase:
    name: str
    pending_refresh: bool
    pending_eta_ticks: int | None
    deadline_in_ticks: int = 3
    launch_latency_ticks: int = 2
    launch_cost: float = 0.50
    cache_valid: bool = False


# Deliberately identical model-visible completed state in both cases.
COMPLETED_TRANSCRIPT_VIEW = {
    "now_tick": 100,
    "cached_result_world_age_ticks": 20,
    "cached_result_knowledge_age_ticks": 20,
    "completed_tool_messages": 1,
    "deadline_in_ticks": 3,
}


def utility(case: RuntimeCase, action: Action) -> float:
    if action == Action.USE_CACHE:
        return 1.0 if case.cache_valid else -1.0

    if action == Action.WAIT:
        if (
            case.pending_refresh
            and case.pending_eta_ticks is not None
            and case.pending_eta_ticks <= case.deadline_in_ticks
        ):
            return 1.0
        return -1.0

    if action == Action.LAUNCH:
        fresh_before_deadline = case.launch_latency_ticks <= case.deadline_in_ticks
        base = 1.0 if fresh_before_deadline else -1.0
        return base - case.launch_cost

    raise ValueError(action)


def best_action(case: RuntimeCase) -> tuple[Action, float]:
    rows = [(action, utility(case, action)) for action in Action]
    return max(rows, key=lambda row: row[1])


def main() -> None:
    cases = [
        RuntimeCase(
            name="no_pending_refresh",
            pending_refresh=False,
            pending_eta_ticks=None,
        ),
        RuntimeCase(
            name="refresh_already_pending",
            pending_refresh=True,
            pending_eta_ticks=1,
        ),
    ]

    print("WidePresent: in-flight state aliasing gate")
    print()
    print("completed transcript / age-plane view (IDENTICAL in both worlds):")
    for key, value in COMPLETED_TRANSCRIPT_VIEW.items():
        print(f"  {key}: {value}")
    print()

    for case in cases:
        print(case.name)
        print(
            f"  hidden process state: pending={case.pending_refresh} "
            f"eta={case.pending_eta_ticks}"
        )
        for action in Action:
            print(f"  utility({action.value:>9s}) = {utility(case, action):+.3f}")
        action, value = best_action(case)
        print(f"  optimum = {action.value} ({value:+.3f})")
        print()

    first_action, _ = best_action(cases[0])
    second_action, _ = best_action(cases[1])
    assert first_action != second_action

    # Any deterministic policy restricted to the identical completed view must
    # choose one fixed action for both worlds. Evaluate the best such action under
    # a balanced prior over the two hidden process states.
    fixed_rows = []
    for action in Action:
        mean_u = sum(utility(case, action) for case in cases) / len(cases)
        fixed_rows.append((action, mean_u))
    best_fixed_action, best_fixed_utility = max(fixed_rows, key=lambda row: row[1])

    state_aware_utility = sum(best_action(case)[1] for case in cases) / len(cases)
    gap = state_aware_utility - best_fixed_utility

    print("observation-sufficiency result")
    print("  completed-view alias: TRUE")
    print(f"  optimal actions differ: {first_action.value} vs {second_action.value}")
    print(
        f"  best deterministic completed-view policy: {best_fixed_action.value} "
        f"mean_utility={best_fixed_utility:+.3f}"
    )
    print(f"  explicit process-state policy: mean_utility={state_aware_utility:+.3f}")
    print(f"  information gap in this balanced toy: {gap:+.3f}")
    print()
    print("Interpretation:")
    print("  * timestamps and completed messages are not sufficient state here;")
    print("  * the missing variable is whether useful work is already in flight;")
    print("  * no larger language model can infer a fact that the observation aliases away;")
    print("  * exposing pending/in-flight work is ordinary runtime bookkeeping, not a new")
    print("    temporal neural architecture.")


if __name__ == "__main__":
    main()
