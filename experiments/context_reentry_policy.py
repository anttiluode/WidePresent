"""Context re-entry policy gate.

A small known-answer benchmark for recovering *current process state* after the
completed transcript has become insufficient (for example after a context reset).

This is intentionally different from the active temporal-semantic probes elsewhere in
WidePresent. Those ask which observation best identifies a hidden source rule. Here the
rule is known; the missing variable is the agent's *current place in a live workflow*.

Four hidden runtime states expose the same completed transcript but require four
different next actions. Live probes can recover the state:

    worktree -> dirty/clean
      dirty -> tests -> editing/testing
      clean -> remote -> waiting_remote/complete

The two-step decision tree is a tiny "re-entry card": it stores how to reconstruct the
current task state, not the state itself.

Known-answer comparisons:

    completed transcript only        25% accuracy, 0 probes
    full live snapshot              100% accuracy, 3 probes
    stored re-entry policy          100% accuracy, 2 probes
    two random distinct probes       75% expected accuracy, 2 probes
    random probes until identified  100% accuracy, 2.333 probes on average

This is ordinary diagnosis / active state estimation. The point is the runtime design
primitive, not a novel inference algorithm.

Run:
    python experiments/context_reentry_policy.py
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import itertools
from typing import Callable


class Action(str, Enum):
    CONTINUE_EDIT = "continue_edit"
    WAIT_TESTS = "wait_tests"
    WAIT_REMOTE = "wait_remote"
    PUBLISH = "publish"


@dataclass(frozen=True)
class World:
    name: str
    worktree_dirty: bool
    tests_running: bool
    remote_running: bool
    correct_action: Action


WORLDS = (
    World("editing", True, False, False, Action.CONTINUE_EDIT),
    World("testing", True, True, False, Action.WAIT_TESTS),
    World("waiting_remote", False, False, True, Action.WAIT_REMOTE),
    World("complete", False, False, False, Action.PUBLISH),
)

# Deliberately identical observation in all four worlds.
COMPLETED_TRANSCRIPT_VIEW = {
    "task": "finish_and_publish_patch",
    "last_completed_message": "patch prepared; continue workflow",
    "completed_tool_calls": 7,
    "last_known_file": "feature.py",
}

Probe = Callable[[World], str]
PROBES: dict[str, Probe] = {
    "worktree": lambda w: "dirty" if w.worktree_dirty else "clean",
    "tests": lambda w: "running" if w.tests_running else "idle",
    "remote": lambda w: "running" if w.remote_running else "idle",
}


def compatible_worlds(observations: dict[str, str]) -> list[World]:
    return [
        world
        for world in WORLDS
        if all(PROBES[name](world) == value for name, value in observations.items())
    ]


def reentry_policy(world: World) -> tuple[Action, int, dict[str, str]]:
    """Task-specific two-step recovery recipe."""
    observations: dict[str, str] = {}

    first = PROBES["worktree"](world)
    observations["worktree"] = first

    second_probe = "tests" if first == "dirty" else "remote"
    observations[second_probe] = PROBES[second_probe](world)

    candidates = compatible_worlds(observations)
    assert len(candidates) == 1
    return candidates[0].correct_action, 2, observations


def full_snapshot(world: World) -> tuple[Action, int]:
    observations = {name: probe(world) for name, probe in PROBES.items()}
    candidates = compatible_worlds(observations)
    assert len(candidates) == 1
    return candidates[0].correct_action, len(PROBES)


def random_until_unique(world: World, order: tuple[str, ...]) -> tuple[Action, int]:
    observations: dict[str, str] = {}
    for cost, name in enumerate(order, start=1):
        observations[name] = PROBES[name](world)
        candidates = compatible_worlds(observations)
        if len(candidates) == 1:
            return candidates[0].correct_action, cost
    raise AssertionError("all probes should identify the world")


def expected_random_two_probe_accuracy() -> float:
    """
    Exact Bayes accuracy under balanced worlds, random distinct probes, and optimal
    guessing among the worlds still compatible after two observations.

    If k worlds remain observationally identical, the best balanced-posterior success
    is 1/k. Averaging over every world and probe permutation gives 0.75 here.
    """
    scores: list[float] = []
    for world in WORLDS:
        for order in itertools.permutations(PROBES):
            observations = {
                name: PROBES[name](world)
                for name in order[:2]
            }
            scores.append(1.0 / len(compatible_worlds(observations)))
    return sum(scores) / len(scores)


def mean_random_identification_cost() -> float:
    costs: list[int] = []
    for world in WORLDS:
        for order in itertools.permutations(PROBES):
            _, cost = random_until_unique(world, order)
            costs.append(cost)
    return sum(costs) / len(costs)


def main() -> None:
    print("WidePresent: context re-entry policy gate")
    print()
    print("completed transcript view (IDENTICAL in all worlds):")
    for key, value in COMPLETED_TRANSCRIPT_VIEW.items():
        print(f"  {key}: {value}")
    print()

    for world in WORLDS:
        action, cost, observations = reentry_policy(world)
        assert action == world.correct_action
        snapshot_action, snapshot_cost = full_snapshot(world)
        assert snapshot_action == world.correct_action
        print(f"{world.name:>14s}")
        print(f"  hidden correct action : {world.correct_action.value}")
        print(f"  re-entry observations : {observations}")
        print(f"  recovered action      : {action.value} (cost {cost})")
        print(f"  full snapshot cost    : {snapshot_cost}")
        print()

    # Four balanced worlds require four different actions, while the completed view is
    # identical. Therefore a fixed transcript-only action is correct in only one world.
    transcript_only_accuracy = 1.0 / len(WORLDS)
    reentry_accuracy = 1.0
    full_snapshot_accuracy = 1.0
    random_two_accuracy = expected_random_two_probe_accuracy()
    random_until_cost = mean_random_identification_cost()

    print("known-answer summary")
    print(f"  transcript only        : accuracy={transcript_only_accuracy:.3f} probes=0")
    print(f"  stored re-entry policy : accuracy={reentry_accuracy:.3f} probes=2")
    print(f"  full live snapshot     : accuracy={full_snapshot_accuracy:.3f} probes=3")
    print(f"  random two probes      : expected_accuracy={random_two_accuracy:.3f} probes=2")
    print(f"  random until unique    : accuracy=1.000 mean_probes={random_until_cost:.3f}")

    assert abs(transcript_only_accuracy - 0.25) < 1e-12
    assert abs(random_two_accuracy - 0.75) < 1e-12
    assert abs(random_until_cost - 7.0 / 3.0) < 1e-12

    print()
    print("Interpretation:")
    print("  * completed history can alias away the agent's current place in a workflow;")
    print("  * a re-entry card stores a recovery procedure, not a hidden answer;")
    print("  * task structure can make recovery cheaper than dumping all live state;")
    print("  * this is ordinary active diagnosis/state estimation, not a new algorithm;")
    print("  * the product question is whether such tiny recovery recipes beat large passive")
    print("    context replay on real agent interruptions and resumptions.")


if __name__ == "__main__":
    main()
