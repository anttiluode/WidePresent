"""Compile a minimum-expected-cost context re-entry policy.

This answers a practical criticism of hand-authored re-entry cards: if a workflow has a
finite declarative state/action schema, the recovery decision tree can be generated from
probe outcomes and costs.

This is standard cost-sensitive decision-tree / active-diagnosis dynamic programming,
not a new algorithm.

The demo reuses the heterogeneous-cost six-world gate and should rediscover:

    route
      code   -> code_validation
      docs   -> docs_validation
      remote -> remote_status

with balanced expected cost 11.0 versus eager verified snapshot cost 31.0.

Run:
    python experiments/reentry_policy_compiler.py
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import math


@dataclass(frozen=True)
class World:
    name: str
    action: str
    prior: float
    observations: tuple[tuple[str, str], ...]

    def outcome(self, probe: str) -> str:
        return dict(self.observations)[probe]


@dataclass(frozen=True)
class Leaf:
    action: str


@dataclass(frozen=True)
class Branch:
    probe: str
    children: tuple[tuple[str, "Tree"], ...]


Tree = Leaf | Branch


PROBE_COST = {
    "route": 1.0,
    "code_validation": 20.0,
    "docs_validation": 8.0,
    "remote_status": 2.0,
}


WORLDS = (
    World(
        "code_pass", "publish_code", 1 / 6,
        (("route", "code"), ("code_validation", "pass"),
         ("docs_validation", "n/a"), ("remote_status", "n/a")),
    ),
    World(
        "code_fail", "fix_code", 1 / 6,
        (("route", "code"), ("code_validation", "fail"),
         ("docs_validation", "n/a"), ("remote_status", "n/a")),
    ),
    World(
        "docs_pass", "publish_docs", 1 / 6,
        (("route", "docs"), ("code_validation", "n/a"),
         ("docs_validation", "pass"), ("remote_status", "n/a")),
    ),
    World(
        "docs_fail", "fix_docs", 1 / 6,
        (("route", "docs"), ("code_validation", "n/a"),
         ("docs_validation", "fail"), ("remote_status", "n/a")),
    ),
    World(
        "remote_running", "wait_remote", 1 / 6,
        (("route", "remote"), ("code_validation", "n/a"),
         ("docs_validation", "n/a"), ("remote_status", "running")),
    ),
    World(
        "remote_idle", "finish_remote", 1 / 6,
        (("route", "remote"), ("code_validation", "n/a"),
         ("docs_validation", "n/a"), ("remote_status", "idle")),
    ),
)


def action_set(ids: tuple[int, ...]) -> set[str]:
    return {WORLDS[i].action for i in ids}


def split(ids: tuple[int, ...], probe: str) -> dict[str, tuple[int, ...]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i in ids:
        groups[WORLDS[i].outcome(probe)].append(i)
    return {outcome: tuple(group) for outcome, group in groups.items()}


def mass(ids: tuple[int, ...]) -> float:
    return sum(WORLDS[i].prior for i in ids)


@lru_cache(maxsize=None)
def compile_optimal(ids: tuple[int, ...]) -> tuple[float, Tree]:
    """Minimum expected probe cost until the required action is uniquely determined."""
    ids = tuple(sorted(ids))
    actions = action_set(ids)
    if len(actions) == 1:
        return 0.0, Leaf(next(iter(actions)))

    total = mass(ids)
    best_cost = math.inf
    best_tree: Tree | None = None

    for probe, probe_cost in PROBE_COST.items():
        groups = split(ids, probe)
        if len(groups) <= 1:
            continue

        expected = probe_cost
        children: list[tuple[str, Tree]] = []
        for outcome, child_ids in sorted(groups.items()):
            child_cost, child_tree = compile_optimal(child_ids)
            expected += (mass(child_ids) / total) * child_cost
            children.append((outcome, child_tree))

        if expected < best_cost:
            best_cost = expected
            best_tree = Branch(probe, tuple(children))

    if best_tree is None:
        raise RuntimeError(
            "No remaining probe can distinguish worlds that require different actions."
        )
    return best_cost, best_tree


def action_entropy(ids: tuple[int, ...]) -> float:
    total = mass(ids)
    by_action: dict[str, float] = defaultdict(float)
    for i in ids:
        by_action[WORLDS[i].action] += WORLDS[i].prior / total
    return -sum(p * math.log2(p) for p in by_action.values() if p > 0)


def greedy_probe(ids: tuple[int, ...]) -> str:
    """Cheap baseline: action-entropy reduction per unit probe cost."""
    base = action_entropy(ids)
    total = mass(ids)
    best_score = -math.inf
    best_probe = ""
    for probe, cost in PROBE_COST.items():
        groups = split(ids, probe)
        if len(groups) <= 1:
            continue
        after = sum(
            (mass(child) / total) * action_entropy(child)
            for child in groups.values()
        )
        score = (base - after) / cost
        if score > best_score:
            best_score = score
            best_probe = probe
    if not best_probe:
        raise RuntimeError("greedy policy cannot distinguish remaining actions")
    return best_probe


def greedy_expected_cost(ids: tuple[int, ...]) -> float:
    if len(action_set(ids)) == 1:
        return 0.0
    probe = greedy_probe(ids)
    groups = split(ids, probe)
    total = mass(ids)
    return PROBE_COST[probe] + sum(
        (mass(child) / total) * greedy_expected_cost(child)
        for child in groups.values()
    )


def render(tree: Tree, indent: str = "") -> list[str]:
    if isinstance(tree, Leaf):
        return [f"{indent}-> {tree.action}"]
    lines = [f"{indent}probe {tree.probe}  [cost={PROBE_COST[tree.probe]:g}]"]
    for outcome, child in tree.children:
        lines.append(f"{indent}  if {outcome}:")
        lines.extend(render(child, indent + "    "))
    return lines


def main() -> None:
    ids = tuple(range(len(WORLDS)))
    optimal_cost, tree = compile_optimal(ids)
    greedy_cost = greedy_expected_cost(ids)
    eager_cost = sum(PROBE_COST.values())

    print("WidePresent: re-entry policy compiler")
    print()
    print("compiled minimum-expected-cost tree:")
    print("\n".join(render(tree)))
    print()
    print("known-answer costs")
    print(f"  optimal compiled policy : {optimal_cost:.3f}")
    print(f"  greedy info/cost policy : {greedy_cost:.3f}")
    print(f"  eager verified snapshot : {eager_cost:.3f}")
    print()

    assert abs(optimal_cost - 11.0) < 1e-12
    assert abs(greedy_cost - 11.0) < 1e-12
    assert abs(eager_cost - 31.0) < 1e-12

    print("Interpretation:")
    print("  * the recovery recipe need not be handwritten when workflow states are declarative;")
    print("  * in this tiny tree, a simple greedy information-per-cost rule is already optimal;")
    print("  * therefore the exact dynamic-programming compiler has not earned complexity yet;")
    print("  * the real burden moves to defining/maintaining the workflow state, actions, probes,")
    print("    costs, and priors accurately enough to compile from them.")


if __name__ == "__main__":
    main()
