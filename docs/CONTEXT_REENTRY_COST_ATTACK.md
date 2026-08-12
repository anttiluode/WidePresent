# Context re-entry cost attack: recorded state versus measured state

Date: 2026-08-12

The first context re-entry toy made every live probe cost one unit. Under that assumption
a task-specific re-entry card beat a full live snapshot by only one probe out of three.
That is not enough to justify a framework: a compact eager snapshot is simpler, more
robust, and does not require a hand-authored decision tree.

The important missing distinction is:

```text
RECORDED STATE
    already exists somewhere and is cheap to read

MEASURED STATE
    does not exist until the system performs work to discover it
```

Examples in a coding workflow:

```text
git branch / worktree metadata       recorded / cheap
running-process status               usually recorded / cheap
whether the current code passes      measured / potentially expensive
whether docs build successfully      measured / potentially expensive
whether a remote job has completed   measured, but often cheaper than local tests
```

A "full live snapshot" is cheap only if all of its fields are already recorded. If some
fields mean "go find out now", eager snapshotting can perform irrelevant expensive work.

---

## 1. New known-answer gate

The implementation is:

```text
experiments/context_reentry_cost_gate.py
```

It uses six balanced hidden worlds with the same completed transcript:

```text
code_pass
code_fail

docs_pass
docs_fail

remote_running
remote_idle
```

A cheap route read identifies which branch matters. The agent then performs only that
branch's measurement.

Assigned visible costs are:

```text
route read          1
code validation    20
docs validation     8
remote status       2
```

These numbers are deliberately arbitrary. They test the arithmetic only.

### Eager verified snapshot

Measure everything on every resume:

```text
1 + 20 + 8 + 2 = 31 cost units
```

### Conditional re-entry

```text
code branch    1 + 20 = 21
docs branch    1 +  8 =  9
remote branch  1 +  2 =  3
```

With balanced routes:

```text
mean re-entry cost       11
mean eager snapshot cost 31
saved                     20
fraction saved          64.5%
```

Both policies are 100% accurate in the constructed world. The difference is purely the
amount of unnecessary measurement.

This is still not a product result. It only demonstrates that heterogeneous probe costs
can make the re-entry idea nontrivial.

---

## 2. The actual principle

The feature should not be sold as "memory for agents" or "procedural context".

The useful engineering principle is ordinary cost-sensitive diagnosis:

> **Use cheap recorded state to decide which expensive live measurements are worth
> creating.**

The re-entry card earns its keep only when it prevents measurements that would otherwise
consume meaningful wall-clock time, compute, money, tool quota, or destructive side
effects.

If every relevant variable is already recorded cheaply, prefer a boring snapshot.

---

## 3. Real command cost calibrator

The implementation is:

```text
experiments/reentry_command_costs.py
```

It executes user-supplied **read-only** commands repeatedly, measures median wall-clock
cost, and compares:

```text
eager verified snapshot
    route + every branch measurement

conditional re-entry
    route + expected cost of only the selected branch
```

Example shape:

```bash
python experiments/reentry_command_costs.py \
  --route "git status --porcelain=v1" \
  --branch "code=python -m pytest -q" \
  --branch "docs=python -m mkdocs build --strict" \
  --branch "remote=gh run view --json status" \
  --weight "code=0.6" \
  --weight "docs=0.2" \
  --weight "remote=0.2"
```

The commands above are examples only. Use commands appropriate to the local repository
and safe to execute repeatedly.

The branch weights should eventually come from observed interruption/resume frequencies,
not preference.

---

## 4. What a real benchmark must measure

Wall-clock probe cost is necessary but not sufficient.

A realistic interruption benchmark should randomly interrupt workflows such as:

```text
edit -> test -> package -> publish
retrieve -> transform -> validate -> write
render -> inspect -> revise -> export
```

Then erase short-term agent state while preserving the live filesystem/process/world.

Compare:

```text
A. passive history replay
B. compact eager live snapshot
C. generic fixed probe list
D. task-specific conditional re-entry
E. adaptive diagnosis
F. perfect checkpoint / oracle
```

Score at least:

```text
correct resumed action
wall-clock/tool cost
context tokens consumed
duplicate expensive operations
missed in-flight work
stale assumptions
maintenance/configuration burden
```

The boring eager snapshot remains the main attacker.

---

## 5. Stop / go rule

### Kill / demote

If real probes are all cheap, or if eager snapshotting is within noise of the conditional
policy after maintenance overhead, use the snapshot.

If task-specific cards require brittle hand-authored logic that fails when workflows
change, the saved measurement cost may not repay the complexity.

### Continue

The idea becomes product-shaped if a small stable routing policy repeatedly avoids
expensive validations while preserving safe continuation.

A particularly strong case would be:

```text
cheap recorded probe: milliseconds
expensive validation: seconds/minutes or paid tool call
expensive branch needed: minority of resumptions
```

Then the re-entry card is not a memory metaphor. It is a practical scheduler for
**which facts need to be re-measured after interruption**.

---

## 6. Revised interpretation

The first note said:

> recover the present instead of replaying the past.

After this attack, the more precise engineering sentence is:

> **Recover the present by reading what is already recorded first, then measure only the
> unresolved live variables that can still change the next action.**

That is the version worth testing on real local workflows.
