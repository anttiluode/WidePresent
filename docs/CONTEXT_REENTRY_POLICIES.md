# Context re-entry policies — recover the present instead of replaying the past

Date: 2026-08-12

This note is a direct continuation of `PRESENT_AS_CAUSAL_CUT.md`, but it came from a
memory-retrieval observation rather than from distributed-systems theory:

> when a person loses a thought, one useful strategy can be to re-enact what they were
> doing, revisit the sequence/context, or let an internal rhythm come around again.

The biological interpretation remains open. The engineering analogue is immediate:

> **after an agent loses its current task context, do not assume the only recovery
> strategy is to replay a large completed transcript. Re-enter the live task with a small
> diagnostic procedure that reconstructs the current process state.**

Call that procedure a **context re-entry policy** or, informally, a **re-entry card**.

This is ordinary active diagnosis / state estimation. The possible product primitive is
how it is packaged into an agent runtime, not a new inference algorithm.

---

## 1. Why this is not the existing active-semantic work

`ACTIVE_SEMANTIC_IDENTIFICATION.md` asks:

```text
which probe best distinguishes hidden SOURCE RULES?
```

For example:

```text
world-time hazard
vs
event-distance hazard
```

`ACTIVE_PROBE_BUDGET_ARBITRATION.md` then showed that deliberate semantic exploration can
lose to ordinary risk-driven refresh when both consume the same scarce tool budget.

The present problem is different:

```text
the task semantics are already known
but the agent has lost its CURRENT PLACE in the task
```

Examples:

```text
was I still editing?
are the tests already running?
am I waiting on a remote operation?
is the patch actually ready to publish?
```

That is process-state recovery, not source-model discovery.

---

## 2. Connection to completed-transcript aliasing

`experiments/inflight_state_aliasing.py` already proves a narrower insufficiency result.

Two runtimes can expose the same:

```text
completed messages
cache ages
now
deadline
```

while differing in whether a useful refresh is already in flight. The optimal next
choice therefore differs although the model-visible completed history is identical.

That result says:

> **unfinished process state belongs in the observation when it changes the correct next
> action.**

A re-entry policy asks what to do when that explicit process-present state was never
stored, became stale, or was lost across an interruption/reset.

---

## 3. Known-answer gate

The implementation is:

```text
experiments/context_reentry_policy.py
```

Four balanced hidden runtime worlds expose exactly the same completed transcript:

```text
editing
    worktree dirty
    tests idle
    remote idle
    -> continue_edit

testing
    worktree dirty
    tests running
    remote idle
    -> wait_tests

waiting_remote
    worktree clean
    tests idle
    remote running
    -> wait_remote

complete
    worktree clean
    tests idle
    remote idle
    -> publish
```

The completed transcript alone cannot distinguish them.

Since all four require different actions, a deterministic transcript-only policy can be
correct in only one balanced world:

```text
accuracy = 0.25
```

---

## 4. Re-entry card

The task schema supplies a two-step decision tree:

```text
probe worktree
    |
    +-- dirty -> probe tests
    |              |
    |              +-- running -> wait_tests
    |              +-- idle    -> continue_edit
    |
    +-- clean -> probe remote
                   |
                   +-- running -> wait_remote
                   +-- idle    -> publish
```

The card contains no current answer.

It only stores **how to reacquire enough live evidence to reconstruct the answer**.

That distinction matters. A re-entry card is not a checkpoint containing hidden process
state. It is a compact recovery procedure that remains useful across many possible live
states of the same task.

---

## 5. Exact toy comparison

The finite state space lets us compute the comparison without sampling noise:

```text
completed transcript only
    decision accuracy    0.25
    live probes          0

full live snapshot
    decision accuracy    1.00
    live probes          3

stored re-entry policy
    decision accuracy    1.00
    live probes          2

two random distinct probes
    expected accuracy    0.75
    live probes          2

random probes until uniquely identified
    decision accuracy    1.00
    mean live probes     7/3 = 2.333...
```

These are known-answer properties of the constructed world, not an empirical agent
benchmark.

The useful systems observation is:

> **task structure can make reacquiring the present cheaper than dumping every live
> variable or replaying a large passive history.**

---

## 6. Re-entry is not just "more memory"

A conventional context-recovery strategy is:

```text
load more transcript
load previous summaries
load old tool outputs
```

That can still fail if the missing fact is about the live world rather than the recorded
past.

For example, no old transcript determines whether a test process that was launched before
an interruption is **still running now**.

So there are at least three recovery objects:

```text
PAST RECONSTRUCTION
    what happened before interruption?

PROCESS-PRESENT RECONSTRUCTION
    what is still happening / already completed now?

TASK RE-ENTRY PROCEDURE
    which cheap live checks recover enough state to continue safely?
```

A useful agent runtime should not collapse those into one giant chat history.

---

## 7. Why this resembles re-enactment without claiming a brain mechanism

The abstract commonality is:

```text
current readout insufficient
        |
        v
perform a structured sequence of state-changing / state-sampling operations
        |
        v
previously missing task-relevant state becomes available
```

In human memory that operation might involve context, imagery, movement, association,
attention, or internally generated sequence dynamics.

In a software agent it can be literal:

```text
open the current file
inspect the worktree
query the running job
look at the selected object
read the last failing test
check the remote operation
```

The analogy stops there. The software mechanism is explicit and inspectable; the neural
mechanism must be established experimentally.

---

## 8. Relation to the Process Present

The preferred case is still to preserve enough process-present state that re-entry is not
needed.

`PRESENT_AS_CAUSAL_CUT.md` proposes tracking:

```text
pending / arrived / consumed / cancelled / superseded
receiver frontiers
ETA / deadline
dependencies
```

But real systems lose context:

```text
agent handoff
model context reset
human interruption
process restart
UI navigation
session restore
compressed checkpoint
```

So the runtime can attach a recovery descriptor to a task type or task instance:

```text
REENTRY
    probes:
      - worktree
      - conditional: tests if dirty
      - conditional: remote if clean
    stop_when:
      action state uniquely identified
    expected_cost:
      2 probes
```

This is a complement to the process-present snapshot, not a replacement for it.

---

## 9. A product-shaped experiment

The toy is deliberately too easy. The next useful benchmark should use real-ish local
workflows and force interruptions.

Candidate tasks:

```text
edit -> test -> package -> publish
retrieve -> transform -> validate -> write
render -> inspect -> revise -> export
```

At random points:

```text
1. erase the agent's short-term task state;
2. retain the live filesystem/process/world;
3. keep a bounded completed-history summary;
4. resume under several recovery strategies.
```

Compare:

```text
A. passive history replay
B. full live-state dump
C. generic fixed probe list
D. task-specific re-entry card
E. adaptive active diagnosis
F. perfect checkpoint / oracle
```

Score:

```text
correct resumed action
number/cost of probes
time to safe continuation
duplicate actions
missed in-flight work
stale assumptions
context tokens consumed
```

The important attacker is **B**, the boring full snapshot.

If a re-entry card cannot beat a straightforward compact live-state dump on cost,
robustness, or context size, do not build a framework around it.

---

## 10. When order should and should not matter

The PresentMoment neural sanity check uses non-commuting transitions to demonstrate that
an exact trajectory can matter for observability.

Do **not** import that requirement into ordinary software re-entry without evidence.

In the current WidePresent toy, order matters only for **diagnostic efficiency**:

```text
worktree first
    tells us which second probe is useful
```

The world is not changed by reading it, and the same full set of probes eventually gives
the same answer in any order.

That is a more honest default for agent runtimes.

State-changing recovery actions can be studied separately if a real workflow needs them.

---

## 11. Current cross-repo synthesis

The pieces now line up without claiming they are one mechanism:

```text
PerceptionLab
    causal age:
    one current medium can contain unfinished consequences at different stages

PresentMoment
    accessibility:
    current state can contain receiver-null and receiver-potent dimensions

PresentMoment active re-entry
    controllability of accessibility:
    an available trajectory can change what becomes readable

WidePresent causal cut
    process state:
    unfinished work is part of an asynchronous agent's present

WidePresent re-entry card
    recovery:
    when that present is lost, reacquire it with a compact live procedure
```

The scalar phrase "wide present" is now almost too small for the object.

A more operational picture is:

> **current latent/process state + receiver-relative accessibility + causal maturity +
> available actions for changing or reacquiring what is accessible.**

---

## 12. Stop / go rule

Do not add another scheduler because this toy wins.

The next go condition is practical:

> **On interrupted local workflows, does a small re-entry card recover safe continuation
> with fewer tool calls/tokens than a full live snapshot and with fewer errors than
> passive history replay?**

If no, this remains a useful conceptual bridge and a tiny known-answer demo.

If yes, it becomes a plausible agent-runtime feature that can be tested with local
models later but does not depend on an LLM for its core value.
