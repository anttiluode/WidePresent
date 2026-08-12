# The present as a causal cut: unfinished work is state

Date: 2026-08-12

This note follows the PerceptionLab wave-field calibration and
`RECEIVER_RELATIVE_PRESENT.md` one step further.

It also collides directly with old distributed-systems theory, so the novelty boundary
should be stated first.

---

## 1. Prior art owns the computer-science skeleton

Lamport's 1978 causal-ordering paper showed that events in a distributed computation
carry a natural partial order rather than requiring one physically privileged global
sequence:

- Leslie Lamport, *Time, Clocks, and the Ordering of Events in a Distributed System*,
  Communications of the ACM 21(7), 1978. DOI: `10.1145/359545.359563`.

Chandy and Lamport then showed how to determine a consistent global state of an
asynchronous distributed computation without stopping it:

- K. Mani Chandy and Leslie Lamport, *Distributed Snapshots: Determining Global States
  of Distributed Systems*, ACM TOCS 3(1), 1985. DOI: `10.1145/214451.214456`.

The important conceptual piece for this repository is that a distributed snapshot is
not merely a list of local process variables. **Channel state / messages in transit are
part of the state that matters.**

Modern event-time stream processors likewise use per-stream progress frontiers called
watermarks. Multi-input operators cannot safely advance merely because wall-clock time
advanced; their effective event-time progress is constrained by incoming progress.

So none of the following is claimed as a new distributed-systems abstraction.

---

## 2. The useful correction to the old WidePresent picture

The early WidePresent image was approximately:

```text
past ---------------- [ NOW ] ---------------- future
```

and later:

```text
world time
knowledge time
```

Those remain useful coordinates.

But an asynchronous machine also has a causal structure like:

```text
source A ---- event -----> module X ---- result -----> planner
                    \
                     \-----------> reflex

source B -------------------------------> planner
```

At one wall-clock instant:

```text
reflex may already have A
planner may still be waiting for A-derived work
B may have arrived at planner
another tool call may be in flight
```

There is no reason for the internal system to occupy one common maturity index.

A more accurate operational `present` is a **cut through this causal graph**:

```text
completed / integrated         |       unfinished / in flight
-------------------------------|------------------------------
receiver A has event e         |  e -> receiver B
module C produced result q     |  tool request r
                               |  sub-agent computation z
```

The wall-clock `now` is still one scalar.
The causal cut is additional state.

---

## 3. Why the PerceptionLab wave result matters here

The wave-field demo is a continuous toy version of the same idea.

One pulse propagates through a medium. At one global clock instant, spatially separated
probes sit at different positions relative to the passing consequence.

The calibrated lag result does not discover distributed causality. It gives us a visual
known-answer fixture in which **the in-flight part of state cannot be ignored without
throwing away the mechanism that creates the delay.**

That is why the living-checkerboard negative control mattered too:

```text
persistent state + no differentiated propagation
    -> little receiver-relative maturity structure

propagating state
    -> explicit crossing / in-flight structure
```

So the cleaner cross-repo criterion is now:

> **Temporal thickness requires more than state persistence. For an asynchronous system,
> ask what causal work crosses the current cut and which receivers have integrated which
> consequences.**

---

## 4. The key AI failure mode: completed-transcript aliasing

A language-model agent is often shown a sequence of completed messages:

```text
user
assistant
completed tool result
assistant
...
```

But a real runtime may simultaneously contain:

```text
request already dispatched
retrieval still running
sub-agent still computing
stream frame queued for one module
result produced but not yet consumed downstream
old request superseded by a newer one
```

If those variables are omitted from the model observation, two physically different
runtime states can collapse to the **same completed transcript**.

That is not merely a hard reasoning problem.
It is observation aliasing.

`experiments/inflight_state_aliasing.py` builds the smallest possible known-answer case.

Both worlds expose exactly the same:

```text
now
cache age
knowledge age
completed tool history
deadline
```

but hidden process state differs:

```text
WORLD A
    no refresh is pending
    -> launch a refresh

WORLD B
    refresh already pending and arrives before deadline
    -> wait
```

The optimal actions differ.

Therefore any deterministic policy that sees only the identical completed-state view
must choose the same action in both worlds and cannot be optimal in both.

This is the important result:

> **No amount of model intelligence can reconstruct a runtime distinction that the
> observation function has erased.**

A frontier/pending-state representation is not helping the model infer better.
It is restoring missing state.

---

## 5. This changes what the cheap/local benchmark should test

The earlier language-tool benchmark asks whether derived ages are more usable than raw
timestamps.

That remains a valid representation question but requires an LLM to be interesting.

The in-flight result gives a more primitive gate that needs no API and no frontier model:

```text
Gate P0: sufficiency
    Can completed history + ages identify the correct action?

If two matched worlds alias under that observation and require different actions:
    NO, by construction.

Gate P1: runtime representation
    Expose pending / in-flight / superseded / deadline state explicitly.
    Does a small local policy or model now solve a broader noisy family reliably?

Gate P2: compression
    How little of the process graph must be rendered to preserve the useful decisions?
```

P0 is already answered by the toy.
P1/P2 are engineering/product questions, not consciousness questions.

---

## 6. A practical `Process Present`

The useful artifact is beginning to look less like a temporal neuron and more like an
agent-runtime sidecar:

```text
OBJECTIVE CLOCK
    now

FACT LEDGER
    world time
    knowledge time
    source watermark / completeness

PROCESS CUT
    operation id
    source
    intended receiver
    started at
    pending / arrived / consumed / cancelled / superseded
    ETA or latency distribution
    deadline
    dependencies

RECEIVER FRONTIERS
    latest source time available to each receiver
    local completeness frontier

FUTURE COMMITMENTS
    scheduled checks
    predictions due
    deadlines
```

The model does not need every raw internal event.
The runtime can render a compact projection such as:

```text
PENDING:
- weather_refresh #31 -> planner, age 1.8s, ETA ~0.7s, deadline 4.0s
- retrieval #44 -> writer, age 0.4s, ETA unknown

SUPERSEDED:
- weather_refresh #28 by #31

FRONTIERS:
- planner/weather complete through 11:42:08
- reflex/sensor complete through 11:42:11
```

That is much closer to something one could actually put into an agent loop.

---

## 7. Watermarks and receiver frontiers should compose

WidePresent already tracks source/event-time completeness with watermarks.

`receiver_present.py` now tracks a simple fixed-delay source->receiver frontier:

```text
transport_frontier(s, r, t) = t - delay(s, r)
```

If source `s` itself is only known complete through watermark `W_s`, then a conservative
receiver completeness boundary is naturally constrained by both:

```text
effective_frontier(s, r, t)
    = min(W_s(t), transport_frontier(s, r, t), processing_frontier(r, t), ...)
```

This is standard stream-processing logic in new clothing, and it should stay boring.

But it gives the AI runtime a useful answer to a question that plain timestamps do not:

> **Up to what source time may this particular receiver safely treat silence as complete
> evidence rather than merely not-yet-arrived evidence?**

That is precisely the distinction WidePresent has been circling since the watermark
work.

---

## 8. Relation to KYY partial maturity

KYY found it useful to separate hidden motion from what a particular receiver can read.

The causal-cut picture adds:

```text
receiver A may have a stable useful variable now
while receiver B's contributing computation is still unfinished
```

This is not paradoxical and does not require a globally synchronized representational
state.

So KYY's `partial maturity` can be stated without geometric mystique:

> **Different receivers may cross their own useful decision frontiers before the whole
> distributed computation reaches a common barrier.**

Whether local geometry helps that under real physical cost is KYY's separate question.

---

## 9. Relation to PresentMoment / brain

For biology, the causal-cut language is only an analysis metaphor unless the relevant
paths and receivers are identified experimentally.

Still, it is cleaner than imagining one giant neural frame.

At one instant an organism can contain:

```text
current local neural activity
signals travelling on axons
cardiovascular consequences propagating through mechanics/circulation
heartbeat-linked afference returning
respiratory phase at several neural receivers
slow endocrine consequences not yet relaxed
```

A receiver-specific brain population only has access to the subset that has reached and
is legible to it.

So the biological question becomes:

> **Does cognition exploit the geometry of these receiver-relative frontiers, or are
> ordinary recurrent neural states sufficient to explain the same behavior?**

That is still open.

---

## 10. Current stop / go rule

Do **not** invent another temporal neural layer from this note.

The immediate useful direction is runtime engineering:

```text
track unfinished work
track local completeness
track supersession
derive deadline relations
render a compact process-present state
```

Then attach a tiny local model only if a decision actually requires learning.

The sentence worth carrying forward is:

> **For an asynchronous agent, the present is not only what has happened. It also
> includes the state of what has started happening but has not finished arriving.**

That statement is not new distributed-systems theory.
It is a design requirement for an agent observation if unfinished work changes the
correct next action.
