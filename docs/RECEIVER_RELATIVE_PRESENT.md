# Receiver-relative present: retarded time and causal maturity

Date: 2026-08-12

This note is the AI-side consequence of the calibrated PerceptionLab wave-field result.

It makes one correction to our own language and then turns the correction into a small
runtime primitive.

The correction is:

> **An event does not literally acquire several objective ages at one global instant.
> What differs across space or modules is which source time has reached each receiver,
> and how mature that causal path currently is.**

That distinction lets `WidePresent`, `PresentMoment`, KYY and the PerceptionLab wave demo
fit together more cleanly.

---

## 1. One objective now, many retarded source times

Let a source emit state `u_s(t)` and let receiver `r` be reached after path delay
`d_(s,r)`.

Then the simplest receiver input is

```text
y_r(t) = u_s(t - d_(s,r))
```

The receiver is at the same wall-clock `t` as everything else, but the newest source
time available to it is

```text
F_(s,r)(t) = t - d_(s,r)
```

Call `F_(s,r)` the **source->receiver causal frontier**.

This is just retarded time / latency bookkeeping.

It is not a new physics claim and it is not a new distributed-systems idea.

The useful point is representational:

```text
global now                     one value
world age of event             one value
source->receiver frontier      receiver-specific
arrival age                    receiver-specific
in-flight / arrived status     receiver-specific
```

So the stronger statement is not

```text
the same event has several ages
```

but

```text
the same global present contains several receiver-relative maturity stages
```

---

## 2. Re-reading the PerceptionLab wave calibration

The PerceptionLab Wave Field uses finite propagation and spatially separated probes.

The known-answer calibration was:

```text
predicted adjacent lag ~= 13.2 frames
mean pair separation    = 10/6 gaps
predicted mean lag      ~= 22.0 frames
measured mean lag       ~= 21.3 .. 22.2 frames
```

That still says exactly what we wanted it to say.

But the clean mathematical reading is:

> a frozen propagating field is a spatial sampling of different **retarded source
> times / propagation stages**.

For a continuously changing source, position `x` samples approximately

```text
u(t - x/v)
```

For a single pulse, different positions are before, inside or after the passing
wavefront.

So `causal-age surface` is useful shorthand, but **retarded-time surface** is the more
precise object.

---

## 3. KYY supplied the semantic half of the same correction

KYY's `READOUT_FIBERS.md` separated

```text
what information exists in hidden state h
```

from

```text
what receiver R can currently read from h
```

and made semantic maturity receiver-relative.

PerceptionLab now gives the temporal analogue.

A system can have one physical/global state while:

```text
receiver A has already received / decoded an event
receiver B has the same consequence still in flight
receiver C sees a different phase or partial projection
```

Therefore there is no reason to demand one global computational maturity either.

A useful combined statement is:

> **meaning is receiver-relative, and causal maturity is receiver-relative, even when
> objective clock time is global.**

This is a much safer bridge between KYY's partial-maturity work and PresentMoment's
signals-in-flight picture than saying that the substrate itself supplies semantics.

---

## 4. Consequence for WidePresent

`bitemporal_present.py` correctly separates:

```text
world/event time
knowledge/arrival time
global now
source completeness / watermarks
```

That is enough for a single consumer.

For an asynchronous multi-module agent, there is another useful coordinate:

```text
receiver availability
```

Two modules can share the same event ledger and the same `now` while having different
causal access because their source->receiver paths differ.

The practical runtime state therefore becomes something like:

```text
GLOBAL
    now
    world/event time
    knowledge/arrival time

PER SOURCE -> RECEIVER PATH
    path delay / latency model
    causal frontier
    in-flight items
    arrival age
    deadline relation
```

This is not a replacement for bitemporality.

It is a receiver-relative layer on top of it.

---

## 5. The small implementation

`receiver_present.py` implements the fixed-delay case.

The core quantity is deliberately boring:

```text
path_frontier(source, receiver) = now_tick - path_delay(source, receiver)
```

An emitted event expands into one `TransitItem` per destination.

At any `now`, a receiver snapshot reports:

```text
arrived items
in-flight items
latest source tick that had enough time to arrive
world age
arrival age
path progress
```

The demo

```bash
python experiments/receiver_relative_present_demo.py
```

uses one source and three receivers:

```text
sensor -> reflex   2 ticks
sensor -> planner  8 ticks
sensor -> logger  15 ticks
```

At global tick 14, an event emitted at tick 10 has objective world age 4 for every
receiver.

But:

```text
reflex   : already arrived
planner  : half-way through path
logger   : early in path
```

The demo also gives the event a deadline. The fast receiver can act before it while the
planner cannot.

That is the simplest AI analogue of the PerceptionLab frozen wave snapshot.

---

## 6. Why this could matter for agents

Tool-using and streaming agents already contain many ordinary versions of this:

```text
tool request dispatched but not returned
sensor frame captured but not processed by every module
retrieval running while planner continues
sub-agent result pending
network message delayed
background computation partially complete
```

A text transcript often represents only completed messages.

The runtime, however, knows more:

```text
what is in flight
how long it has been in flight
which receiver is waiting for it
what deadline it can still meet
what source-time frontier that receiver has reached
```

Making that state explicit does not create new information.

It prevents each model/module from repeatedly reconstructing an asynchronous process
state from prose or event order.

This is the same philosophy as the original deterministic temporal kernel:

> **derive exact temporal/process facts in the runtime when the runtime already knows
> them.**

---

## 7. What would be worth testing later

No frontier benchmark is required to keep this note alive.

The local object is already usable.

If we later want a representation test, it can be done without a paid API:

```text
same synthetic asynchronous episodes
same small/local model

A. raw request/response history
B. raw history + global ages
C. raw history + receiver-relative frontier / in-flight state
```

The target should be an operational decision such as:

```text
act now
wait for an already-pending result
launch a new request
ignore a late superseded result
```

If a small model gains nothing from C once history is matched, then the receiver-state
rendering is convenience only.

That is an acceptable result.

---

## 8. Brain-side implication

`PresentMoment` should keep the same distinction.

The biological claim is not that a heartbeat, breath or cortical wave somehow creates
several wall-clock times.

The interesting candidate is:

```text
one objective instant
+ many physical paths
+ many path delays / phases / ringdowns
+ many receiver-specific readouts
= a distributed set of causal maturity stages
```

Recent experiments make the ingredients increasingly concrete:

- human intracranial recordings show that theta/alpha travelling-wave direction changes
  with memory encoding versus recall (Mohan et al., 2024, PMID 38459263);
- direct human recordings show planar, spiral, source/sink and other travelling-wave
  patterns that covary with memory-related behavior (Das et al., 2026,
  PMID 41963323, DOI 10.1038/s41467-026-71386-z);
- externally imposed travelling-wave stimulation can directionally alter neural timing
  and cognitive performance (PNAS 2026, PMID 42085148);
- human intracranial recordings show widespread forebrain synchronization to breathing,
  including entrainment under external mechanical ventilation (Mowla et al., 2026,
  PMID 42209532, DOI 10.1038/s41467-026-73828-0);
- vagal PIEZO2 cardiac mechanoreceptors show blood-volume-dependent activity with every
  heartbeat, time-locked to systole (Liu et al., Nature 2026,
  DOI 10.1038/s41586-025-10010-4).

None of that proves a `wide present`.

It does make **receiver-relative causal state** a better biological target than an
imagined single temporal strip.

---

## 9. Current synthesis

The cross-repo picture is now:

```text
WidePresent
    exact world/knowledge time
    + receiver-relative causal frontiers

PerceptionLab
    calibrated propagation / lag fixture

KYY
    receiver-relative readout and partial maturity

PresentMoment
    physical signals in flight + cyclic/ringdown body loops
```

The sentence I would keep is:

> **A wide present need not mean a wider clock instant. It can mean one clock instant
> containing many receiver-relative causal frontiers and unfinished processes.**

That is sharper than `statefulness = memory`, and practical enough to turn into runtime
state rather than another speculative neuron.
