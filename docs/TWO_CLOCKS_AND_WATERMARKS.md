# Two clocks, late evidence, and the shape of a real present

Date: 2026-08-11

This note records a correction to WidePresent v0.1.

## The one-dimensional chain is not enough

The first sketch assumed that when an observation enters the agent, it belongs to `now`.

That silently equates two times:

```text
when the event happened == when the agent learned about it
```

Real online systems violate this constantly.

A camera frame may be delayed by inference. A network packet can arrive late. A laboratory result describes a sample taken yesterday. A user can say today, "I fell off the bicycle on Sunday." A retrieved memory can become available now while referring to an event years ago.

So a serious present needs at least two time coordinates.

## World time and knowledge time

Borrowing established temporal-database language:

- **valid / event time** — when a fact/event is true in the modeled world;
- **transaction / knowledge time** — when the agent/system acquired or recorded that fact.

Call them

```text
t_world
t_known
```

and keep a separate moving reference

```text
t_now
```

Then an item has at least

```text
(value, t_world, t_known, source, uncertainty)
```

The two useful relative coordinates are

```text
world_age     = t_now - t_world
knowledge_age = t_now - t_known
```

A fact learned immediately has nearly equal ages. A late observation can have large `world_age` but near-zero `knowledge_age`.

## Prediction fits naturally

A forecast made now for a future time is simply a belief whose valid/world time lies ahead of its knowledge/creation time:

```text
t_known = now
t_world = now + horizon
type    = prediction
```

When `t_world` reaches `now`, the prediction is due. But the confirming observation may still arrive later:

```text
prediction created       target world time        evidence arrives
      |                         |                        |
      v                         v                        v
---- t_known ---------------- t_world ---------------- t_arrival ---->
```

So the old "prediction rendezvous" picture survives, but it is better understood as a special case of bitemporal belief revision than as a new primitive.

## Watermarks: the missing epistemic variable

Stream-processing systems such as Apache Flink use **watermarks** to represent progress in event time. A watermark says, approximately, that events older than a certain event timestamp are no longer expected to arrive.

That suggests a useful AI state variable:

> **How complete do I believe my evidence is for each region of recent world time?**

Without this, the agent can confuse:

```text
no event occurred
```

with

```text
no event has arrived yet
```

Those are not the same epistemic state.

A WidePresent slot should therefore eventually carry something like

```text
state[t]
observation_mask[t]
completeness[t]
uncertainty[t]
```

not merely a value.

## This is heavily prior-art constrained

None of the two-clock machinery is new:

- temporal databases have formalized valid time and transaction time for decades;
- stream processors distinguish event time, ingestion time and processing time and explicitly handle late/out-of-order data with watermarks;
- target tracking and control have a large out-of-sequence-measurement literature, including Kalman/filtering/smoothing methods;
- a 2026 L4DC paper studies model-based RL under random observation delays and reports that naive stacking of past observations is insufficient;
- 2026 work already applies bitemporal stores to conversational-agent memory.

Therefore WidePresent cannot claim "two times for AI" as new.

## What may still be worth testing

The interesting synthesis is to make these data-engineering / estimation semantics **part of the learned agent state**, not merely infrastructure around a neural model.

Candidate architecture:

```text
                     knowledge time
                          ^
                          |
late evidence   o        |        o newly learned old fact
                \        |       /
                 \       |      /
------------------\------NOW--------------------> world time
 past world state  \      |      future predictions
                    \     |
                     o    |
```

The dense 2-D sheet is probably wasteful. A sparse bitemporal ledger can be projected into a fixed-width working present around `t_now`.

That working present would distinguish:

1. observed-and-complete past;
2. observed-but-still-revisable past;
3. unobserved/missing intervals;
4. present observations;
5. future predictions with deadlines;
6. predictions whose target time passed but whose validating evidence is still outstanding.

This is much richer than "remember the last N tokens."

## New falsifiable task: late-evidence source confusion

Generate a dynamical world with two asynchronous sensors. Events carry correct world timestamps but incur random communication delays and can arrive out of order.

Ask the agent online for:

- current-state estimate;
- whether a recent interval is truly empty vs merely incomplete;
- forecast calibration at fixed future world times.

Compare:

- arrival-order RNN;
- timestamp-aware RNN;
- timestamp Transformer;
- delay-aware model-based filter / OOSM control;
- bitemporal WidePresent projection.

**Kill condition:** if a standard delay-aware filter or timestamped model matches the bitemporal projection, the neural WidePresent machinery is unnecessary for this capability.

## Consequence for the original idea

The bicycle remains a useful intuition for a **moving reference point**, but the research object is no longer a literal ring.

A better phrase is:

> **WidePresent is a temporally typed working state around a moving now.**

Time is not one scalar tag. At minimum the agent needs to know *when the world state belongs* and *when that belief entered the agent*.
