# Architecture pivot — test a temporal kernel before inventing a temporal neuron

Date: 2026-08-11

This note follows the prior-art dig into LLM temporal blindness, delayed observations, stream processing and bitemporal systems.

## A simpler possibility

The first WidePresent picture treated time as something a neural architecture should represent internally.

That may be unnecessarily difficult.

Elapsed objective time is not a latent variable when the runtime already has a clock. Message arrival times, tool-call times, sensor timestamps and deadlines can also be known exactly. A learned model need not reconstruct these quantities from token statistics.

So the strongest baseline for WidePresent is now:

> **Do the temporal bookkeeping deterministically outside the model and expose the derived temporal state at every decision.**

Call that component the **temporal kernel**.

This is not claimed as a new computer-systems idea. It is a deliberately boring baseline.

## What the kernel knows

For every relevant fact/event/belief:

```text
world_time       when the represented event belongs in the world
knowledge_time   when the agent acquired the information
source           where it came from
kind             observation / memory / prediction / action / deadline
uncertainty      confidence or measurement uncertainty
expiry           optional validity / freshness policy
```

At decision time it derives, rather than asking the language model to calculate:

```text
world_age        now - world_time
knowledge_age    now - knowledge_time
freshness        policy(world_age, source, expiry)
due_in           target_time - now
lateness         max(now - deadline, 0)
completeness     whether late evidence may still arrive for a world-time interval
```

These values change when time passes even if no new token arrives.

## Why this matters for the original failure mode

A stationary dialogue transcript does not itself change while the user is away. If the model is invoked hours later with essentially the same token history, the temporal meaning of old tool results and old statements may have changed.

There are three possible fixes, and WidePresent should compare all three:

```text
A. passive timestamp text
   "[2026-08-11 14:03] ..."

B. derived temporal state
   age=7182s, stale=true, deadline_in=-42s

C. learned moving temporal representation
   WidePresent / recurrent time-aware state
```

If B matches or beats C, then a learned temporal substrate is unnecessary for that task.

## Important direct prior art

Two recent LLM papers make this comparison urgent.

- **Wang et al., EMNLP Findings 2025, _Discrete Minds in a Continuous World_** show that LLMs can infer some wall-clock duration from token counts and can adapt under time pressure. This means token sequence can contain a weak proxy for physical time.
- **Cheng et al., ACL Findings 2026, _Your LLM Agents are Temporally Blind_** show a different side: multi-turn agents poorly align tool-use decisions with real elapsed time, and explicit timestamp augmentation alone leaves normalized alignment below 65% for all tested models on TicToc.

So the interesting distinction is no longer "time versus no time." It is:

> **proxy time vs passive timestamp information vs continuously maintained temporal state.**

## Architectural options after the kernel

Only if deterministic temporal state is insufficient should we test richer mechanisms.

### Option 1 — state side-channel

At each inference, concatenate a small structured temporal vector to the agent/model state. No ring, no oscillator.

### Option 2 — fixed-width working projection

Project the bitemporal ledger around `now` into rows indexed by world-time offset. Include observation mask, knowledge age, completeness and forecast channels.

### Option 3 — learned continuous memory

Use LMU/HiPPO, ODE/CDE, SSM or another continuous-time state model.

### Option 4 — cyclic / phase machinery

Only revisit bicycle rings, theta-like phase, Visertäjä-style oscillators or KYY-style reciprocal geometry if a specific failure of Options 1–3 calls for them.

## Project discipline

The temporal kernel is intentionally dangerous to the original idea.

If it wins, the correct result is:

> **Agents benefit from explicit temporal state, but no special WidePresent neural architecture was needed.**

That would still be a useful engineering outcome and a cleaner explanation of the motivating failure than a consciousness story.
