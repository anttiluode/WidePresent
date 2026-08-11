# Active temporal refresh: useful, but ordinary

Date: 2026-08-11

The offline-mode attack demoted the sleep-inspired branch. A fresh diagnostic query
while remaining online performed at least as well as the same query after entering a
quiet state.

That leaves a simpler candidate operation:

> **refresh NOW when the passive temporal state is too uncertain.**

This note attacks that idea with cheap scheduling controls.

## Experiment

`experiments/adaptive_refresh_budget.py`

The benchmark is the same hard delayed-evidence setting as
`provenance_vs_offline_attack.py`:

- three hidden states;
- one late state change;
- noisy sensor observations;
- long-tailed delivery delays;
- old-world observations can arrive near the final decision.

A refresh consists of three noisy current-state observations near the decision. The
probe is intentionally strong and conventional. We ask only **which episodes should
pay for it**.

Three policies are compared at matched approximate budgets:

```text
random
    refresh random episodes

stale
    refresh episodes whose most recent received evidence is oldest

uncertain
    refresh episodes where the passive classifier is least confident
```

The staleness threshold and uncertainty threshold are chosen on training data only.

## Five-seed exploratory result

`7000` paired episodes per seed, `4500` training episodes.

No refresh:

- accuracy `0.704`

Always refresh:

- accuracy `0.976`

### About 10% refresh budget

| policy | accuracy |
|---|---:|
| random | 0.733 |
| stale | 0.749 |
| uncertain | **0.755** |

### About 25% refresh budget

| policy | accuracy |
|---|---:|
| random | 0.772 |
| stale | 0.803 |
| uncertain | **0.822** |

### About 50% refresh budget

| policy | accuracy |
|---|---:|
| random | 0.838 |
| stale | 0.869 |
| uncertain | **0.904** |

The training-derived thresholds produced test query fractions close to their target
budgets.

## Interpretation

There is a real value to deciding **when the current temporal state is unreliable**.

But nothing here requires a new neural mechanism.

The ordering is exactly what ordinary active sensing intuition would predict:

```text
random refresh
    < simple freshness/staleness policy
    < uncertainty-triggered refresh
    < always refresh
```

So WidePresent should not claim active refresh as novel.

The useful architectural lesson is instead one of composition:

```text
passive temporal state
        |
estimate NOW + uncertainty
        |
uncertainty / staleness exceeds cost-dependent threshold
        |
request fresh evidence
        |
update NOW
```

## What this does to the project

The research path has become more disciplined:

1. **Clock arithmetic** belongs in a deterministic temporal kernel.
2. **World/valid-time provenance** helps when delivery is delayed.
3. **Arrival time as a second clock** is task-dependent, not automatically useful.
4. **Active refresh** is valuable when stale evidence leaves NOW ambiguous.
5. **Quiet/offline mode** is not currently justified.
6. **Wave / boundary / WidePresent geometry** must beat this boring stack if it is to
   earn architectural status.

That boring stack is now a serious baseline:

> **temporal kernel + valid-time provenance + uncertainty + active refresh**

## Next attack

The next important comparison is no longer sleep versus wake.

It is:

> **Can the lake/boundary dynamical substrate estimate current state or uncertainty
> better than an ordinary timestamped event ledger/filter with the same active-refresh
> budget?**

If not, the wave branch remains a beautiful physical implementation metaphor rather
than a useful AI architecture.
