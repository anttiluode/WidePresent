# Shared tool-budget arbitration for semantic probing

Date: 2026-08-11

The free-probe experiment in `ACTIVE_SEMANTIC_IDENTIFICATION.md` looked promising:

```text
probe where temporal semantic models disagree
-> identify source semantics quickly
-> improve later OOD reuse/refresh decisions
```

But those diagnostic probes were effectively free.

A real tool-using agent usually has one shared budget:

```text
refresh the current risky request
or
spend the call on a request that is especially informative about source semantics
```

This attack forces those goals to compete.

The experiment is:

`experiments/active_probe_budget_arbitration.py`

The main result is negative for dedicated semantic exploration:

> **When diagnostic probes consume the same scarce tool calls used to protect current requests, myopic refresh-by-immediate-risk outperforms deliberate semantic exploration over the tested horizons. The operationally risky cases already provide enough incidental semantic information that dedicated exploration learns faster but does not repay its immediate cost.**

This substantially demotes active semantic information gain as a default runtime feature.

## 1. Shared-budget world

Each round presents:

```text
6 cached requests
```

with different:

```text
world ages
event ages
```

The source obeys one hidden stable rule:

```text
world_hazard
or
event_hazard
```

The runtime starts from `200` exactly confounded audits at:

```text
1 second / event
```

so the initial semantic posterior is unresolved.

Only **one** request per round may use the tool.

### Refreshed request

Gets immediate safe refresh utility:

```text
+0.55
```

and reveals whether its old cache was actually valid, producing a semantic audit.

### Five unrefreshed requests

Must reuse their cache in this deliberately harsh fixed-budget benchmark.

Their expected utility is:

```text
valid reuse   +1.00
stale reuse   -1.50
```

So choosing which single request to refresh has immediate and learning consequences.

## 2. Policies

### `risk`

Choose the request with the greatest predicted immediate utility advantage of refresh over reuse.

No explicit exploration bonus.

### `risk_info_tie`

Find requests within `0.05` utility of the best immediate-risk candidate, then use semantic information gain only as a tie-breaker.

This asks whether semantic exploration can be obtained nearly for free among operationally equivalent choices.

### `info_until_confident`

Maximize semantic information gain until one semantic model reaches posterior `>=0.90`, then switch permanently to immediate risk.

### `disagreement_until_confident`

Same explore-then-exploit schedule using cheap semantic disagreement rather than exact information gain.

### `oracle`

Knows the hidden semantic class only for choosing which current request has the greatest true immediate need for refresh.

It is an upper immediate-allocation reference, not a deployable policy.

## 3. Twenty rounds

Twenty seeds for each hidden semantic (`40` runs/policy).

| policy | mean utility | bad reuse | true semantic weight | >.90 identified | deliberate exploration rounds |
|---|---:|---:|---:|---:|---:|
| **risk** | **0.4923** | 0.4327 | 0.785 | 0.425 | 0 |
| risk + info tie-break | 0.4922 | 0.4327 | 0.798 | 0.350 | 0 |
| information until confident | 0.4658 | 0.4581 | **0.838** | 0.525 | 15.2 |
| disagreement until confident | 0.4742 | 0.4490 | **0.840** | 0.525 | 15.1 |
| oracle immediate risk | 0.4957 | **0.4312** | 0.778 | 0.300 | 0 |

Semantic exploration learns faster.

But it refreshes requests that are less immediately dangerous while leaving more dangerous requests forced to reuse.

The learning gain does not repay that current utility loss.

## 4. Forty rounds

| policy | utility | bad reuse | true semantic weight | identified |
|---|---:|---:|---:|---:|
| **risk** | **0.4924** | 0.4303 | 0.866 | 0.725 |
| risk + info tie-break | 0.4922 | 0.4303 | 0.862 | 0.700 |
| information then risk | 0.4746 | 0.4475 | **0.929** | 0.800 |
| disagreement then risk | 0.4798 | 0.4419 | 0.908 | **0.825** |
| oracle immediate risk | 0.4952 | **0.4292** | 0.867 | 0.675 |

By this point immediate-risk refreshes have already learned a lot about the temporal rule *incidentally*.

The information policies remain semantically ahead but operationally behind.

## 5. Eighty rounds

| policy | utility | bad reuse | true semantic weight | identified |
|---|---:|---:|---:|---:|
| **risk** | **0.4879** | 0.4398 | 0.930 | 0.875 |
| risk + info tie-break | 0.4876 | 0.4398 | 0.951 | 0.900 |
| information then risk | 0.4778 | 0.4498 | **0.984** | **0.975** |
| disagreement then risk | 0.4808 | 0.4465 | 0.981 | **0.975** |
| oracle immediate risk | 0.4896 | **0.4392** | 0.946 | 0.900 |

Now deliberate exploration has largely solved the semantic problem.

But the myopic policy has almost solved it too, without paying as much exploration cost.

## 6. 160 rounds

| policy | utility | bad reuse | true semantic weight | identified |
|---|---:|---:|---:|---:|
| **risk** | **0.4881** | 0.4417 | 0.972 | 0.950 |
| risk + info tie-break | 0.4877 | 0.4417 | 0.970 | 0.925 |
| information then risk | 0.4831 | 0.4468 | **0.999** | **1.000** |
| disagreement then risk | 0.4847 | 0.4451 | **0.999** | **1.000** |
| oracle immediate risk | 0.4892 | **0.4414** | 0.986 | 0.975 |

The exploration cost is shrinking as a fraction of the horizon, but the ordinary risk policy has now learned the semantic rule well enough that little future advantage remains to exploit.

## 7. 320 rounds

| policy | utility | bad reuse | true semantic weight | identified |
|---|---:|---:|---:|---:|
| **risk** | **0.4900** | 0.4402 | 0.977 | 0.975 |
| risk + info tie-break | 0.4896 | 0.4403 | 0.984 | 0.975 |
| information then risk | 0.4878 | 0.4427 | **1.000** | **1.000** |
| disagreement then risk | 0.4885 | 0.4418 | **1.000** | **1.000** |
| oracle immediate risk | 0.4908 | **0.4400** | **1.000** | **1.000** |

Even after heavy amortization, dedicated semantic exploration does not overtake myopic operational risk in this benchmark.

## 8. Why the free-probe result did not survive the shared budget

The free-probe benchmark showed that information gain and semantic disagreement are excellent **experiments**.

This benchmark asks whether those experiments are worth stealing scarce tool calls from current decisions.

The answer here is mostly no.

The key reason is:

> **Operationally risky cases are already informative enough.**

A request that the model thinks may be stale is often located in a region of the `(world age, event age)` plane where the candidate temporal rules also make distinguishable predictions.

So ordinary myopic refresh produces useful semantic labels as a side effect.

The exploration policy buys cleaner/faster identification, but much of that information is redundant with what future risk-driven refreshes would have learned anyway.

## 9. The tie-breaker also fails to earn a strong claim

`risk_info_tie` tries to use semantic information only when candidates are within `0.05` immediate utility of the best risk choice.

Its utility is essentially identical to pure risk, which is good.

But its semantic-identification advantage is small and inconsistent across horizons.

So there is no strong evidence here for adding even a semantic-information tie-breaker as a default.

It remains a cheap optional heuristic, not a core runtime requirement.

## 10. Important limitation

The benchmark deliberately forces all unselected requests to reuse their cache.

A production agent might instead:

```text
abstain
defer
batch requests
borrow future budget
use a cheaper secondary source
```

Those options could change the exploration economics.

Likewise, dedicated semantic probing can still make sense when:

- probes are cheap or free;
- source semantics affect a very long future horizon;
- ordinary risky requests do not naturally span the informative age-plane directions;
- stale reuse is catastrophic enough to justify explicit experimentation;
- one diagnostic call can govern a large population of later decisions.

The current result is narrower:

> **Do not add semantic-information probing merely because it identifies the model faster. Compare it against the information already obtained from ordinary risk-driven refresh under the same tool budget.**

## 11. Revised active-refresh principle

The older result was:

```text
uncertainty-triggered refresh > stale heuristic > random
```

The semantic work refines that to:

```text
CURRENT DECISION
    refresh according to expected operational risk

SEMANTIC LEARNING
    learn opportunistically from those refresh outcomes

DEDICATED DIAGNOSTIC PROBE
    require a separate value-of-information case
    do not enable by default
```

That is a considerably simpler product rule.

## 12. Current practical runtime after this kill

The default architecture is now almost disappointingly conventional:

```text
evidence coordinates
    world age
    known/arrival age
    event age
    version/invalidation
        |
        v
source semantic posterior
    rolling if source can drift
        |
        v
P(valid now)
        |
        v
expected utility of reuse
vs
refresh cost
        |
        v
refresh highest-risk items under budget
        |
        v
audit refresh outcomes
        |
        v
update semantic posterior
```

No dedicated exploration controller is currently justified.

## 13. Where to dig next

The next useful attack should no longer invent another scheduler.

The strongest unresolved question is whether the **language model itself benefits from receiving this runtime state**.

We already have:

- known-contract language benchmark;
- causal counterfactual pairs;
- narrow-vs-wide semantics-discovery prompts;
- calendar-shift metamorphic controls.

Those are ready for genuine model sampling when a model endpoint is available.

Until then, the systems side has reached a coherent stopping point:

> **preserve the right coordinates, learn source validity semantics with uncertainty and forgetting, refresh by operational utility, and treat semantic learning as a side effect unless active experimentation demonstrably pays for itself.**
