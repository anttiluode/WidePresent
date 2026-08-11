# Lake versus ledger attack — the wave mostly dies

Date: 2026-08-11

This is the next adversarial test after `docs/ACTIVE_REFRESH_ATTACK.md`.

The question was deliberately simple:

> **Can a fixed local lake/wave state represent the current world, or know when its
> estimate is unreliable, better than a boring timestamped temporal state under the
> same information and state budget?**

The answer on this benchmark is **no**.

A slow diffusive field performs well, but an ordinary exponential filter bank explains
that performance. The oscillatory wave component adds essentially nothing.

## 1. Benchmark

`experiments/lake_vs_ledger_attack.py`

Three hidden states are possible. The environment changes once late in an episode,
with the switch time sampled uniformly from `t=80..94`. The decision is at `t=100`.

Noisy observations are generated continuously. Every observation has:

```text
value / label
valid or world time       when the observation was true
delivery or arrival time  when the agent received it
```

Delivery has a long-tailed delay distribution, so evidence generated under the old
world can arrive near the final decision.

Only observations delivered by `t=100` are available to any representation.

The task is to classify the **current** hidden state.

## 2. Fairness constraints

All learned readouts use the same standardized multinomial logistic regression.

The main state budget is `60` real numbers.

The lake gets a generous concession: it receives the same valid/world timestamp as the
ledger. A late event is therefore **valid-time compensated** -- its dynamical
contribution is aged according to its valid time rather than pretending that the event
was generated when it arrived.

So this attack does not handicap the lake by hiding temporal provenance.

## 3. Representations

### `arrival_hist` — 60 numbers

`3 labels x 20 arrival-age bins`.

This is the sanity baseline that knows only when evidence arrived.

### `valid_hist` — 60 numbers

`3 labels x 20 valid/world-age bins`.

A deterministic timestamped ledger projection.

### `exp_bank` — 60 numbers

`3 labels x 20 exponential valid-time kernels`.

This is the strongest boring baseline in the attack. It is a fixed multi-timescale
fading-memory filter bank.

### `wave_only` — 60 numbers

A fixed local 30-node spring ring. The state is all node displacements and velocities:

```text
30 x displacement
30 x velocity
```

Each label enters through a different local port. The graph is fixed. The damping
constant was chosen on a separate development split and then frozen.

### `wave_oil` — 60 numbers

A fixed local 20-node spring ring with three state variables per node:

```text
20 x displacement
20 x velocity
20 x slow diffusion / "oil"
```

The slow field obeys a graph diffusion/decay law. Its constants were chosen on the same
separate development split, not on the five reported evaluation seeds.

### `oil_only` — 20 numbers

The slow diffusion field alone. It deliberately under-uses the 60-number budget.

### `modal_exp20` — 20 numbers

The exact non-geometric modal form of `oil_only`.

This representation is important because it exposes what the oil field actually is.

## 4. Separate development result

A small damping/timescale sweep was done only on development seeds 10/11 before the
five evaluation seeds below.

The initial lightly damped wave scored only about `0.46`. Selecting a sensible damping
raised pure-wave validation accuracy to about `0.68`, still well below the ledger.

Adding the slow field raised the 60-number `wave_oil` state to about `0.79`, level with
the timestamped baselines.

Then the key ablation:

```text
oil_only   ~0.791
wave_oil   ~0.791
```

The wave was no longer doing visible work.

## 5. Five-seed evaluation

Training is IID only: `4000` episodes per seed.

Each reported regime has `1800` untouched test episodes per seed.

### Passive current-state accuracy

| representation | IID | sparse-rate OOD | dense-rate OOD | long-delay OOD |
|---|---:|---:|---:|---:|
| arrival histogram | 0.757 | 0.637 | 0.818 | 0.643 |
| valid-time histogram | 0.779 | 0.661 | 0.851 | 0.695 |
| **exponential bank** | **0.786** | **0.659** | **0.868** | **0.701** |
| wave only | 0.683 | 0.551 | 0.793 | 0.576 |
| oil only | 0.785 | 0.659 | 0.863 | 0.701 |
| wave + oil | 0.783 | 0.658 | 0.865 | 0.697 |

The main result is not subtle:

> **pure oscillatory wave state loses in every regime.**

The slow field recovers essentially all of the performance, but the ordinary
exponential bank matches it.

## 6. Matched 25% active refresh

Each representation ranks test episodes by its own classifier uncertainty. Exactly
25% of episodes receive the same three fresh noisy observations at world times
`96, 97, 98`.

This gives every system the same refresh budget and tests whether its uncertainty is
especially useful for deciding when `NOW` needs new evidence.

| representation | IID | sparse-rate OOD | dense-rate OOD | long-delay OOD |
|---|---:|---:|---:|---:|
| arrival histogram | 0.861 | 0.767 | 0.909 | 0.767 |
| valid-time histogram | 0.882 | 0.785 | 0.936 | 0.810 |
| **exponential bank** | **0.894** | **0.784** | **0.950** | **0.816** |
| wave only | 0.821 | 0.712 | 0.903 | 0.731 |
| oil only | 0.890 | 0.784 | 0.948 | 0.815 |
| wave + oil | 0.889 | 0.783 | 0.949 | 0.815 |

There is no active-refresh rescue for the wave representation.

## 7. Uncertainty quality

Error-detection AUC uses `1 - max(class probability)` to rank episodes where the
passive prediction is wrong.

| representation | IID | sparse-rate OOD | dense-rate OOD | long-delay OOD |
|---|---:|---:|---:|---:|
| arrival histogram | 0.733 | 0.691 | 0.776 | 0.681 |
| valid-time histogram | 0.767 | 0.714 | 0.815 | 0.705 |
| exponential bank | **0.784** | 0.723 | 0.832 | 0.719 |
| wave only | 0.764 | **0.726** | 0.815 | **0.727** |
| oil only | 0.780 | 0.719 | 0.832 | 0.712 |
| wave + oil | 0.781 | 0.722 | **0.835** | 0.721 |

This is worth recording honestly: pure waves are not uniformly bad at uncertainty
ranking. Under sparse and long-delay OOD they slightly exceed the exponential bank in
error-detection AUC.

But because their passive state estimate is much worse, that ranking advantage does
not translate into better accuracy under the exact same refresh budget.

## 8. The oil geometry is exactly removable

This is the strongest conceptual attack.

Let the slow graph field obey

\[
\dot m = - (D L + \lambda I)m + s(t).
\]

Diagonalize the graph Laplacian:

\[
L = Q\Lambda Q^T.
\]

In modal coordinates `z = Q^T m`, each mode is just an independent exponential filter:

\[
\dot z_k = -r_k z_k + c_{k,y(t)},
\qquad
r_k = D\lambda_k + \lambda.
\]

At decision time, for discrete observations `i`,

\[
z_k(T) = \sum_i c_{k,y_i}\,e^{-r_k(T-t_i)}.
\]

And graph-space oil is merely

\[
m(T)=Qz(T).
\]

Because `Q` is orthogonal, a linear readout has access to the same information in
`m` and `z`.

The committed experiment contains an exact numerical assertion of this identity:

```python
oil_only_feature(events) == OIL_Q @ modal_exp20_feature(events)
```

up to floating-point tolerance.

So on this linear benchmark the graph geometry does **not** create the useful fading
memory. It chooses a basis and a set of decay rates.

That is an important kill.

## 9. What survives

### Survives: valid/world-time provenance

The valid-time histogram consistently beats the arrival-time histogram. Evidence is
more useful when the state preserves **when it was true**, not only when it arrived.

### Survives: multi-timescale fading state

The exponential bank and slow diffusive field are strong. A bounded continuously
changing state can indeed summarize temporal evidence usefully.

### Does not survive: simple wave advantage

The pure local spring state is substantially worse than the boring filters.

### Does not survive: wave + oil synergy

Adding wave coordinates to the oil state does not materially improve the oil state.

### Does not survive: linear local geometry as the source of the oil benefit

The slow graph field is exactly equivalent, under a basis change, to a compressed bank
of exponential modes.

## 10. Revised interpretation of the lake metaphor

The sentence

> the matrix stays; the flow does not

is still mechanically true and can still be a useful implementation picture.

But this benchmark says that **being a flowing local wave is not itself a useful AI
inductive bias for current-state temporal inference**.

The slow field is more interesting, but in the linear regime it belongs to familiar
state-space / fading-memory mathematics.

So do not promote lake dynamics into the WidePresent architecture merely because they
look physically natural.

## 11. H5 status after this attack

The simple version of H5 should be marked **negative**:

> cyclic/oscillatory local geometry does not beat matched non-oscillatory temporal
> filters on the delayed current-state task.

That does not prove oscillations can never help. It says they need a **specific future
failure** that cannot already be solved by:

```text
valid-time provenance
+ multi-timescale exponential state
+ ordinary uncertainty
+ active refresh
```

Only then should wave phase, interference, local routing, or nonlinear plasticity be
reintroduced.

## 12. What is now genuinely worth attacking

The strongest current WidePresent-shaped question is no longer "lake or ledger?"

It is:

> **Can a bounded temporal state preserve useful provenance and current-state
> observability under a tighter state/computation budget than ordinary timestamped
> filters, especially when valid time is uncertain or must itself be inferred?**

The 20-number oil/modal state matching a 60-number explicit exponential bank hints at a
compression question, but the graph cannot claim credit: its modal representation is
already a 20-number non-geometric filter state.

That compression question should therefore be attacked next with ordinary learned or
random low-dimensional temporal projections before any geometric interpretation.
