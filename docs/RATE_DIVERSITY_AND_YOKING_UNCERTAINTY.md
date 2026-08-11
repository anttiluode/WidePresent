# Rate diversity and yoking uncertainty

Date: 2026-08-11

`docs/TIME_STRUCTURE_IDENTIFIABILITY.md` established the exact single-rate ambiguity:

\[
q = ac+b
\]

is identifiable, while `a` (physical-time decay) and `b` (event-distance decay) are not.

This note asks the practical follow-up:

> **How much rate diversity is needed before the decomposition becomes numerically trustworthy?**

The committed demonstration is:

`experiments/rate_diversity_uncertainty.py`

## 1. The conditioning quantity is rate variance

Suppose we observe effective decay rates

\[
q_j=a c_j+b+\epsilon_j
\]

at experienced event rates / gaps `c_j`.

The linear design matrix has rows

\[
X_j=[c_j,1].
\]

Then

\[
X^TX=
\begin{bmatrix}
\sum c_j^2 & \sum c_j\\
\sum c_j & n
\end{bmatrix}
\]

and therefore

\[
\det(X^TX)
=n\sum c_j^2-(\sum c_j)^2
=n^2\operatorname{Var}(c).
\]

This is exact.

So the amount of information separating the two yoking coordinates collapses with the **variance of experienced rates**.

Two distinct rates technically restore rank 2, but if those rates are almost the same the problem can remain extremely ill-conditioned.

## 2. Finite-noise toy curve

A Monte Carlo sanity check used:

```text
20 rate observations
nominal rate/gap c = 0.5 s/event
Gaussian noise sigma = 0.01 on q
2000 repeats
```

The true axes were either:

```text
time target:      a=0.4, b=0
structure target: a=0,   b=0.2
```

Representative results:

| rate half-width around 0.5 | median condition number | median parameter error |
|---:|---:|---:|
| 0 | infinite | non-identifiable |
| 0.005 | ~444 | ~0.59 |
| 0.010 | ~222 | ~0.30 |
| 0.020 | ~111 | ~0.15 |
| 0.050 | ~44 | ~0.059 |
| 0.100 | ~22 | ~0.030 |
| 0.200 | ~11 | ~0.015 |

The time- and structure-axis targets show essentially the same conditioning curve once the rank deficiency is removed.

The numerical values are toy-specific. The scaling principle is not:

> **small rate variance means large uncertainty about what the memory horizon is yoked to.**

## 3. The hidden uncertainty is invisible IID

At the fixed nominal rate `c=0.5`, every point on

\[
0.5a+b=0.2
\]

predicts the same nominal effective decay.

The unconstrained direction is

\[
v=(1,-c).
\]

Move along it:

\[
(a,b)\rightarrow(a,b)+\delta(1,-c).
\]

At the training rate,

\[
\delta(c-c)=0,
\]

so nothing changes.

At a new rate `c'`, however,

\[
q'\rightarrow q'+\delta(c'-c).
\]

That gives a useful interpretation:

> **the uncertainty is latent during IID operation and is exposed by rate shift.**

Models can therefore agree perfectly on every nominal example while disagreeing sharply about the same content played faster or slower.

## 4. Concrete null-space example

All rows below fit the nominal `0.5 s/event` rate exactly:

| `a` | `b` | nominal `q` | compressed `q` | stretched `q` |
|---:|---:|---:|---:|---:|
| 0.00 | 0.200 | 0.200 | 0.200 | 0.200 |
| 0.10 | 0.150 | 0.200 | ~0.179 | ~0.237 |
| 0.20 | 0.100 | 0.200 | ~0.158 | ~0.273 |
| 0.30 | 0.050 | 0.200 | ~0.137 | ~0.310 |
| 0.40 | 0.000 | 0.200 | ~0.115 | ~0.346 |

The first row is purely structure-yoked.

The last row is purely time-yoked.

There is no nominal-rate experiment that can choose between them in this model.

## 5. Why this matters more than a point-estimate architecture

A learner forced to output one `(a,b)` pair may look confident simply because optimization selected one point on the ridge.

But under weak rate diversity, a better state description should arguably retain uncertainty along the unresolved yoking direction.

That suggests a different WidePresent-shaped object:

```text
not only:
    current temporal state

but also:
    what transformation of that state under rate change is actually identified?
```

In other words, the system can know the present well at the currently experienced rhythm while still being uncertain about **how its present should deform when rhythm changes**.

That is a more precise uncertainty than generic classifier entropy.

## 6. Reinterpretation of the hard clock

A hard physical-time channel sets, by design,

```text
b = 0
```

for part of the representation.

A hard structural channel sets

```text
a = 0
```

for another part.

These are priors.

They are beneficial when:

1. the true task uses the corresponding invariant; and
2. the training distribution does not contain enough rate variation to identify that invariant reliably.

They are harmful when the prior is wrong.

So a dual-yoked representation can be understood as **hedging against underidentification** by retaining both axis-aligned possibilities.

That is a stronger justification than "brains may have two clocks," and a weaker claim than "dual yoking is a new neural mechanism."

## 7. Connection to LLM temporal blindness

This gives a plausible engineering interpretation of why timestamp prompting can be weak.

A model can be exposed to a timestamp or `dt` value at inference without having learned a stable rule for how memory should transform with elapsed wall time.

That is **clock exposure**.

A temporal runtime kernel or constrained continuous-time state can instead impose some elapsed-time arithmetic outside the learned content dynamics.

That is **clock constraint**.

This does not prove the explanation for TicToc. The external B-vs-C gate remains the relevant test.

## 8. Strong prior-art attacker

Mozer, Kazakov & Lindsey (2017), **Discrete Event, Continuous Time RNNs** (arXiv:1710.04110), are especially relevant here. Their CT-GRU uses timestamps to drive intrinsic continuous-time decay across multiple memory scales rather than merely concatenating time as an ordinary input feature.

That is very close to the distinction between clock constraint and clock exposure.

Their reported experiments found CT-GRU and ordinary timestamp-fed gated RNNs broadly comparable across their tested datasets, so WidePresent should not assume a constrained clock will automatically win.

The useful unanswered question is specifically the rate-yoking/identifiability regime studied here.

## 9. Next attack

The next experiment should compare four models as training rate variance is swept from nearly zero to broad:

```text
event GRU
explicit-dt GRU
CT-GRU / continuous-time recurrent baseline
hard dual-yoked state
```

Measure:

- time-vs-structure yoking index;
- OOD compressed/stretched accuracy;
- calibration / uncertainty under unseen rates;
- sample efficiency as a function of Var(rate).

Predicted pattern if the current reasoning is right:

```text
low rate diversity:
    hard prior can help if correct
    ordinary dt model underidentifies yoking

high rate diversity:
    learned continuous-time model should catch up
    special hard clock advantage should shrink
```

If that pattern does not appear, the identifiability story is insufficient and should be revised.
