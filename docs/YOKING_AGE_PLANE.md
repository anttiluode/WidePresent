# The yoking age plane

Date: 2026-08-11

Several branches of WidePresent have now collapsed into a smaller object.

The useful state coordinate is not obviously a fixed-width temporal window, a lake, a clock token, or two named memory modules.

It is the pair

\[
(\Delta t,\Delta n),
\]

where:

```text
Δt = physical / wall-clock age in seconds
Δn = structural / event-position age
```

A fading memory can then be understood as a kernel over this two-dimensional **age plane**.

## 1. Yoking is an orientation

The simplest family used in the current experiments is

\[
k(\Delta t,\Delta n)
=\exp(-a\Delta t-b\Delta n).
\]

The quantity

\[
\rho=a\Delta t+b\Delta n
\]

acts like an effective temporal distance.

Level sets

\[
a\Delta t+b\Delta n=\text{constant}
\]

are straight forgetting contours in the age plane.

Their orientation is the yoking.

```text
b = 0    pure wall-time yoking
 a = 0    pure event/structure yoking
 a,b > 0  mixed yoking
```

So "what is the integration window yoked to?" becomes a geometric question in the space of **data coordinates**, not a claim about a physical geometric substrate.

## 2. Why one event rate hides orientation

At a fixed rate `c`, all observed ages lie on

\[
\Delta t=c\Delta n.
\]

That is a one-dimensional line through the age plane.

Sampling a two-dimensional kernel only along one line cannot determine its full orientation.

For the exponential family,

\[
k(c\Delta n,\Delta n)
=\exp[-(ac+b)\Delta n].
\]

Only `ac+b` is visible.

This is the identifiability result in `docs/TIME_STRUCTURE_IDENTIFIABILITY.md`.

Rate variation changes the slope of the sampled line and exposes the second dimension.

That gives a useful picture:

```text
single-rate training
    sees one diagonal slice through the age plane

rate-diverse training
    sees several slices
    -> orientation can become identifiable
```

## 3. One mixed kernel can hedge unresolved yoking

`experiments/mixed_yoking_hedge.py` asks an intentionally tiny question.

A single kernel per content channel is constrained to have the correct nominal effective decay but can vary its orientation between the two axes.

Parameterize orientation by `g`:

\[
a=\frac{gq}{c_0},
\qquad
b=(1-g)q.
\]

Then:

```text
g = 0  pure structure/event yoking
g = 1  pure physical-time yoking
g = 0.5 balanced mixture
```

The toy contains two simultaneous targets:

- one genuinely physical-time-yoked;
- one genuinely structure-yoked.

Compressed and stretched rate-shift scores are averaged.

Five-seed exploratory sweep:

| `g` | time head | structure head | joint |
|---:|---:|---:|---:|
| 0.0 | ~0.887 | ~0.924 | ~0.819 |
| 0.2 | ~0.898 | ~0.921 | ~0.827 |
| 0.4 | ~0.907 | ~0.915 | ~0.830 |
| 0.5 | ~0.911 | ~0.911 | ~0.829 |
| 0.6 | ~0.915 | ~0.907 | ~0.829 |
| 0.8 | ~0.921 | ~0.898 | ~0.826 |
| 1.0 | ~0.923 | ~0.887 | ~0.818 |

The worst-head score is maximized at approximately

\[
g=0.5.
\]

The best joint score is a broad plateau near the middle (`g≈0.4..0.7`; the particular five-seed maximum was near `0.55`).

This is exactly what a symmetric robust compromise should look like.

## 4. The basis-size attack

Before the one-kernel test, mixed kernel banks were compared at several state budgets.

Each kernel was applied to both content channels, so `K` kernels means `2K` state numbers.

Representative three-seed joint scores:

### One-rate training

| state numbers | joint |
|---:|---:|
| 4 | ~0.822 |
| 8 | ~0.820 |
| 16 | ~0.823 |
| 32 | ~0.823 |

### Moderate rate-diverse training

| state numbers | joint |
|---:|---:|
| 4 | ~0.826 |
| 8 | ~0.824 |
| 16 | ~0.830 |
| 32 | ~0.833 |

And then the two-number one-kernel state reached roughly

```text
joint ~0.827
```

on the symmetric toy.

So this benchmark absolutely does **not** justify a large temporal architecture.

The target itself is simple enough that a very small robust summary works.

## 5. Axis-aligned filters are not essential here

A 16-number bank containing only mixed orientations was compared against a bank explicitly containing physical-time and event-time axes.

At one-rate training:

```text
axis dual bank     joint ~0.832
mixed grid         joint ~0.832
random mixed bank  joint ~0.832
```

With moderate rate diversity:

```text
axis dual bank     joint ~0.842
mixed grid         joint ~0.837
random mixed bank  joint ~0.834
```

There is a small benefit to spanning the pure axes once the data contain enough rate variation to distinguish them, but no dramatic separation.

Therefore the current evidence supports:

> **span useful orientations in the age plane**

more than:

> **build one wall-clock module and one event-clock module.**

## 6. A better abstraction for attention

The same coordinate can be used in an attention bias:

\[
\operatorname{score}_{ij}
=\frac{q_i^Tk_j}{\sqrt d}
-a_h\Delta t_{ij}
-b_h\Delta n_{ij}.
\]

Different heads could have different yoking orientations `(a_h,b_h)`.

This does not automatically produce an "unoverrideable clock," because a sufficiently large content score can overcome an additive bias.

A genuinely constrained physical-time channel would instead decay or transform part of the memory state *before* content attention sees it.

Still, the age-plane parameterization makes the comparison precise:

```text
position-only attention:  a_h = 0
wall-time-only attention:  b_h = 0
mixed attention:           a_h,b_h > 0
```

At a single event rate, the slopes are again confounded.

## 7. Extension to provenance

The earlier bitemporal branch distinguished:

```text
world / valid time
arrival / knowledge time
```

Those could extend the age plane to a larger coordinate system, for example

\[
(\Delta t_{world},\Delta t_{arrival},\Delta n).
\]

But the earlier attacks showed that every coordinate should have to earn its keep.

Arrival time did not add beyond valid time on the simple delayed-state task.

So the current minimal geometry remains two-dimensional:

\[
(\Delta t,\Delta n).
\]

Add more axes only when a task exposes a specific ambiguity that needs them.

## 8. What this does to WidePresent

The phrase "wide present" started by suggesting a literal temporal extent around `now`.

The current experiments suggest a different interpretation:

> **a present is wide when current evidence is represented with enough temporal coordinates to preserve the transformations that matter.**

Width may therefore be representational rather than a single duration.

A two-second-old event can be:

```text
2 seconds old
1 event old
```

or

```text
2 seconds old
100 events old
```

Those are the same physical age and radically different structural ages.

Collapsing them to one scalar age silently chooses a yoking.

The age-plane representation refuses to make that choice too early.

## 9. What is still ordinary

Nothing in this note establishes a novel architecture.

Two-coordinate kernels, continuous-time recurrent models, relative position/time biases, temporal point-process models, and multiscale fading memories all occupy nearby territory.

The useful result so far is diagnostic:

1. identify the temporal coordinates;
2. test whether the training distribution actually separates them;
3. measure what the learned integration window is yoked to;
4. preserve unresolved coordinates if collapsing them would create OOD ambiguity.

That is a research procedure before it is an invention.

## 10. Next place to dig

The synthetic exponential toy is now close to exhausted.

The next worthwhile task should have **content-dependent yoking** rather than globally fixed yoking.

Example:

```text
some event types become stale in wall-clock seconds
other event types remain relevant for a fixed number of structural updates
```

The model would have to infer the yoking orientation from content while still respecting elapsed-time dynamics.

That is where a multiscale CT-GRU, mixed-yoking attention, or constrained age-plane memory might begin to differ in a nontrivial way.

It also connects directly to real agents:

```text
weather result          -> wall-time freshness
reservation state       -> world-valid state / change events
conversation reference  -> structural discourse distance
scheduled prediction    -> future wall time
```

A useful system must not apply one universal forgetting metric to all of them.

That is the next branch worth testing.
