# Linear geometry modal kill

Date: 2026-08-11

This note extracts the strongest result from the lake-vs-ledger attack and states it at the level that is actually justified.

The short version is:

> **A fixed linear geometry does not create representational information merely by being geometric. Under an unrestricted linear readout, an invertible change of state coordinates preserves exactly the same input-output information. For symmetric graph diffusion, the statement becomes stronger: the local field is exactly a bank of independent exponential filters written in a spatial basis.**

This is an algebraic statement, not a benchmark observation.

It does **not** say that every local linear system is "just decay rates." General fixed linear systems can be non-normal or non-diagonalizable and can show transient amplification or Jordan structure. The more general invariant is coordinate equivalence, not diagonal independence.

## 1. General fixed-linear proposition

Consider a finite-dimensional, input-independent linear state system

\[
\dot x(t)=A x(t)+B u(t),
\qquad
y(t)=C x(t),
\]

with fixed matrices `A`, `B`, `C`.

Let `P` be any invertible matrix and define new coordinates

\[
z=P^{-1}x.
\]

Then

\[
\dot z=(P^{-1}AP)z+(P^{-1}B)u,
\qquad
y=(CP)z.
\]

So the transformed system has exactly the same input-output map.

### Consequence for learned linear heads

If a downstream linear head can choose arbitrary weights, then representing state as `x` or as `z=P^{-1}x` cannot change what linear functions of that state are available. If

\[
\ell_x=w_x^T x,
\]

then with `x=Pz`, choose

\[
w_z=P^T w_x
\]

and obtain

\[
w_z^Tz=w_x^Tx.
\]

Therefore an invertible geometric basis change cannot, by itself, increase linearly readable information.

This is the general kill.

It is weaker than saying that all fixed linear systems are collections of decays, but stronger and more defensible than an empirical "the graph did not help" claim.

## 2. Diffusion corollary: here it really is only exponential modes

For the WidePresent slow/oil field,

\[
\dot m=-(D L+\lambda I)m+s(t),
\]

where `L` is an undirected graph Laplacian.

Because `L` is real symmetric,

\[
L=Q\Lambda Q^T
\]

for an orthogonal matrix `Q`.

Define modal coordinates

\[
z=Q^Tm.
\]

Then

\[
\dot z=-(D\Lambda+\lambda I)z+Q^Ts(t).
\]

Every mode is independent:

\[
\dot z_k=-r_k z_k+c_k(t),
\qquad
r_k=D\lambda_k+\lambda.
\]

For discrete impulses at valid times `t_i`,

\[
z_k(T)=\sum_i c_{k,i}\,e^{-r_k(T-t_i)}.
\]

Thus the graph diffusion field is **exactly a multi-timescale exponential filter bank written in graph coordinates**:

\[
m(T)=Qz(T).
\]

This is why `oil_only` and `modal_exp20` in `experiments/lake_vs_ledger_attack.py` are numerically identical up to the orthogonal transform.

For this case the concise statement is justified:

> **The geometry chooses a basis and a spectrum of decay rates; it does not create an additional temporal memory mechanism.**

## 3. What this kills

Under the assumptions above, one should not attribute extra representational power to a fixed linear field merely because it is implemented as:

- diffusion on a graph;
- a linear mesh or lattice;
- a fixed linear "coherence field";
- a fixed linear splat/field medium;
- a local linear reservoir followed by an unrestricted linear head;
- any other invertible spatial rewriting of the same finite-dimensional linear state.

A geometry can still be a useful **implementation** of a filter bank. It may provide locality, sparsity, parallel physical execution, wiring constraints, numerical structure, or hardware advantages. Those are implementation claims and must be measured as such.

It does not earn an information-processing claim merely from spatial form.

## 4. What this does *not* kill

### 4.1 Non-normal fixed linear dynamics

A general fixed `A` need not be orthogonally diagonalizable. Non-normal systems can show transient amplification even when all eigenmodes eventually decay, and defective systems can require Jordan blocks.

The general coordinate-invariance proposition still holds, but the phrase "independent exponential filters" does not.

### 4.2 Restricted readouts

If the readout is constrained to be local, sparse, low-rank, low-precision, or otherwise unable to absorb an arbitrary basis transform, then geometry can change what is cheaply accessible.

That becomes a resource/inductive-bias question, not a pure information question.

### 4.3 Nonlinear local operations

If nonlinearities occur between local propagation steps, a global linear change of basis generally does not remove the computation.

### 4.4 State-dependent or input-dependent coupling

Suppose

\[
\dot x=A(u(t),x(t),t)x+B(u(t),x(t),t)u.
\]

There is no longer one fixed operator `A` to eliminate once and for all.

A particularly clean escape is a sequence of input-dependent linear operators

\[
x_{n+1}=A(u_n)x_n.
\]

If different input-conditioned operators do not commute,

\[
[A(u_i),A(u_j)]\neq 0,
\]

then in general there is no single basis in which the whole computation becomes a fixed set of independent scalar modes. Order now matters:

\[
A(u_2)A(u_1)\neq A(u_1)A(u_2).
\]

This is the mathematically relevant escape hatch for ideas such as token-dependent scattering angles or gates. The escape is not "oscillation" itself; it is **input-conditioned, noncommuting computation**.

### 4.5 Physical locality and cost

Two coordinate-equivalent state-space descriptions can be radically different to build. One may require only nearest-neighbor coupling while its modal representation requires dense global mixing. If the research question is hardware, energy, latency, robustness, or local learning, geometry may matter even though abstract representational information is unchanged.

## 5. A useful hierarchy of claims

When a geometric/dynamical model appears to win, ask in this order:

1. **Basis test** — is the win invariant under an invertible change of coordinates?
2. **Modal test** — if the operator is normal/symmetric, does the model reduce to independent scalar modes?
3. **Ablation test** — does the geometric/oscillatory part add beyond the slow modes?
4. **Readout test** — does the win remain with an equally capable non-geometric readout?
5. **Constraint test** — is any remaining advantage specifically due to locality, sparsity, hardware, or limited readout resources?
6. **Noncommutation test** — if the operator depends on input/state/time, do the conditioned operators actually fail to commute in a task-relevant way?

Only after these survive should "geometry" receive explanatory credit.

## 6. Relation to WidePresent

This result is not evidence for WidePresent. It is a methodological result produced while trying to kill one WidePresent branch.

For WidePresent it means:

- the lake/wave story cannot be promoted merely as a richer temporal substrate;
- the slow oil field is ordinary multi-timescale fading memory in disguise;
- future geometric branches must name a resource constraint or a genuinely input-dependent/nonlinear interaction that cannot be transformed away;
- the core time question should return to the skipped empirical hook: whether an explicit content-blind clock changes a network's **time-yoked versus structure-yoked integration behavior under rate shift**.

That last question is independent of the modal kill and should be tested directly.

## 7. Status

**Proved algebraically within stated assumptions.**

The exact diffusion corollary is already asserted numerically in `experiments/lake_vs_ledger_attack.py`.

The important discipline is the scope:

> fixed + linear + input-independent + unrestricted linear readout

is the kill zone.

Input-dependent coupling, nonlinear operations, constrained/local readout, and physical implementation costs are outside that theorem and require separate tests.
