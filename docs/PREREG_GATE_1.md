# Gate 1 preregistration — equal-information test

Date registered: 2026-08-11

## Question

Does an explicit moving time-coordinate frame provide any measurable advantage once competing models receive the **same objective timing information**?

Gate 0 is not enough: timestamp-only solved it perfectly.

## Models

Minimum comparison set:

1. **Event GRU** — event content only. Negative control, expected to fail rate-confounded tasks.
2. **dt-GRU** — same GRU plus exact elapsed time since the previous event.
3. **Timestamp Transformer** — small causal Transformer with explicit continuous-time features.
4. **LMU/HiPPO-family baseline** — continuous-history state over a matched horizon.
5. **WidePresent** — fixed `dt`, age-indexed past, marked now, future slots when the task uses forecasting.

Optional if implementation cost stays reasonable:

6. ODE-RNN / Neural CDE style continuous-time baseline.

## Match conditions

Report:

- trainable parameters;
- state dimension / persistent bytes;
- nominal operations per objective second and per event;
- training wall time;
- inference wall time;
- seeds.

Do not make WidePresent larger just because its state is interpretable.

## Task family A — rate warp

Generate latent continuous processes, then observe the same process at variable and irregular event rates.

Train sampling regimes and test regimes must include held-out rate ranges.

Primary metric: OOD task accuracy/error as a function of rate warp.

## Task family B — empty time

Include intervals containing no input events. Targets depend on elapsed duration or on the evolution of a latent process during silence.

This prevents an event counter from being a sufficient clock.

## Task family C — asynchronous binding

Two modalities observe events on independent clocks. Targets require identifying which events were simultaneous or fell within a fixed millisecond tolerance.

Jitter and event density vary independently across train/test.

## Task family D — prediction rendezvous

A model predicts a value/event for an absolute future time. When that time arrives, the forecast is scored against observation.

Primary metrics:

- value error;
- arrival-time error;
- calibration if probabilistic;
- error after sampling-rate shift;
- error after silent intervals.

## Registered interpretation

### Strong positive
WidePresent beats all equally-informed baselines on at least one task family under matched resources, and the advantage survives seeds and rate-warp controls.

This earns a targeted architectural claim tied to that task/mechanism only.

### Weak positive
WidePresent ties the strongest baseline but provides substantially simpler temporal provenance, calibration or diagnostics.

This earns an engineering/interface result, not a superior-computation claim.

### Null
`dt`-aware GRU / LMU / continuous-time baseline matches WidePresent throughout.

Conclusion: explicit physical time matters, but the WidePresent geometry did not earn its keep.

### Negative
WidePresent is systematically worse or materially more expensive.

Conclusion: use it as a visualization/diagnostic tool at most; do not rescue by unregistered architecture inflation.

## Things we are not allowed to claim from Gate 1

- consciousness;
- subjective time;
- a hippocampal mechanism;
- human-like cognition;
- novelty of fixed clocks, sliding windows or prediction.
