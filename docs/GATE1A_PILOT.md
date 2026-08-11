# Gate 1A pilot — irregular-rate fixed-horizon forecasting

Date: 2026-08-11

Status: **exploratory pilot inside the broader Gate 1 preregistration. Not a positive result.**

## Task

A latent continuous signal is the sum of two sinusoids plus a small trend. Models observe noisy samples over a 2.0 s window and forecast the latent value 350 ms after `now`.

Training observation rate: 7–14 events/s.

Tests:

- IID: 7–14 events/s;
- slow OOD: 2–4 events/s;
- fast OOD: 22–35 events/s.

All time-aware models receive objective timing information. The fixed-grid models deposit events into 50 ms bins with an explicit observation mask.

## Models

| model | timing | access pattern | parameters |
|---|---|---|---:|
| Event GRU | none | recurrent event stream | 4,449 |
| dt-GRU | exact event `dt` | recurrent event stream | 4,545 |
| Time Transformer | explicit timestamp/age | direct attention over irregular events | 2,561 |
| Grid GRU | fixed 50 ms grid | recurrent over same grid as Wide | 3,529 |
| Wide MLP | fixed 50 ms age-indexed grid | direct access to entire past matrix | 3,745 |

## Three-seed pilot

Training: 2,500 trials, 6 epochs per seed. Test: 600 trials per split. Seeds 0, 1, 2.

Mean RMSE (sample standard deviation in parentheses):

| model | IID | slow OOD | fast OOD |
|---|---:|---:|---:|
| Event GRU | 0.6862 (0.0119) | 0.7037 (0.0219) | 0.6949 (0.0108) |
| dt-GRU | 0.6868 (0.0128) | 0.7009 (0.0237) | 0.7051 (0.0190) |
| Time Transformer | 0.6756 (0.0140) | 0.6992 (0.0208) | 0.6473 (0.0068) |
| Grid GRU | 0.6829 (0.0110) | 0.6995 (0.0139) | 0.6592 (0.0047) |
| Wide MLP | 0.6692 (0.0219) | 0.6956 (0.0187) | 0.6352 (0.0225) |

## What this says

The recurrent event models do not exploit `dt` enough to improve this particular task at this training budget. Direct access to objective-time coordinates does better under the fast-rate shift: both the timestamped Transformer and fixed-grid models improve.

The Wide MLP has the lowest mean fast-OOD RMSE in this small pilot, but the margin over the timestamped Transformer is only ~0.012 and is seed-sensitive. On slow OOD, all time-aware models are effectively tied.

## Why this is not yet evidence for WidePresent

Several confounds remain:

1. **Binning denoises.** Multiple noisy observations landing in one fixed-time bin are averaged. Some fast-rate gain may be free preprocessing.
2. **Training is short.** The Transformer and recurrent models may simply be undertrained.
3. **The Wide MLP sees the whole matrix simultaneously.** Any advantage could be direct-access memory rather than a special `now` geometry.
4. **No LMU/HiPPO baseline yet.** The strongest continuous-history control from prior art is absent.
5. **No future register is used.** This task only tests the retention side of WidePresent, not prediction rendezvous.
6. **Task loss is still large.** The pilot is useful for architecture triage, not performance claims.

## Next discriminator

Do not tune this task to make WidePresent win.

Next, build a prediction-rendezvous task in which future forecasts are assigned absolute deadlines and later mature into observations. Compare against:

- timestamped Transformer;
- dt-GRU;
- a rolling world-model / multi-horizon forecast head;
- LMU/HiPPO memory where practical.

If the future-coordinate mechanism gives no advantage in calibration, timing error or rate-shift robustness, then the past/now/future geometry has not earned its keep.
