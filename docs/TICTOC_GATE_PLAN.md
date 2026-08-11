# TicToc as an external adversary for WidePresent H1

Date: 2026-08-11

## Why use it

Cheng et al.'s ACL 2026 TicToc benchmark is unusually close to the motivating WidePresent failure. It holds a multi-turn tool-use conversation essentially fixed while changing how much real-world time has elapsed. Humans decide whether the agent should reuse previous information or call the tool again.

The official data make the intervention explicit. Raw scenario samples can contain three alternative timestamps for the final user turn and candidate outputs. The human-label CSV records preferences for **Time Variant 1/2/3** separately.

This means we do not need to invent a toy version of the user's observation that an LLM can confuse hours, days and months. There is now an external benchmark for a closely related failure.

## What the official timestamp condition does

The official OpenAI handler prefixes timestamp-visible message content with text of the form:

```text
[ISO_TIMESTAMP] message content
```

when `--use_time_stamp` is enabled.

That is a passive-information condition: the language model receives the clock readings and must infer the elapsed durations relevant to the tool-use decision.

The paper reports that timestamps help only modestly overall; no tested model exceeds 65% normalized human alignment with timestamp information.

## WidePresent H1 condition

`experiments/tictoc_temporal_kernel_adapter.py` creates a label-blind **derived temporal state** condition.

It uses exactly the timestamp fields already present in the sample to compute:

```text
current decision time
elapsed since conversation start
elapsed age of each timestamp-visible prior message
```

It does **not** compute:

```text
fresh / stale
call tool / answer directly
recommended threshold
human preference
scenario-specific volatility class
```

The state is injected into the system message. The resulting file should be evaluated through the official TicToc inference code **without** `--use_time_stamp`, because the derived state replaces passive timestamp text.

## Why the restriction matters

TicToc's target is partly a human judgment about how quickly information becomes worth refreshing. If WidePresent were allowed to convert elapsed time into `stale=true` using the benchmark labels, the experiment would be circular.

H1 asks a narrower question:

> **Does doing exact clock arithmetic outside the language model make the same temporal information more usable?**

That is all.

## Recommended paired evaluation

For the same model and same dataset split:

### A — no time

```bash
python eval_from_api.py --model "$MODEL" --data "$DATA" --output_dir outputs_no_time
python get_metric.py --model "$MODEL" --data "$DATA" --output_dir outputs_no_time
```

### B — official passive timestamps

```bash
python eval_from_api.py --model "$MODEL" --data "$DATA" --use_time_stamp --output_dir outputs_timestamp
python get_metric.py --model "$MODEL" --data "$DATA" --use_time_stamp --output_dir outputs_timestamp
```

### C — derived temporal kernel

From WidePresent:

```bash
python experiments/tictoc_temporal_kernel_adapter.py \
  --input "$DATA" \
  --output tictoc_kernel.json
```

Then through the official TicToc evaluator:

```bash
python eval_from_api.py --model "$MODEL" --data tictoc_kernel.json --output_dir outputs_kernel
python get_metric.py --model "$MODEL" --data tictoc_kernel.json --output_dir outputs_kernel
```

Do **not** add `--use_time_stamp` in C.

## Important licensing / repository boundary

WidePresent does not copy or vendor TicToc data. The adapter expects a separately obtained local TicToc JSON file. This also keeps the external benchmark independently versioned.

## Interpretation

### C > B

Supports H1: exact derived temporal state is more usable than passive timestamp text for temporal tool-use decisions.

This does **not** show that a WidePresent matrix or learned temporal layer is needed. The deterministic kernel may be sufficient.

### C ≈ B

H1 is null. The LLM can use the timestamp information about as well as the derived ages; no reason to insert the kernel for this benchmark.

### C < B

Derived state hurts. Inspect whether it removes useful absolute-time semantics or bloats/distracts the context; do not redefine the metric.

## Next only after H1

H2/H3 compare the deterministic kernel against bitemporal and wide relative-time working state. There is no reason to run those more elaborate models if H1 cannot first show that temporal representation format matters.
