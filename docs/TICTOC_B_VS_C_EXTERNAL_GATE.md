# TicToc external gate — B versus C

Date: 2026-08-11

This is the external gate WidePresent should run before spending more time on elaborate temporal machinery.

Official benchmark: Cheng et al., **Your LLM Agents are Temporally Blind: The Misalignment Between Tool Use Decisions and Human Time Perception**, Findings of ACL 2026.

Official repository: `chengez/TicToc`.

## Question

Does a deterministic, label-blind temporal runtime state help an unchanged model more than simply showing it the same timestamps as text?

### B — official timestamp condition

Use TicToc exactly as documented:

```bash
python eval_from_api.py \
  --model "$MODEL" \
  --data merged_fully_labeled_data.json \
  --use_time_stamp \
  --output_dir "$OUT"
```

The official OpenAI handler prefixes each timestamp-visible message with its absolute timestamp.

### C — WidePresent derived-kernel condition

Use `experiments/tictoc_temporal_kernel_adapter.py` to convert the same timestamp information into deterministic runtime quantities:

- current decision time;
- elapsed conversation age;
- elapsed age of prior timestamp-visible messages.

No preference label, freshness threshold, staleness rule, semantic interpretation, or tool recommendation is injected.

Then run the official model handler **without** `--use_time_stamp`.

## One-command runner

`experiments/run_tictoc_b_vs_c.py` now performs the complete procedure while delegating inference and scoring to the official repository.

Example:

```bash
python experiments/run_tictoc_b_vs_c.py \
  --tictoc-dir ../TicToc \
  --model gpt-4.1-mini-2025-04-14-FC \
  --data merged_fully_labeled_data.json
```

It:

1. loads the official samples;
2. prepares condition C;
3. asserts identical sample count, IDs, and order;
4. runs official condition B;
5. runs condition C through the same official model handler;
6. calls official `get_metric.py` for both;
7. prints the two Normalized Alignment Rates and `C - B`.

Use `--prepare-only` to generate and validate C without making model calls.

## Fairness

The comparison intentionally gives C **no extra semantic information**.

B receives absolute timestamps in message text.

C receives the same clock information after deterministic arithmetic. The raw timestamps remain in the JSON because the official dataset schema expects them, but the model handler is invoked without `--use_time_stamp`, so they are not exposed in message content. The only time information visible to the model is the derived kernel block.

C retains the current absolute decision time. This prevents an unfair comparison in which B knows calendar time but C knows only relative durations.

## Metric

Do not invent a WidePresent metric.

Use TicToc's official `get_metric.py`, which determines whether the model attempted a tool call and computes its reported **Normalized Alignment Rate** from tool/direct true-positive and true-negative rates.

## Current execution status

**Prepared/reviewed, not yet externally executed in this environment.**

The current ChatGPT execution shell has no outbound network access for the official API runner, and no local TicToc model weights are available. Therefore there is no honest B-vs-C score to record here yet.

This is not a scientific failure and must not be converted into an estimated or label-derived pseudo-score.

The next run on a normal machine with an API key or supported local model should fill in:

```text
model:
full/test split:
B normalized alignment:
C normalized alignment:
C-B:
failed samples B/C:
```

## Kill criterion

The gate is deliberately severe:

> **If C does not materially beat B on the same model and samples, H1 receives no external support from TicToc.**

A tie or loss means deterministic age arithmetic may still be convenient engineering, but there is no benchmark evidence that it changes agent temporal decisions beyond timestamp text.

## Why this gate matters

Most internal WidePresent toys are synthetic. TicToc has an external dataset, human preference labels, an existing evaluation protocol, and a published failure mode involving elapsed real-world time.

That makes it more informative than another custom benchmark even if the result is negative.
