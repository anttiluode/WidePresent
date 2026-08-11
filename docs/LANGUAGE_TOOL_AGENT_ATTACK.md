# Language/tool validity attack

Date: 2026-08-11

This is the first WidePresent benchmark in which the temporal decision is made by an actual language/tool-agent interface rather than by a numeric classifier.

The experiment is:

`experiments/language_tool_validity_benchmark.py`

Its question is operational:

> **Given a timestamped conversation and a cached fact, should the agent answer directly from cache or call a refresh tool?**

The benchmark generator, pairing checks, function-call parser and scorer have been executed locally. The current ChatGPT execution environment has no `OPENAI_API_KEY` and no cached local language model, so genuine model sampling is **not** recorded here yet. No classifier or hand-written policy is substituted for the missing LLM run.

## 1. Three temporal semantics in one language task

The system supplies an explicit validity contract.

### Weather

A cached weather record is valid through `8.0` seconds after its `observed_at` time.

The important timestamp is the world/source observation time, not when the cache result happened to arrive.

### Discourse

A cached conversation-focus resolution is valid through `8` intervening conversation messages after the cache record was received.

Wall-clock seconds do not expire it.

### Reservation

A cached reservation state does not expire merely because time or conversation turns pass.

It becomes invalid only after an explicit reservation-change notification.

This deliberately prevents the benchmark from collapsing to one universal TTL.

## 2. Actual agent action surface

Every case exposes one function:

```text
refresh_source(source, key)
```

The model must do exactly one of two things:

```text
cache valid   -> answer the user's question directly from the cached value
cache invalid -> call refresh_source for the correct source/key
```

The scorer therefore checks more than a binary action label.

For direct reuse, the answer must actually contain the cached value.

For refresh, the model must call the correct function with the correct `source` and `key`.

## 3. Three paired conditions

Every hidden episode is rendered three ways. The conversation history and tool schema are byte-for-byte identical after the system message.

### `raw`

The model receives:

- the explicit validity contract;
- timestamped conversation messages;
- cached `observed_at` and `received_at` times;
- any visible reservation invalidation event.

It has all information required to solve the task, but must perform elapsed-time arithmetic and message counting itself.

### `age_plane`

The same history additionally receives deterministic, label-blind runtime quantities:

```text
valid/world age in seconds
arrival/knowledge age in seconds
intervening conversation-message count
explicit invalidation flag
```

No recommendation is supplied.

This isolates the value of runtime temporal arithmetic from semantic/tool policy.

### `resolver`

The same history receives the deterministic contract decision:

```text
cache_valid_under_contract: yes/no
recommended action: REUSE/REFRESH
```

This is the strongest boring engineering attacker.

If it dominates, the useful product may be to keep validity policy outside the LLM rather than ask the model to rediscover it from timestamps.

## 4. Rate and delay regimes

The source semantics never change.

Only conversation tempo and cache delivery delay change.

```text
iid         0.95..1.05 s/message, ordinary delay
dense       0.25..0.45 s/message, ordinary delay
sparse      1.80..2.40 s/message, ordinary delay
long_delay  nominal tempo, long cache delay
dense_long  dense tempo + long cache delay
```

A final user request occurs after one more local time interval but does not itself count as an intervening structural message.

This makes physical age and structural age genuinely separable.

## 5. Deterministic adversarial characterization

Before spending model calls, `sanity` was run on `1,000` generated paired cases (`200` per regime, seed `42`).

These are **metadata policies, not LLM results**. They only show that the benchmark contains the intended conflicts.

### Action agreement with the source-specific oracle

| policy | IID | dense | sparse | long delay | dense + long delay |
|---|---:|---:|---:|---:|---:|
| arrival-age TTL | 0.905 | 0.855 | 0.680 | 0.840 | **0.635** |
| one world-time TTL | 0.865 | 0.895 | **0.655** | 0.775 | 0.760 |
| one event-count TTL | 0.915 | **0.810** | 0.825 | 0.835 | 0.830 |
| source-specific resolver | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |

The errors are intentionally asymmetric.

- dense conversation makes event distance grow faster than wall time;
- sparse conversation makes wall time grow faster than event distance;
- long cache delay makes arrival recency especially misleading;
- reservation facts break both generic age-decay policies because explicit invalidation, not age, determines validity.

This is the language-agent analogue of the earlier age-plane identifiability attack.

## 6. Operational utility

The scoring utility is the same asymmetric contract used in the earlier event-agent attack:

```text
valid reuse   +1.00
refresh       +0.55
stale reuse   -1.50
```

The LLM scorer reports:

- direct-vs-tool action agreement;
- full task success;
- expected utility;
- dangerous stale-reuse rate;
- refresh rate;
- regime breakdown;
- source breakdown;
- paired condition flips and utility deltas.

`task_success` is stricter than action agreement:

```text
REUSE   -> correct cached value must appear in the direct answer
REFRESH -> refresh_source must receive the correct source and key
```

## 7. Running the language-agent gate

Generate a small pilot:

```bash
python experiments/language_tool_validity_benchmark.py generate \
  --output language_cases.jsonl \
  --per-regime 20
```

Verify pairing and benchmark structure:

```bash
python experiments/language_tool_validity_benchmark.py sanity \
  --input language_cases.jsonl
```

Run an actual function-calling model:

```bash
python experiments/language_tool_validity_benchmark.py run-openai \
  --input language_cases.jsonl \
  --output language_responses.jsonl \
  --model gpt-5 \
  --conditions raw age_plane resolver
```

Then score:

```bash
python experiments/language_tool_validity_benchmark.py score \
  --cases language_cases.jsonl \
  --responses language_responses.jsonl
```

Use `--limit` for a cheap first pilot.

The runner uses function calls rather than asking the model to print an artificial `REUSE/REFRESH` label.

## 8. Pre-registered interpretation

### If `raw ~= age_plane`

The model can already operationalize timestamps/message structure well enough. Derived age-plane arithmetic is convenience, not evidence for a new runtime representation.

### If `age_plane > raw`

There is a practical result:

> deterministic temporal preprocessing makes the same model more reliable without adding semantic/tool labels.

This still does not imply a new neural architecture.

### If `resolver > age_plane`

The likely engineering conclusion is stronger and more boring:

> **do not ask the language model to solve temporal validity when a deterministic runtime can solve it first.**

The LLM should consume the resolved validity state rather than repeatedly recompute clocks, message counts and invalidation rules.

### If the resolver condition still fails

Then the remaining error is no longer temporal inference. It is instruction following / tool-selection compliance or output formatting.

That distinction is useful because it separates temporal-state failures from general agent failures.

## 9. What this benchmark does not test

The contract track tells the model the source semantics explicitly.

It therefore does **not** test whether an LLM can discover from noisy experience that weather belongs to seconds while discourse belongs to event distance.

That identification problem was already tested numerically in:

- `docs/TIME_STRUCTURE_IDENTIFIABILITY.md`;
- `docs/EVENT_AGENT_AGE_PLANE_ATTACK.md`.

The language benchmark instead isolates the next question:

> **Once the temporal semantics are known, where should the arithmetic and validity decision live: inside the language model, in a label-blind age side channel, or in the runtime resolver?**

That is now the external-facing WidePresent gate.

## 10. Current status

**Benchmark implementation: complete.**

**Pairing/sanity/scoring path: locally executed.**

**Actual LLM samples: not yet executed in this environment because no API key or local model is available.**

Do not record a synthetic classifier result as the answer to this gate.
