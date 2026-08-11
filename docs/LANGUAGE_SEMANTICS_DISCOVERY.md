# Language-agent temporal-semantics discovery

Date: 2026-08-11

This benchmark removes the explicit source validity contract from the language/tool task.

The question is now:

> **Can a language agent infer what "old" means for each source from prior audited experience, and does rate-diverse experience let it identify the causal temporal coordinate rather than a correlated proxy?**

The experiment is:

`experiments/language_semantics_discovery_benchmark.py`

It is designed as the language-agent counterpart of:

- `docs/TIME_STRUCTURE_IDENTIFIABILITY.md`;
- `docs/EVENT_AGENT_AGE_PLANE_ATTACK.md`;
- `docs/COUNTERFACTUAL_LANGUAGE_TOOL_PAIRS.md`.

## 1. Arbitrary source labels

The sources are deliberately named:

```text
alpha
beta
gamma
```

rather than weather/discourse/reservation.

This removes the obvious semantic prior that "weather probably expires in wall time."

The hidden stable rules are:

```text
alpha
    valid through 8 seconds after observed_at

beta
    valid through 8 intervening conversation messages

gamma
    valid until an explicit source-change notification
```

Those rules are **not** stated in the `raw` or `age_plane` discovery prompts.

The model must infer them from past audited episodes.

## 2. Audited experience

The system prompt contains past episodes.

Each audit includes:

- a cached record;
- `observed_at`;
- `received_at`;
- timestamped neutral intervening conversation messages;
- an audit decision time;
- the audit outcome: `STILL VALID` or `STALE`.

The examples therefore supply supervision through experience rather than through an explicit rule.

The final episode is then presented using the same language/tool action surface as the existing benchmark:

```text
cache valid
    -> answer directly from cached value

cache stale
    -> call refresh_source(source, key)
```

## 3. Narrow experience is deliberately underidentified

In the `narrow` audit context, alpha and beta examples occur at exactly:

```text
1 second / intervening message
```

with `observed_at == received_at`.

Therefore:

```text
world age in seconds == intervening-message count
```

on every alpha/beta audit.

The deterministic candidate-rule fit is:

| source | seconds rule | event-count rule |
|---|---:|---:|
| alpha | **1.000** | **1.000** |
| beta | **1.000** | **1.000** |

So the audit evidence itself cannot tell the model which coordinate is causal.

If a language model chooses:

```text
alpha -> seconds
beta  -> events
```

under this condition, that choice comes from its prior/guess or from some other bias, not from statistical identification in the examples.

This is the language version of the rank-1 result:

```text
single rate
    Δt proportional to Δn
    causal yoking underidentified
```

## 4. Wide experience breaks the confound

The `wide` audit set contains examples such as:

```text
12 messages in 3.6 seconds
4 messages in 10 seconds
```

so seconds and message count make opposing predictions.

The deterministic candidate fits become:

| source | seconds | events | invalidation |
|---|---:|---:|---:|
| alpha | **1.000** | 0.500 | 0.500 |
| beta | 0.500 | **1.000** | 0.500 |
| gamma | 1.000 | 0.500 | **1.000** |

Therefore the correct coordinate is now identifiable from the demonstrations.

This gives a clean prediction:

> if a model truly learns source temporal semantics from experience, wide audits should produce helpful action changes on rate-shift tests relative to narrow audits.

## 5. Final cases are exactly matched

This is a critical control.

The final hidden case generator is independent of `--experience`.

With the same seed and generation arguments, narrow and wide files have identical:

- source;
- key;
- cached value;
- timestamps;
- message history;
- world age;
- arrival age;
- message count;
- invalidation;
- oracle action.

The command:

```bash
python experiments/language_semantics_discovery_benchmark.py compare \
  --left discovery_narrow.jsonl \
  --right discovery_wide.jsonl
```

asserts that identity field-by-field.

So any behavioral difference is caused by the prior audit evidence, not an easier final test distribution.

## 6. Three prompt conditions

Each narrow or wide context still has three paired conditions.

### `raw`

The model sees the audited timestamped mini-transcripts and the final timestamped conversation.

It must:

- infer source semantics;
- do timestamp arithmetic;
- count messages;
- apply the inferred rule.

### `age_plane`

The exact same audits additionally contain deterministic:

```text
world age
arrival age
intervening-message count
invalidation flag
```

The final case receives the same arithmetic side channel.

No semantic source rule and no reuse/refresh recommendation are supplied.

This separates:

```text
semantic identification
from
temporal arithmetic/counting burden
```

### `resolver`

The true stable source contract is supplied explicitly.

The final action is still left to the model.

This is the boring upper attacker:

> if the runtime already knows the semantic validity rule, do not force the LLM to infer it repeatedly.

## 7. Test regimes

Final cases cover the same operational shifts as the known-contract benchmark:

```text
iid
dense
sparse
long_delay
dense_long
```

The hidden semantics never change.

Only interaction tempo and delivery delay change.

This matters because a wrong causal proxy may look fine IID and fail only when:

```text
seconds / message
```

changes.

## 8. Running it

Generate narrow experience:

```bash
python experiments/language_semantics_discovery_benchmark.py generate \
  --experience narrow \
  --output discovery_narrow.jsonl \
  --per-regime 30 \
  --seed 777
```

Generate wide experience:

```bash
python experiments/language_semantics_discovery_benchmark.py generate \
  --experience wide \
  --output discovery_wide.jsonl \
  --per-regime 30 \
  --seed 777
```

Check the evidence geometry:

```bash
python experiments/language_semantics_discovery_benchmark.py sanity \
  --experience narrow \
  --input discovery_narrow.jsonl

python experiments/language_semantics_discovery_benchmark.py sanity \
  --experience wide \
  --input discovery_wide.jsonl
```

Assert matched final cases:

```bash
python experiments/language_semantics_discovery_benchmark.py compare \
  --left discovery_narrow.jsonl \
  --right discovery_wide.jsonl
```

Run the existing language/tool model harness on each file:

```bash
python experiments/language_tool_validity_benchmark.py run-openai \
  --input discovery_narrow.jsonl \
  --output discovery_narrow_responses.jsonl \
  --model gpt-5 \
  --conditions raw age_plane resolver

python experiments/language_tool_validity_benchmark.py run-openai \
  --input discovery_wide.jsonl \
  --output discovery_wide_responses.jsonl \
  --model gpt-5 \
  --conditions raw age_plane resolver
```

The existing scorer can score each file separately.

Then compare the matched behaviors:

```bash
python experiments/language_semantics_discovery_benchmark.py compare-responses \
  --narrow-cases discovery_narrow.jsonl \
  --narrow-responses discovery_narrow_responses.jsonl \
  --wide-cases discovery_wide.jsonl \
  --wide-responses discovery_wide_responses.jsonl
```

## 9. Narrow-to-wide response metrics

For each prompt condition the comparator reports:

```text
narrow accuracy
wide accuracy
accuracy delta

narrow utility
wide utility
utility delta

action_changed
helpful_flips
harmful_flips
```

It also breaks the changes down by:

```text
source
regime
```

The important signal is not merely that wide experience changes answers.

It is:

```text
wrong under narrow
correct under wide
```

without a corresponding growth in:

```text
correct under narrow
wrong under wide.
```

## 10. Comparator dry run

The comparison code was attacked using artificial responses.

The narrow response policy was deliberately wrong:

```text
alpha -> event count
beta  -> world seconds
gamma -> invalidation
```

while wide responses used the correct oracle.

On a 30-case smoke set the comparator reported:

```text
narrow action accuracy   0.767
wide action accuracy     1.000
delta                   +0.233

helpful flips            7
harmful flips             0
```

The improvements were concentrated in alpha and beta.

Gamma remained correct in both.

The rate/delay-shift regimes carried most of the benefit.

These are scorer tests, not LLM results.

## 11. Pre-registered interpretations

### Narrow `raw` performs poorly; wide `raw` improves

This is the cleanest support for the identifiability story at the language-agent level.

The model needed rate-diverse experience to infer the causal temporal coordinate.

### Narrow `raw` already performs well

Then the model is imposing a useful prior despite underidentified audit evidence.

Because source labels are arbitrary, this deserves examination rather than immediate credit to learned semantics.

The counterfactual fingerprint suite can help determine which proxy it used.

### `age_plane > raw`, especially under wide experience

Then deterministic temporal arithmetic helps the model exploit information that is already statistically identifiable.

That is a runtime representation result, not an architecture result.

### Wide `raw ~= age_plane`

Then a capable model can both infer the semantics and perform the arithmetic/counting from timestamped experience.

The side channel is optional engineering convenience.

### `resolver > wide age_plane`

Then the strongest conclusion is again the boring one:

> if source validity semantics are known to the runtime, resolve them outside the LLM.

### Wide experience does not improve alpha/beta

Then either:

- the language model failed to extract the identifiable rule from demonstrations;
- the in-context audit format is inadequate;
- or the task is dominated by generic instruction/tool failures.

The explicit-contract `resolver` condition separates those possibilities.

## 12. What this adds to the WidePresent line

The project has now separated three questions that were previously blurred together:

```text
1. REPRESENTATION
   Are the relevant temporal coordinates preserved?

2. IDENTIFICATION
   Has experience varied enough to determine which coordinate matters?

3. DECISION
   Can the language/tool agent apply the known or inferred validity rule?
```

A timestamp can solve only the first part.

An age-plane side channel can simplify the first part.

Rate diversity is needed for the second when coordinates are confounded.

A validity resolver can remove the second and much of the third from the LLM entirely.

That decomposition is currently more useful than proposing another temporal neural layer.
