# Prior-art map — WidePresent

Date: 2026-08-11

This is a claim-boundary document, not a bibliography dump. The point is to identify which parts of WidePresent are already known and therefore must not be claimed as inventions.

## 1. Fixed clocks and multirate recurrence

### Clockwork RNN — Koutník et al., ICML 2014
Hidden state is partitioned into modules that update at prescribed clock rates. This already establishes that neural computation can be explicitly organized by external temporal granularity.

- Paper: https://proceedings.mlr.press/v32/koutnik14.html

**Boundary:** WidePresent cannot claim "neural modules with clocks" or "different neural timescales."

### Phased LSTM — Neil, Pfeiffer & Liu, NeurIPS 2016
Adds an oscillatory time gate to LSTM cells. The gate is a function of time and supports irregular/asynchronous event streams.

- Paper: https://proceedings.neurips.cc/paper/2016/hash/5bce843dd76db8c939d5323dd3e54ec9-Abstract.html

**Boundary:** WidePresent cannot claim "oscillatory time gating," "continuous timestamps in RNNs," or "handling asynchronous sensors by a time gate."

## 2. Explicit time representations

### Time2Vec — Kazemi et al., 2019
A model-agnostic learned vector representation of time.

- Preprint: https://arxiv.org/abs/1907.05321

### Continuous/irregular-time Transformers and ODE/CDE families
There is now a large literature that makes timestamps and continuous dynamics explicit. One example is ContiFormer (2024), which combines Transformer relation modelling with continuous-time dynamics.

- ContiFormer: https://arxiv.org/abs/2402.10635

**Boundary:** "Transformers do not know time" is false as a general statement. The narrower target is token/event-centric cognition where elapsed time is not a mandatory state coordinate.

## 3. Sliding continuous history

### Legendre Memory Unit — Voelker, Kajić & Eliasmith, NeurIPS 2019
LMU maintains a compressed representation of a sliding continuous-time history using an ODE whose state maps onto a Legendre basis over the past window.

- Paper: https://papers.nips.cc/paper/9689-legendre-memory-units-continuous-time-representation-in-recurrent-neural-networks

### HiPPO — Gu et al., 2020
Generalizes the online projection view of history; a foundational route into modern state-space sequence models.

- Preprint: https://arxiv.org/abs/2008.07669

**Boundary:** A "temporally wide past" is emphatically prior art. WidePresent must beat or complement compressed continuous-history models, not ignore them.

## 4. Biological time coordinates

### Hippocampal time cells
Time cells fire at successive moments in structured experiences, including empty gaps. Eichenbaum's review describes evidence that hippocampal populations encode elapsed time and temporal organization of memory.

- Review: https://pmc.ncbi.nlm.nih.gov/articles/PMC4348090/
- Human time cells: https://pmc.ncbi.nlm.nih.gov/articles/PMC7668099/

### Theta sequences
Within individual theta cycles, hippocampal place-cell populations form ordered sequences. Work has described sequences spanning past/current/future locations and goal-dependent look-ahead.

- Foster & Wilson 2007: https://pubmed.ncbi.nlm.nih.gov/17663452/
- Cei et al. 2014: https://pubmed.ncbi.nlm.nih.gov/24667574/
- Wikenheiser & Redish 2015: https://pubmed.ncbi.nlm.nih.gov/25559082/

A 2026 review argues that theta cycles can carry present state followed by forward simulations, while emphasizing open questions and controversy.

- Damphousse et al. 2026: https://pubmed.ncbi.nlm.nih.gov/42421588/

**Boundary:** A cyclic representation containing recent/current/anticipated state is biologically plausible but not novel as a general idea.

## 5. Extended present / retention-protention

### Time consciousness — Kent & Wittmann, 2021
Argues that conscious experience is temporally extended, continuous and field-like rather than exhausted by brief discrete functional moments. Discusses the "experienced present" on seconds-long scales.

- https://academic.oup.com/nc/article/2021/2/niab011/6224347

### Computational phenomenology / active inference — Bogotá & Djebbara, 2023
Explicitly analyzes Husserlian retention, present and protention in active inference. Their paper describes a present-moment Markov blanket integrating immediate past and future in an asynchronous structure.

- https://pubmed.ncbi.nlm.nih.gov/36937108/

**Boundary:** WidePresent cannot claim the conceptual combination "past + now + anticipated future = extended present."

## 6. Time-yoked vs structure-yoked integration

### Norman-Haignere et al., Nature Neuroscience 2025
This is the most useful empirical paper for the project.

- DOI: https://doi.org/10.1038/s41593-025-02060-8
- Open article: https://www.nature.com/articles/s41593-025-02060-8
- Authors' code/data are linked from the paper.

They time-stretched/compressed speech and used temporal context invariance (TCI) to estimate neural integration windows. Human auditory-cortex integration was predominantly yoked to absolute time. Their DeepSpeech2 model, in contrast, developed increased structure yoking across trained layers.

**Why it matters here:** it provides an assay. We can measure whether a learned architecture uses absolute-time or structure/event-relative windows without assuming either is automatically superior.

## 7. Time-aware world models and delayed observations

### Time-Aware World Model — Nhu, Son & Lin, ICML 2025
TAWM explicitly conditions world-model dynamics on time-step size `dt` and trains across varied `dt`, improving control and prediction across observation rates.

- https://proceedings.mlr.press/v267/nhu25a.html

**Boundary:** conditioning a learned world model on elapsed time and seeking sampling-rate robustness is direct prior art.

### Model-Based RL under Random Observation Delays — Karamzade et al., L4DC 2026
Studies sensors whose observations can arrive late and out of sequence. The paper reports that naive stacking of past observations is insufficient and introduces a delay-aware model-based filtering approach.

- https://proceedings.mlr.press/v331/karamzade26a.html

**Boundary:** delay-aware state estimation and robustness to delay-distribution shift are not new WidePresent claims.

### Out-of-sequence measurement filtering
Delayed and out-of-order measurements have a long literature in Bayesian/Kalman filtering, smoothing and sensor fusion.

- Challa, Evans & Wang 2003: https://doi.org/10.1016/S1566-2535(03)00037-X

**Boundary:** retroactively incorporating delayed measurements into a state estimate is established estimation theory.

## 8. Event time, processing time and bitemporal memory

### Stream processing
Apache Flink explicitly distinguishes event time (when an event happened), ingestion time and processing time, and uses watermarks to reason about event-time progress and late/out-of-order records.

- https://nightlies.apache.org/flink/flink-docs-stable/docs/learn-flink/streaming_analytics/

**Boundary:** distinguishing event/world time from arrival/processing time, and tracking completeness of an interval, is mature stream-processing machinery.

### Temporal databases
Bitemporal databases distinguish **valid time** (when a fact is true in modeled reality) from **transaction time** (when it is stored/known to the database). Formal treatments go back decades.

- Clifford & Isakowitz 1992: https://archive.nyu.edu/jspui/handle/2451/14356

### Bitemporal conversational-agent memory — 2026
Very recent work already applies valid-time / transaction-time memory to conversational AI, including a graph-native bitemporal memory store and TGMS.

- https://arxiv.org/abs/2607.26520
- https://arxiv.org/abs/2607.10265

**Boundary:** "AI memory should have world time and knowledge time" is already active prior art.

## 9. What might remain interesting

No novelty claim is made yet. The surviving research object is becoming narrower:

1. A **temporally typed working state around a moving `now`**, rather than only a long-term bitemporal database.
2. A projection that combines event/world time, knowledge/arrival time, observation completeness, uncertainty and future deadlines in the state consumed by a learned online policy/model.
3. Controlled tests of whether keeping these temporal types explicit reduces source confusion: observed-now vs remembered-past vs predicted-future vs newly learned old evidence.
4. A TCI-style diagnostic of learned time-vs-structure yoking, treated as a measurement rather than a brain-likeness objective.
5. A fair comparison against time-aware world models, delay-aware filters, timestamp Transformers, LMU/HiPPO, and ordinary database/stream-processing infrastructure.

The likely contribution, if any, is therefore **not a new notion of time**. It would be evidence that importing several mature temporal semantics into the *working representation of a learned agent* produces a useful inductive bias or diagnostic capability.
