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

## 7. What might remain interesting

No novelty claim is made yet. Candidate remainder:

1. A **hard moving temporal origin** (`now`) shared across modalities.
2. Past state indexed by **age from now**, not merely arbitrary recurrent state.
3. Predicted state indexed by **time-to-now**, so forecasts mathematically mature into observations as the clock advances.
4. A content-independent base clock that remains invariant while attention/gating/content processing operates inside the frame.
5. A direct TCI-style diagnostic of time-vs-structure yoking in online agents.

Each of those may also have close prior art in control, signal processing, robotics, event cameras, predictive coding or world models. The next literature search should explicitly try to kill this remainder.
