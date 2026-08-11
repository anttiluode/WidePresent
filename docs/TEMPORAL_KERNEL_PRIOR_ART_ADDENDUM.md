# Temporal-kernel prior-art addendum — 2026-08-11

The temporal-kernel pivot immediately collided with work published/preprinted in 2026. This note records the collision rather than letting the README quietly absorb the same ideas as if they emerged here first.

## TGMS — deterministic bitemporal operators for agents

**Xiaofei Zhang, _TGMS: An Agent-Native Bi-Temporal Graph Management System_, July 2026.**

TGMS is extremely close in spirit to the "boring temporal kernel" direction:

- valid time and transaction time are separate;
- temporal operations are typed and deterministic;
- the LLM plans and verbalizes while the system performs graph/time computation;
- claims are checked against an execution trace;
- evidence completeness is explicitly important because correct arithmetic over truncated evidence can still mislead.

This is a strong warning against claiming that WidePresent invented "keep temporal computation outside the LLM" or "give agents bitemporal operators."

Preprint: https://arxiv.org/abs/2607.10265

## Graph-native bitemporal conversational memory

**Alp Niksarli & Gopesh Baheti, _A Graph-Native Bitemporal Memory Store for Conversational AI Agents_, July 2026.**

Uses immutable/versioned memory records with both valid-time and transaction-time intervals and evaluates point-in-time retrieval on LongMemEval.

Preprint: https://arxiv.org/abs/2607.26520

Again, bitemporal agent memory itself is occupied prior art.

## Engram — bitemporal memory with deterministic contradiction handling

**Liuyin Wang, _Less Context, More Accuracy: A Bi-Temporal Memory Engine for LLM Agents..._, June 2026.**

Uses a bitemporal knowledge graph, retains provenance/supersession, and separates a fast lossless episode path from asynchronous fact extraction. Reported gains depend on a broader hybrid retrieval/assembly system, so it should not be read as an isolated proof that bitemporality alone improves agents.

Preprint: https://arxiv.org/abs/2606.09900

## Don't Ask the LLM to Track Freshness

**Vikas Reddy & Sumanth Challaram, _Don't Ask the LLM to Track Freshness: A Deterministic Recipe for Memory Conflict Resolution_, May 2026.**

This paper is directionally aligned with WidePresent's kernel pivot: instead of asking the LLM to resolve which conflicting fact is newer, deterministic version/timestamp logic is used during assembly. The reported pipeline changes several components jointly, and the authors explicitly note that isolating the resolver remains future work, so this is not clean causal evidence for one operator.

Still, it reinforces a project discipline:

> **If time/version arithmetic is exactly known, benchmark deterministic resolution before training a model to rediscover it.**

Preprint: https://arxiv.org/abs/2606.01435

## Timely Machine — wall-clock time as an agent resource

**_Timely Machine: Awareness of Time Makes Test-Time Scaling Agentic_, ACL 2026.**

This line reframes test-time scaling in wall-clock terms, where tool latency decouples generation length from elapsed time, and trains policies to adapt strategy to time budgets.

It is another independent reason not to equate token/inference steps with world time.

ACL Anthology: https://aclanthology.org/2026.acl-long.211/

## Consequence for WidePresent

The surviving claim boundary is narrow:

```text
not new: clocks
not new: elapsed-time features
not new: valid time / transaction time
not new: deterministic temporal operators
not new: bitemporal conversational memory
not new: late-observation filtering
not new: LLM temporal blindness
not new: timestamp prompting

still testable:
1. passive timestamp text vs label-blind derived runtime state on TicToc;
2. scalar temporal kernel vs bitemporal typed state on source-confusion tasks;
3. kernel/bitemporal state vs bounded moving wide-present projection;
4. TCI-style measurement of whether learned agent representations become time- or structure-yoked.
```

If all four are null, the project has still produced a useful negative map connecting several normally separate literatures.
