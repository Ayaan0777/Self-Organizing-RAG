# Project Progress vs. Initial Spec — Coverage Audit

Comparison of the current Self-Organising RAG codebase against the original
deep-dive notes. Roughly **85–90% complete** relative to the documented
vision, with several capabilities that go beyond it.

---

## Section §1 — What is Self-Organising RAG?

| Spec item | Status | Where |
|---|---|---|
| **(A) Auto-chunking — semantic boundaries** | ✅ | `rechunk_semantic`, `RecursiveCharacterTextSplitter` |
| **(A) Auto-chunking — entropy** | ✅ | `rechunk_entropy` (vocabulary novelty) |
| **(A) Auto-chunking — structural cues** | 🟡 partial | Separator hierarchy `\n\n → \n → . → …` |
| **(A) Auto-chunking — video/audio temporal markers** | ❌ | Out of scope (text-only system) |
| **(A) Auto-chunking — adaptive compression** | ❌ | Not implemented |
| **(B) Auto-indexing — new docs arrive** | ✅ | `POST /ingest` |
| **(B) Auto-indexing — embedding drift** | ✅ | `auto_indexer/engine.py::detect_stale_chunks` |
| **(B) Auto-indexing — model upgrades** | 🟡 partial | No automated re-embed on model swap |
| **(B) Auto-indexing — bad-query-driven** | ✅ | The cascade itself |
| **(C) Knowledge-gap detection — unanswered** | ✅ | Rule 3 `llm_uncertainty` |
| **(C) Knowledge-gap detection — recurring topics** | 🟡 partial | `UNFIXABLE` events tracked but no topic clustering |
| **(C) Knowledge-gap detection — incorrect answers** | ✅ | Rule 5 `evidence_mismatch` |
| **(C) Knowledge-gap detection — hallucinations** | ✅ | `hallucination_rate` metric + rule 5 |
| **(C) Knowledge-gap detection — latency-driven** | ❌ | Latency tracked but never triggers repair |
| **(D) Self-repair — regenerate embeddings** | ✅ | `reembedder.py` |
| **(D) Self-repair — build new chunks** | ✅ | Cascade S2/S3 |
| **(D) Self-repair — update retrieval params** | ✅ | S1 dynamic K + one-way promotion |
| **(D) Self-repair — retrain adapters** | ❌ | Spec called this "optional" |

---

## Section §3 — Core Components (all 7 present)

| # | Component | Status | Implementation |
|---|---|---|---|
| 1 | Query Listener | ✅ | `logger/query_logger.py` + QueryLog table |
| 2 | Low-Recall Detector | ✅ | 5 rules in `detector/detectors.py` |
| 3 | Auto-Chunker | ✅ | `repair/chunker.py` (3 strategies) |
| 4 | Auto-Embedder | ✅ | `repair/reembedder.py` (partial, snapshot-backed) |
| 5 | Index Monitor + Rebuilder | ✅ | `auto_indexer/engine.py` |
| 6 | Self-Evaluation Engine | ✅ | `controllers/evaluation.py` + `_probe_metrics` |
| 7 | Reporting & Dashboard | ✅ | `dashboard/app.py` (9 pages) |

---

## Section §5 — 6-Month Timeline

| Month | Deliverable | Status |
|---|---|---|
| 1 | Query diagnostic dashboard v1 | ✅ Complete (Overview, Diagnostics, Eval History, Pipeline Config, Adaptation Log) |
| 2 | Low recall event generator | ✅ Complete (5 rules — `user_frustration` was built then removed per user direction) |
| 3 | Auto-chunker pipeline | ✅ Complete (semantic / LLM / entropy strategies, configurable chunk sizes 200–1700) |
| 4 | Auto indexing engine v1 | ✅ Complete (staleness detection, partial re-embed, consistency checks) |
| 5 | RAG self-evaluator | 🟡 Mostly complete (Recall@K ✅, Precision@K ✅, Semantic correctness ✅, Citation correctness ❌, synthetic query generation ❌ — manual `/evaluate-local`) |
| 6 | Self-healing RAG prototype | ✅ Complete (end-to-end pipeline runs, dashboard, before/after graphs) |

---

## Section §7 — Final Output checklist

| Item | Status |
|---|---|
| Self-healing retrieval loop | ✅ |
| Auto detection of poor answers | ✅ |
| Auto chunk refinement | ✅ |
| Automatic embedding refreshing | ✅ |
| Automatic index rebuilding | ✅ |
| Evaluation + Dashboard | ✅ |
| End-to-end automated RAG maintenance pipeline | ✅ |

---

## What you've built that's NOT in the spec

The project went past the brief in several ways the original notes didn't
anticipate:

1. **Ordered 4-strategy cascade with promotion** — the spec described
   "self-repair" generically; implemented as S1→S2→S3→S4 with per-strategy
   success counters and one-way promotion of S1 to the main pipeline once
   it wins 5 times.

2. **GT-backed inline metrics** — `gt_lookup` enriches the QueryLog with
   precision/recall/sufficiency/hallucination for any user query that
   matches `long_ans.json`. Spec only mentioned this for batch eval.

3. **Retrieval gate** — Tier 1 chitchat bypass + Tier 2 LLM classifier so
   the system doesn't hit Pinecone for "hello".

4. **Dynamic K with cliff detection + category-based bounds** — the spec
   said "adaptive retrieval"; this implementation does score-cliff
   detection over `_dynamic_k_selection`.

5. **Snapshot + rollback safety** — `ChunkSnapshot` table backs every
   Pinecone modification so failed repairs are atomic.

6. **Per-strategy K-matched local baselines** — the cascade comparison is
   fair across strategies because each one probes its own baseline at the
   same K it tests.

7. **K-adaptive detection** — rule 2 uses max adjacent gap (K-invariant),
   rule 4 uses `0.65 × top1` (relative threshold).

8. **Background-thread post-processing** — answer returns before
   enrichment/detection finishes, hiding ~1s of latency.

9. **Pipeline latency budget understood and tuned** — ~4.4s → ~3.0s
   through coordinated A+B+C+D optimizations.

---

## Gaps worth flagging

| Gap | Note |
|---|---|
| **Video/audio chunking** | Out of scope — text-only system |
| **Adaptive compression** | Not implemented; could be added if needed |
| **Model-upgrade-driven re-embed** | Auto-indexer detects drift but doesn't auto-trigger on model swap |
| **Topic clustering of UNFIXABLE events** | `source_document` is captured but never aggregated for the "top problematic document areas" dashboard panel the spec called out |
| **Citation correctness metric** | Not implemented (would require LLM to mark citations in answers) |
| **Synthetic query generation** | `/evaluate-local` accepts a JSON dataset but doesn't auto-generate test queries |
| **Latency-based flagging** | Latency is recorded; no rule fires on slow queries |
| **Drift-detection graphs** | Eval History page exists; no specific drift chart |
| **Adapter retraining** | Spec marked this "optional"; skipped |

---

## Bottom line

All **7 core components**, all **6 monthly deliverables** at substantial
depth, and all **7 final-output items** are present. The architecture is
more sophisticated than the spec described — the cascade + promotion
design, the GT enrichment path, and the per-strategy local-baseline
plumbing are upgrades over what was originally sketched.

**What's missing is mostly periphery**: video/audio, citation-style metrics,
the "top problematic doc areas" panel, synthetic query generation, and a
model-upgrade trigger. None of those affect the core "detect → diagnose →
repair → evaluate" loop, which is the part a demo will be judged on.

For a junior 6-month project the deliverable comfortably exceeds the brief.
Good place to stop or pivot toward polish (demo script, eval set,
presentation graphs) rather than more features.

---

*Last updated: 2026-06-18*
