# Patih — Architecture Overview (v2)

> A narrative explanation of how Patih works end-to-end, for a peer-level reader or a new
> contributor. It is the short companion to the full reference in
> [`docs/patih_v3.pdf`](docs/patih_v3.pdf), which carries the theory, the per-module
> walkthrough, the mathematics, and the annotated bibliography. Read this first (~15 min),
> then the PDF when you need depth.

**Version 2 (2026-05-29).** v1 described a *single-document* assistant deployed to a public
cloud VM. The project has since pivoted to **local-first** and a **multi-document** corpus.
This document reflects the current system; the deployment history is summarised in §8.

**Language.** English, with Indonesian legal terms (*Pasal*, *ayat*, *huruf*, *Permensos*,
…) kept verbatim because article citations must stay in the original Indonesian.

---

## 1. What it is

Patih is a question-answering assistant over the Indonesian Ministry of Social Affairs
(*Kemensos*) regulatory corpus. A user asks a natural-language question (Indonesian or
English); Patih answers with **valid, traceable article citations** back to the source
regulation — e.g. *"(Pasal 5 ayat (2) huruf a)"*.

- **Corpus**: 22 documents — 19 article-structured regulations + 3 reference documents
  (an RPJMN summary, two SOPs). Anchor document: **Permensos 8/2023** (human trafficking &
  distressed migrant workers). The corpus grows by dropping PDFs into a watched folder.
- **Runs locally**: data, embeddings, retrieval, and indexes live on the laptop; only the
  LLM call leaves the machine (§8).
- **Honest for a legal domain**: every answer carries a citation and a confidence badge;
  out-of-scope questions are refused rather than answered.

**Constraints that shaped it**: zero monetary cost (free-tier LLMs + FOSS), CPU-only,
Indonesian language, verifiability first.

---

## 2. High-level: which kind of RAG

Patih is a **single-pass, Hybrid (dense + sparse, RRF-fused) Parent-Document RAG**,
augmented with two domain steps — cross-reference resolution and an always-on definitions
article — and **gated by a post-hoc citation-and-grounding validation layer**
("citation-enforced RAG"). It is deliberately advanced on the *retrieval* axes and simple
on the *control-flow* axis (no agentic/self/corrective loop in Phase 1).

The guiding principle is **retrieval is the ceiling**: an answer can be no better than the
passages retrieved, so most of the engineering lives in retrieval and parsing, and the LLM
is treated as a constrained summariser that is validated afterwards.

Four things make it different from a generic chatbot:

- **Document structure is preserved** — the system knows "Pasal 5 ayat (2) huruf b" is a
  hierarchical node, not loose text.
- **Citation enforcement** — the LLM must cite an article for every claim, and cited
  articles that do not exist are flagged.
- **Provider fallback chain** — if the primary LLM is rate-limited, it falls through to the
  next free-tier provider automatically.
- **Bilingual-aware** — an English query is translated to Indonesian for retrieval, while
  the article citations stay in the original Indonesian.

---

## 3. End-to-end data flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  OFFLINE: INGESTION (per new document — via the inbox watcher)               │
└─────────────────────────────────────────────────────────────────────────────┘

  Drop <doc>.pdf  (+ optional <doc>.pdf.meta.json) into data/raw/inbox/
       │            (if the sidecar is missing, the watcher auto-generates one)
       ▼
  ┌─────────────────┐
  │  PDF Loader     │  ← PyMuPDF native text; Tesseract OCR (ind) fallback for scans
  └────────┬────────┘
           ▼
  Route by doc_type:
     regulation ──► Structure Parser (regex BAB/Bagian/Pasal/ayat/huruf → AST)
                    + the five parser fixes (§4.1)
     reference  ──► Generic Ingester (section/page chunks, pasal = null)
           │
           ▼
  Parent/child chunks  {doc_id, parent_id, bab, pasal?, ayat, huruf, section_title?,
                        source_page, text, text_for_embed}
           │
           ├───────────────────────────────┐
           ▼                               ▼
  ┌─────────────────────┐      ┌─────────────────────┐
  │  e5-large ONNX FP32 │      │   BM25 (rank_bm25)  │
  │  (dense embedder)   │      │   no stemming       │
  └──────────┬──────────┘      └──────────┬──────────┘
             ▼                            ▼
  ┌─────────────────────┐      ┌─────────────────────┐
  │ Chroma (cosine,HNSW)│      │ Pickled BM25 +      │
  │ data/chroma/        │      │ parent_lookup.json  │
  └─────────────────────┘      └─────────────────────┘
        (parsed JSON in data/parsed/ is the source of truth; indexes rebuild from it)


┌─────────────────────────────────────────────────────────────────────────────┐
│  ONLINE: QUERY (per request)                                                 │
└─────────────────────────────────────────────────────────────────────────────┘

  User query  ──►  Lang detect (lingua-py, default ID)
                        │  [if EN: translate query → ID for retrieval]
                        ▼
        ┌──────────────────────┐        ┌──────────────────────┐
        │  Dense (e5 + Chroma) │        │  Sparse (BM25)       │
        │  top_k_dense = 15    │        │  top_k_sparse = 15   │
        └──────────┬───────────┘        └──────────┬───────────┘
                   └─────────────┬─────────────────┘
                                 ▼
                   Reciprocal Rank Fusion (k = 60) → top_k_fused = 8 (12 in eval)
                                 ▼
                   Parent expansion (child → whole Pasal / section), de-dup
                                 ▼
                   Cross-reference resolver (per-document, cap 3)
                                 ▼
                   Always-on definitions article (Pasal 1 of the DOMINANT document)
                                 ▼
                   Final, per-document-labelled context
                                 ▼
        ┌─────────────────────────────────────────────────┐
        │  LiteLLM gateway (token-bucket + fallback chain) │
        │   Groq Llama 3.3 70B   (workhorse)               │
        │     → Gemini 2.5 Flash → Cerebras Qwen 3         │
        │     → OpenRouter (free slot; currently empty)    │
        └────────────────────┬────────────────────────────┘
                             ▼
                   Raw answer (with citations), T = 0.1
                             ▼
        ┌────────────────────────────────────────┐
        │  Validators (defence in depth)          │
        │   Citation extractor → whitelist        │
        │   Entity-Grounding (EG) + Relation (RP) │
        │   threshold gate → HITL flag + badge    │
        └────────────────────┬───────────────────┘
                             ▼
            🟢/🟡/🔴 badge + answer + click-through citation cards
                  (low-confidence cases → data/hitl_queue.jsonl)
```

Warm latency is dominated by the LLM round-trip; local retrieval is ~150 ms. The first
query after launch is slow (~7–9 s) because the FP32 embedder loads (§8).

---

## 4. The seven layers

The system is seven layers of small single-purpose modules so each can be tested and
replaced independently, the corpus can grow without touching serving, and failures stay
localised.

### Layer 1 — Ingestion (`app/ingest/`)

Convert a PDF into structured, chunked, registered data.

- `pdf_loader.py` — PyMuPDF native extraction; auto-falls back to Tesseract OCR (`ind`) for
  scanned PDFs (the helper auto-locates the binary + `models/tessdata/`, so OCR is
  unattended, including from the watcher).
- `structure_parser.py` — builds the BAB/Bagian/Pasal/ayat/huruf AST by regex. **§4.1**
  lists the five real-world fixes. Zero parsed articles signals "not a regulation".
- `chunker.py` — parents (*Pasal*) and children (*ayat*/*huruf*) with stable ids
  (`doc_id::pasalN::ayatM::hurufX`), `parent_id`, page numbers, the `"passage: "`-prefixed
  embed text, and an `always_on` tag on Pasal 1.
- `generic_ingest.py` — the article-less ingester (SOP/statistics/RPJMN): heading/page
  *sections* with size-windowed children, same schema with `pasal = null` and a
  `section_title`.
- `doc_registry.py` — one record per document, SHA-256 keyed, written atomically.
- `validators.py` — parse-time quality checks (return flags, never crash).
- `cli.py` — single-document ingest entry point; reads a `<pdf>.meta.json` sidecar.

#### 4.1 Five parser fixes (why parsing Indonesian law is hard)

1. **Penjelasan inflation** — the *Penjelasan* (elucidation) repeats every article,
   doubling the count. Fix: truncate at the heading, but only past 40 % of the document.
2. **Omnibus laws** — an omnibus (UU 6/2023, *Cipta Kerja*) embeds dozens of other laws
   (~700 spurious articles). Fix: exclude it from the corpus.
3. **Pre-BAB articles** — short regulations place articles before the first BAB. Fix: also
   scan the pre-BAB region.
4. **Trailing-period headers** — older laws write "Pasal 5."; the pattern tolerates the dot.
5. **Cross-chapter duplicates** — keep the first, drop the rest.

### Layer 2 — Retrieval (`app/retrieval/`)

Given a query, return the most relevant articles with their full parent context.

- Hybrid because legal questions need **both** meaning (dense, e5-large) **and** exact
  tokens (sparse BM25 — literal "Pasal 5", rare terms like "TPPO"). No stemming (the
  Indonesian stemmer over-reduces legal terms).
- `hybrid.py` fuses the two ranked lists with **RRF (k = 60)** — ranks only, so the two
  incomparable score scales never need calibration.
- `parent_expander.py` maps fused children to their parent *Pasal* (or section), de-dups,
  keeps the best child score.
- `cross_ref_resolver.py` pulls in up to 3 articles a retrieved article references —
  **scoped to the same document** (key for multi-doc; §5).
- `always_on.py` prepends Pasal 1 (definitions) **of the dominant retrieved document**.
- `pipeline.py` orchestrates; `indexer.py` builds Chroma + BM25 from `data/parsed/*.json`,
  de-duplicating chunk ids (last-write-wins); `vector_store.py`/`bm25_store.py` wrap the
  stores; `embedder.py` runs e5-large via ONNX (backend pinned by `EMBEDDER_BACKEND`).

### Layer 3 — LLM gateway (`app/llm/`)

Generate the answer from (query + context), surviving free-tier limits.

- `gateway.py` — LiteLLM router, four providers in priority order with a fallback chain;
  records per-attempt latency and the provider used. **Groq Llama 3.3 70B is the
  workhorse** (Gemini was demoted to fallback after its real free limit turned out to be
  ~20 requests/*day*; §9).
- `rate_limiter.py` — per-provider token buckets (proactive throttle before a hard 429).
- `prompts.py` — loads the strict system prompt; labels each context chunk per document
  ("[Pasal N — BAB X — *label*]" or "[*section* — hal. P — *label*]") so the model
  attributes each fact to the right law; T = 0.1, max_tokens = 1500.
- `lang_detect.py` / `translator.py` — language detection and EN→ID query translation
  (SQLite-cached).
- `generator.py` — the end-to-end orchestrator the UI/sidecar call.

### Layer 4 — Validators (`app/validators/`)

Defence in depth: never trust the LLM on its face. Independent cheap checks stack, so their
miss-rates multiply down.

- `citation_extractor.py` — regex pull of every "Pasal N [ayat (M)] [huruf x]".
- `whitelist_validator.py` — range check + Chroma existence (valid if found in **any**
  retrieved document; huruf fallback confirms the letter appears in the ayat text).
- `entity_grounding.py` (EG) — fraction of the answer's legal terms also in context.
- `relation_preservation.py` (RP) — per-sentence content overlap with context.
- `pipeline.py` — threshold gate: invalid citation, or EG < 0.95, or RP < 0.85 → raise the
  HITL flag, colour the badge, append to `data/hitl_queue.jsonl`.

### Layer 5 — UI (`app/ui/`)

- `chainlit_app.py` — chat app (welcome + background generator pre-warm; per message a
  thinking placeholder → generator → badge + answer → citation cards → persist → feedback).
- `components.py` — confidence badge (🟢/🟡/🔴) and per-document citation cards.
- `history.py` — SQLite conversation store.
- `api_sidecar.py` — a FastAPI app mounted inside the Chainlit ASGI server exposing
  `/health` and `/api/query` (for smoke tests, monitoring, evaluation) — no second server.

### Layer 6 — Evaluation (`app/eval/`)

Quantify accuracy (§10). A tiered 50-question set drives citation accuracy
(precision/recall/F1/Jaccard, micro + per-tier), refusal precision, latency percentiles,
and RAGAS Faithfulness/Answer-Relevancy; `report.py` renders the report.

### Layer 7 — Tooling & observability (`tools/`, infra)

- `inbox_watcher.py` — polling daemon over `data/raw/inbox/`; on a stable PDF it routes by
  `doc_type`, ingests, indexes incrementally, and moves files to `data/raw/` (failures
  quarantined to `data/raw/failed/`). If a sidecar is missing it **auto-generates** one
  from the filename (best-effort, flagged for review); `--no-auto-meta` requires an explicit
  sidecar.
- `triage_pdfs.py`, `generate_meta_sidecars.py`, `bulk_ingest.py`, `smoke_multidoc.py` —
  corpus tooling.
- Langfuse is wired for per-request tracing (optional, local).

---

## 5. Multi-document routing (the v2 core)

With many laws in one index, a question competes across all of them; the hybrid ranker
surfaces the most relevant articles regardless of source, and the **dominant retrieved
document wins** — implicit routing, no separate classifier. The central multi-document risk
is **Document-Level Retrieval Mismatch (DRM)**: pulling text from the *wrong* source.

Patih counters DRM by keeping cross-references and the definitions article
**document-scoped**: a "Pasal 5" inside a given UU resolves to *that* UU's Pasal 5, and the
always-on Pasal 1 is the dominant document's, not a hard-coded one. Verified live:
child-protection → UU 35/2014, human-rights → UU 39/1999, trafficking → Permensos 8/2023,
development priorities → the RPJMN summary.

---

## 6. Bilingual handling (ID ⇄ EN)

```
1. Detect language (lingua-py; ambiguous → default ID).
2. Retrieve in Indonesian always (the corpus is ID).
   ├─ ID query: embed + retrieve directly.
   └─ EN query: translate query → ID (Gemini Flash, cached), then retrieve.
3. Answer in the user's language.
4. Article citations stay verbatim in Indonesian — a translated "Article 5 paragraph (2)"
   is not a valid legal reference.
```

The corpus is never translated (translating legal terms is lossy); only the *query* is.

---

## 7. Hallucination defence in depth

No single check is trusted; four independent checks stack and their miss-rates multiply.

| Layer | Mechanism | Catches |
|---|---|---|
| 1. Prompt | Strict system prompt + few-shot citation format | Most well-behaved generations |
| 2. Citation whitelist | Regex extract + range/existence check | A fabricated "Pasal 99" not in the document |
| 3. Entity-Grounding (EG) | Answer legal terms ⊆ context terms | An institution/term absent from context |
| 4. Relation-Preservation (RP) | Per-sentence content overlap with context | A relation unsupported by context |

Risky answers raise a HITL flag and colour the badge. Accuracy is a **calibrated target**
(≥ 90 % with citations + human review), not a guarantee — the realistic ceiling for a
zero-budget legal assistant (commercial legal tools still hallucinate 17–34 %).

---

## 8. Local-first run model (replaces v1's cloud deployment)

Patih now runs on the laptop. There is no server to provision.

```
Terminal 1:  chainlit run app/ui/chainlit_app.py --port 8000   →  http://localhost:8000
Terminal 2:  python -m tools.inbox_watcher   (optional — auto-ingest dropped PDFs)
```

**Privacy — "local" is precise, not absolute.** Data, embeddings, retrieval, and indexes
are local and never leave the machine. **One** step touches the network: the LLM call sends
the query + the retrieved article text to a cloud provider over HTTPS. The corpus is public
regulation, so this is low-risk — but a query containing personal/confidential data would
leave the machine. A fully-local LLM (Ollama) was rejected on quality (small
Indonesian-capable models are near-random on legal reasoning), so "local" means **data and
retrieval are local; inference is cloud**.

**Why not the v1 public deployment.** v1 targeted ~100 users/day on an Oracle Always-Free
VM behind a Cloudflare tunnel. Three deploy attempts failed (Oracle payment verification
declined; a HuggingFace Space flagged by anti-spam twice for a new account pushing large
binaries), so the project pivoted to local-first — which also unlocked the move from 1 to
22 documents. The FastAPI sidecar and a HuggingFace-convention Dockerfile remain in the
repo, so public hosting (a Cloudflare tunnel from the laptop is the recorded fastest path)
can be revisited later.

---

## 9. Free-tier economics (brief)

The binding free-tier limit is rarely requests-per-minute; it is the **daily** budget.
Gemini 2.5 Flash's real free limit is ~20 requests/day (not the assumed 1500), which is why
**Groq became the workhorse**; Groq's binding limit is ~100k tokens/day, which a 50-question
evaluation burst (~170k tokens) exhausts mid-run. Aggregate serving capacity (~70–90 RPM)
comfortably absorbs the steady arrival rate (~2 RPM for 100 users/8 h), but evaluation is a
spiky 100-call burst that daily caps cannot. RAGAS judge metrics never completed on free
judges and are deferred to a one-off paid run. Full detail: `docs/patih_v3.pdf`
§Free-Tier Economics.

---

## 10. Evaluation state

Acceptance thresholds: citation accuracy ≥ 90 %; Faithfulness & Answer-Relevancy ≥ 0.85;
RP ≥ 0.80; EG ≥ 0.90; refusal precision ≥ 80 %; P95 ≤ 30 s; hard-failure < 5 %.

Latest single-document run (v6): RP 0.98 ✅, EG 1.00 ✅, refusal 100 % ✅, P95 ~4–7 s ✅,
hard-failure 0/50 ✅. **Citation accuracy ~79 %** (Tier-1 factual is production-grade; the
gap is on Tier-2/3 cross-reference questions, a model limit). **RAGAS Faithfulness/Answer-
Relevancy = nan** (no free judge could complete the run; deferred to a paid judge).
**The batch evaluation has not yet been re-run on the multi-document corpus** — the main
open quality item, since some once-out-of-scope questions are now answerable from another
law.

---

## 11. Roadmap

- **Phase 2 — breadth & structure.** More regulations; Summary-Augmented Chunking against
  DRM; a Pasal–ayat–reference knowledge graph + a GraphRAG path for thematic "how do UU X,
  PP Y, Permensos Z interact" questions; a filtered-ANN vector DB (Qdrant/pgvector) when
  per-regulation filtering is needed; a query-complexity classifier; a 100-question
  cross-document eval set; move to a paid LLM tier when free RPM is exhausted.
- **Phase 3 — agentic & production HITL.** Bounded multi-step reasoning (Self-RAG/CRAG,
  ≤ 3 hops); replace the regex EG/RP with an NLI verifier; a full HITL reviewer dashboard;
  a PDP (UU 27/2022) audit trail; a possible move to a stronger model tier.

---

## 12. Trade-offs explicitly accepted

| Trade-off | Phase 1 choice | Rejected | Reason |
|---|---|---|---|
| Citation discipline | Free model + validators | Claude Sonnet | paid |
| Multi-doc reasoning | Hybrid + cross-ref | GraphRAG now | premature; cost not justified yet |
| Reranking | RRF + parent expansion | Cohere Rerank | paid; sufficient without |
| Embedder | e5-large ONNX (FP32) | paid embedding API | free tier rate-limited; CPU self-host viable |
| Agentic loop | none in Phase 1 | Self-RAG/CRAG now | compounding-hallucination caution |
| LLM location | cloud free tier | local LLM (Ollama) | small ID models near-random on legal reasoning |
| HITL | JSONL flag queue | reviewer dashboard | personal MVP |
| Deployment | local laptop | public cloud URL | deploy attempts failed; audience is local |
| INT8 quantisation | FP32 (for now) | INT8 | blocked by a Windows export bug; retry with UTF-8 console |

---

## 13. References

- **Complete reference**: [`docs/patih_v3.pdf`](docs/patih_v3.pdf) — theory + technical
  documentation + mathematics + annotated bibliography.
- **Setup & usage**: [`README.md`](README.md).
- **Adding documents / metadata**: `data/raw/inbox/README.md`.
