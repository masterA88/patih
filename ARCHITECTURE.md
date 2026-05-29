# Arsitektur Chatbot Permensos — Penjelasan Detail

> Dokumen ini menjelaskan arsitektur sistem chatbot regulasi Kemensos secara naratif untuk pembaca yang ingin memahami **bagaimana sistem bekerja end-to-end** tanpa harus membaca 10,000 kata build-spec. Spec teknis lengkap (dependency pin, schema, acceptance criteria) ada di `D:\Research\Project Data\k1\research\chatbot-permensos-tppo\drafts\build-spec-phase1-zero-budget.md`.

**Audience**: Hilmi (peer-level technical), atau siapapun yang akan kontribusi ke proyek.

**Bahasa**: Indonesia (primary), Inggris untuk istilah teknis yang sudah konvensional.

---

## 1. Apa yang Dibangun

Chatbot QA (question-answering) untuk korpus regulasi Kementerian Sosial RI. Pengguna bertanya dalam bahasa natural (Indonesia atau Inggris) tentang isi peraturan; sistem menjawab dengan kutipan Pasal yang valid dan traceable ke dokumen asli.

**Phase 1 scope**: single document — Permensos No 8 Tahun 2023 tentang TPPO & PMI Bermasalah (13 halaman, ~20-30 Pasal).

**Phase 2+ scope**: multi-document corpus — UU 21/2007 TPPO, UU 18/2017 PMI, PP 59/2021, Permensos lain (ratusan PDF eventually).

**Konstrain absolut**:
- **Zero budget** — free-tier API + FOSS only.
- **~100 active users/hari** (spread 8 jam kerja, BUKAN 100 simultaneous RPS).
- **Akurasi tinggi** (>90% target) dengan human-in-the-loop.
- **Bilingual ID-EN** — kutipan Pasal selalu Bahasa Indonesia asli.

---

## 2. Arsitektur Tingkat Tinggi (High-Level)

Sistem ini adalah **Retrieval-Augmented Generation (RAG)** dengan beberapa lapisan tambahan untuk legal domain. Konsep dasar:

1. **Index** seluruh isi PDF jadi chunks kecil dengan metadata struktural (Pasal, ayat, huruf).
2. Saat user bertanya, **retrieve** chunks yang paling relevan.
3. **Generate** jawaban dengan LLM yang dibatasi hanya menjawab berdasarkan chunks tersebut.
4. **Validate** jawaban — pastikan setiap kutipan Pasal benar ada di sumber.
5. **Display** jawaban + kartu sumber yang bisa di-klik balik ke PDF asli.

Arsitektur ini **bukan** chatbot generic seperti ChatGPT. Ada empat hal yang membuatnya beda:
- **Struktur dokumen di-preserve** — sistem tahu bahwa "Pasal 5 ayat (2) huruf b" itu node hierarkis, bukan teks lepas.
- **Citation enforcement** — LLM dipaksa selalu menyertakan referensi pasal, dan jawaban yang menyebut Pasal yang tidak ada akan di-flag.
- **Fallback chain provider** — kalau Gemini API rate-limit, otomatis pindah ke Groq → Cerebras → OpenRouter, semua free tier.
- **Bilingual aware** — query EN di-translate ke ID untuk retrieve, lalu response di-bahas dalam EN dengan kutipan Pasal tetap Bahasa Indonesia.

---

## 3. Diagram Aliran Data (End-to-End)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  OFFLINE: INGESTION PIPELINE (run sekali per dokumen baru)                  │
└─────────────────────────────────────────────────────────────────────────────┘

  PDF (Permensos_Nomor_8_Tahun_2023.pdf)
       │
       ▼
  ┌─────────────────┐
  │  PDF Parser     │  ← PyMuPDF (teks-native), Tesseract OCR (fallback)
  │  (pdf_loader)   │
  └────────┬────────┘
           ▼
  Plain text + page numbers
           │
           ▼
  ┌──────────────────────┐
  │ Structure Parser     │  ← Regex BAB / Bagian / Pasal / Ayat / Huruf
  │ (structure_parser)   │
  └──────────┬───────────┘
             ▼
  AST (Abstract Syntax Tree):
    BAB I
    ├── Pasal 1
    │     ├── ayat (1)
    │     │     ├── huruf a
    │     │     └── huruf b
    │     └── ayat (2)
    └── Pasal 2
    ...
             │
             ▼
  ┌──────────────────────┐
  │  Parent-Doc Chunker  │  ← Child=ayat/huruf, Parent=Pasal utuh
  │     (chunker)        │
  └──────────┬───────────┘
             ▼
   List[Chunk] dengan metadata {bab, bagian, pasal, ayat, huruf, parent_id}
             │
             ├──────────────────────────┐
             ▼                          ▼
  ┌─────────────────────┐    ┌─────────────────────┐
  │  e5-large ONNX INT8 │    │   BM25 Indexer      │
  │   (Dense Embedder)  │    │   (rank_bm25)       │
  └──────────┬──────────┘    └──────────┬──────────┘
             ▼                          ▼
       Vector embeddings        Sparse term index
             │                          │
             ▼                          ▼
  ┌─────────────────────┐    ┌─────────────────────┐
  │  Chroma (SQLite)    │    │  Pickled BM25       │
  │  data/chroma/       │    │  data/bm25/         │
  └─────────────────────┘    └─────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│  ONLINE: QUERY PIPELINE (per user request)                                  │
└─────────────────────────────────────────────────────────────────────────────┘

  User query: "Apa saja bentuk eksploitasi menurut Permensos ini?"
       │
       ▼
  ┌──────────────────┐
  │  Lang Detect     │  ← lingua-py, threshold 0.7, default ID
  │  (lang_detect)   │
  └────────┬─────────┘
           │ Bahasa terdeteksi: ID
           ▼
  [Optional: kalau EN, translate query → ID via Gemini Flash]
           │
           ▼
  ┌──────────────────────┐         ┌──────────────────────┐
  │  Dense Retriever     │         │  Sparse Retriever    │
  │  (e5-large encode    │         │  (BM25 in-memory)    │
  │  query + Chroma sim) │         │                      │
  └──────────┬───────────┘         └──────────┬───────────┘
             │                                │
             ▼                                ▼
        top-10 chunks                   top-10 chunks
             │                                │
             └──────────┬─────────────────────┘
                        ▼
            ┌────────────────────────┐
            │  Reciprocal Rank       │  ← RRF: score = Σ 1/(60+rank)
            │  Fusion (RRF)          │
            └───────────┬────────────┘
                        ▼
                  top-8 chunks
                        │
                        ▼
            ┌────────────────────────┐
            │  Parent Expander       │  ← Child → fetch parent Pasal utuh
            │  (parent_expander)     │
            └───────────┬────────────┘
                        ▼
                  list[Parent Pasal]
                        │
                        ▼
            ┌────────────────────────┐
            │  Cross-Ref Resolver    │  ← Regex "Pasal X ayat (Y)" → fetch
            │  (cross_ref_resolver)  │
            └───────────┬────────────┘
                        ▼
                  Expanded context
                        │
                        ▼
            ┌────────────────────────┐
            │  Always-On Injector    │  ← Selalu prepend Pasal 1 (definisi)
            │  (always_on)           │
            └───────────┬────────────┘
                        ▼
            Final context (dipakai LLM)
                        │
                        ▼
            ┌─────────────────────────────────────────────────┐
            │            LiteLLM Gateway                      │
            │                                                 │
            │  Primary: Gemini 2.5 Flash (Google AI Studio)   │
            │     │ [rate limit / error]                      │
            │     ▼                                           │
            │  Fallback 1: Groq Llama 3.3 70B                 │
            │     │ [rate limit / error]                      │
            │     ▼                                           │
            │  Fallback 2: Cerebras Qwen 3 32B                │
            │     │ [rate limit / error]                      │
            │     ▼                                           │
            │  Fallback 3: OpenRouter DeepSeek R1 (free)      │
            └────────────────────┬────────────────────────────┘
                                 ▼
                  Raw LLM response (dengan citation)
                                 │
                                 ▼
            ┌────────────────────────────────────────┐
            │      Layer 2: Validators               │
            │                                        │
            │  ├ Citation Extractor (regex)          │
            │  ├ Whitelist Validator (Pasal exist?)  │
            │  ├ Entity Grounding (HalluGraph EG)    │
            │  └ Relation Preservation (HalluGraph RP)│
            └────────────────────┬───────────────────┘
                                 ▼
                  Validated response + confidence score
                                 │
                  ┌──────────────┴──────────────┐
                  │                             │
                Valid                       Invalid / Low conf
                  │                             │
                  ▼                             ▼
           Display to user            Flag for HITL review +
           (Chainlit UI)              fallback to "tidak diatur"
```

---

## 4. Lapisan Arsitektur (Layer-by-Layer)

Sistem dipecah jadi 7 lapisan, masing-masing dengan tanggung jawab terisolasi. Pemisahan ini bukan estetika — tujuannya supaya:
- **Phase 2 (multi-doc + KG)** tidak perlu rewrite lapisan ini, cukup extend.
- **Eval** bisa test per-lapisan (retrieval saja, generation saja) untuk diagnose bottleneck.
- **Failure mode** terlokalisir (kalau LLM provider down, retrieval tetap jalan).

### Lapisan 1: Ingestion (`app/ingest/`)

**Tugas**: Convert PDF mentah → AST terstruktur → chunks dengan metadata.

**Kenapa rumit**: PDF regulasi Indonesia punya struktur hierarkis ketat (BAB → Bagian → Pasal → ayat → huruf). Generic text-splitter (LangChain `RecursiveCharacterTextSplitter`) akan menghasilkan chunks yang melanggar batas Pasal — fatal untuk citation.

**Strategi**:
1. **Extract text** dengan PyMuPDF (PDF teks-native; Permensos 8/2023 termasuk).
2. **Fallback OCR** dengan Tesseract `tesseract-ocr-ind` untuk PDF scan (tidak dibutuhkan Phase 1 tapi siap untuk Phase 2 multi-doc).
3. **Parse struktur** dengan regex eksplisit:
   ```
   r"Pasal\s+(\d+)\s*\n(.*?)(?=Pasal\s+\d+|\Z)"
   r"\((\d+)\)\s+(.*?)(?=\(\d+\)|\Z)"
   r"([a-z])\.\s+(.*?)(?=[a-z]\.|\Z)"
   ```
4. **Validate** hasil parse — manual count Pasal vs daftar isi PDF. Kalau parse miss > 5%, regex perlu diperbaiki.
5. **Chunking parent-document**:
   - **Child chunk** (unit retrieval) = 1 ayat, atau 1 huruf kalau ayat panjang (>300 token).
   - **Parent chunk** (unit context untuk LLM) = 1 Pasal utuh.

**Output**: `data/parsed/permensos8.json` (AST) + Chroma vector store + BM25 pickle.

### Lapisan 2: Retrieval (`app/retrieval/`)

**Tugas**: Diberi query, return top-k chunks yang paling relevan + parent Pasal utuh.

**Kenapa hybrid (BM25 + dense)**: Legal domain butuh keduanya.
- **BM25** unggul untuk istilah teknis eksak: "TPPO", "Rehabilitasi Sosial", "Pekerja Migran Bermasalah" — embedding kadang melewatkan exact-match istilah jarang.
- **Dense (e5-large)** unggul untuk semantik: "apa hak korban?" → match "korban berhak memperoleh..."

**Strategi**:
1. **Dense retrieval**: embed query dengan multilingual-e5-large ONNX INT8 (self-host CPU di Oracle VM), cosine similarity di Chroma, top-10.
2. **Sparse retrieval**: BM25 in-memory di chunks yang sama, top-10.
3. **Reciprocal Rank Fusion (RRF)**: gabungkan dua ranking. Formula sederhana: `score = Σ 1/(60+rank_i)`. Hasil: top-8.
4. **Parent expansion**: untuk setiap top-8 child, fetch parent Pasal-nya. Deduplikasi.
5. **Cross-reference resolver**: regex `r"Pasal\s+(\d+)\s+ayat\s+\((\d+)\)"` di parent text — kalau ada referensi ke Pasal lain, fetch juga.
6. **Always-on Pasal 1**: Pasal 1 (definisi terminologi) selalu di-prepend. Hampir setiap pertanyaan butuh definisi "Korban TPPO" atau "Rehabilitasi Sosial" yang ada di Pasal 1.

**Output**: ordered list of Pasal teks utuh, ready untuk dimasukkan ke LLM context.

### Lapisan 3: LLM Gateway (`app/llm/`)

**Tugas**: Generate jawaban dari (query + context), dengan resilience terhadap rate-limit free tier.

**Kenapa fallback chain**: Setiap free-tier provider punya cap berbeda:
- **Gemini 2.5 Flash** (Google AI Studio): 15 RPM, 1500 RPD, 1M TPM free.
- **Groq Llama 3.3 70B**: 30 RPM, 14400 RPD, 6000 TPM free.
- **Cerebras Qwen 3 32B**: ~30 RPM (newer free tier), 1M tokens/hari.
- **OpenRouter DeepSeek R1 free**: variable, biasanya 20 RPM tapi can be revoked.

Total aggregated headroom: ~70-90 RPM. Untuk 100 active users/hari dengan ~12-15 RPM peak, single provider sering cukup. Tapi burst-aware fallback wajib supaya UX tidak break saat satu provider down.

**Strategi**:
1. **LiteLLM Router** dengan config 4-tier fallback chain.
2. **Rate limit token bucket** per provider — proactive throttle sebelum hit hard limit.
3. **Retry with backoff** untuk transient errors (5xx, network).
4. **Prompt template** struktur eksplisit:
   ```
   <system>Anda asisten regulasi Kemensos. Aturan WAJIB:
     1. Jawab HANYA berdasarkan kutipan Pasal di <context>.
     2. SELALU sertakan referensi: "(Pasal X ayat (Y) huruf z)".
     3. Jika informasi tidak ada di <context>, jawab persis:
        "Informasi tersebut tidak diatur secara spesifik dalam ..."
     4. JANGAN mengarang nomor pasal.
   </system>
   <context>{retrieved_pasals}</context>
   <question>{user_query}</question>
   ```
5. **Bilingual prompt switch**: kalau `query_lang == "en"`, swap ke system prompt EN dengan instruksi "kutipan Pasal pertahankan Bahasa Indonesia asli, terjemahan EN opsional dalam tanda kurung".

### Lapisan 4: Validators (`app/validators/`)

**Tugas**: Defense in depth terhadap hallucination — jangan trust LLM begitu saja.

**Kenapa berlapis**: Stanford HAI (Magesh et al. 2024) dokumentasi tools premium komersial (Westlaw $X/bulan) hallucinate 17-34%. Free-tier LLM lebih buruk lagi — expected 25-45% citation imprecision raw.

**Layer**:
1. **Citation Extractor**: regex extract semua `[Pasal X ayat (Y) huruf z]` dari response.
2. **Whitelist Validator**: cek setiap citation — apakah Pasal X benar ada di document registry? Kalau Pasal 99 di-mention tapi dokumen cuma punya Pasal 1-30, flag invalid.
3. **Entity Grounding (HalluGraph EG)**: entity yang muncul di response (nama lembaga, istilah hukum) harus subset dari entity di context. Pakai NER sederhana + legal-term-list Phase 1; upgrade ke proper NER di Phase 2.
4. **Relation Preservation (HalluGraph RP)**: relasi antar entity di response harus didukung di context. Phase 1 implementation sederhana (regex pattern), Phase 2 pakai dependency parsing.

**Output**: scored response + flag `citations_valid: bool` + `confidence: 0-1`. Kalau gagal threshold, response di-flag untuk HITL review.

### Lapisan 5: UI (`app/ui/`)

**Tugas**: Conversational interface dengan source transparency.

**Pilihan Chainlit** (bukan Streamlit/Gradio):
- Native streaming response.
- Built-in source citation cards.
- Conversation history persistence.
- Feedback buttons (thumbs up/down) untuk HITL data collection.

**Komponen**:
- Chat panel utama.
- Sidebar dengan retrieved chunks (clickable → highlight di PDF preview).
- "Lihat sumber" button per kutipan Pasal.
- Confidence badge ("Tinggi" / "Sedang — review disarankan" / "Rendah — perlu verifikasi manual").

### Lapisan 6: Eval (`app/eval/`)

**Tugas**: Quantify accuracy. Tanpa eval, "akurasi tinggi" hanya claim subjective.

**Test set**: 50 Q-A pairs manual, distribusi:
- 25 Pasal-extraction ("Apa bentuk eksploitasi?")
- 10 Cross-reference ("Asistensi rehabilitasi diberikan dalam bentuk apa?")
- 10 Definitional ("Siapa Korban TPPO?")
- 5 Out-of-scope refusal ("Berapa denda pelaku TPPO?" — Permensos 8/2023 tidak atur sanksi pidana).

**Metrik**:
- RAGAS Faithfulness ≥ 0.85
- RAGAS Answer Relevancy ≥ 0.85
- Custom Citation Accuracy ≥ 90%
- Refusal Precision ≥ 80%
- P95 Latency ≤ 30 detik
- Rate-limit endurance: 100 query dalam 30 menit, < 5% hard failure

**Judge**: RAGAS pakai Gemini Flash sebagai LLM-judge (sama free tier). Trade-off: judge bias terhadap dirinya sendiri kalau generator-nya juga Gemini Flash. Mitigasi: cross-validate sample 10 questions dengan Groq sebagai second judge.

### Lapisan 7: Observability (`app/infra/`)

**Tugas**: Trace setiap query end-to-end untuk debugging dan production monitoring.

**Pilihan Langfuse** (self-host di Oracle VM, FOSS):
- Trace per request: query → retrieved chunks → LLM call (dengan fallback chain attempts) → response → validation → user feedback.
- Latency breakdown per stage.
- Cost tracking (walaupun zero untuk free tier, useful saat migrate ke paid).
- Eval result history.

---

## 5. Bilingual Handling (ID-EN)

Kasus khusus yang perlu eksplisit karena Hilmi minta dukung kedua bahasa:

**Pattern**:
```
1. User input → lingua-py detect bahasa (ID, EN, atau ambiguous → default ID).
2. Korpus retrieve: SELALU di Bahasa Indonesia (karena PDF asli ID).
   ├─ Kalau query ID: langsung embed query + retrieve.
   └─ Kalau query EN: translate query → ID via Gemini Flash (tetap free tier),
                      lalu embed translated query, retrieve di korpus ID.
3. LLM generate response dalam BAHASA QUERY ASLI (bukan bahasa terjemahan).
4. Kutipan Pasal SELALU Bahasa Indonesia asli, di-preserve verbatim.
   ├─ Kalau response EN: format "Article 5 paragraph (2) states: '... [teks ID asli]'.
   │                              In English, this means: '...'"
   └─ Kalau response ID: format normal "(Pasal 5 ayat (2)) ..."
5. UI: language toggle button untuk user override.
```

**Kenapa retrieve di korpus tetap ID**: kalau translate korpus EN dulu, lossy — kehilangan nuansa istilah hukum yang penting ("eksploitasi", "rehabilitasi", "asistensi" punya makna spesifik di konteks regulasi yang sulit di-translate konsisten).

---

## 6. Pertahanan Hallucination (Defense in Depth)

Sistem punya 4 lapis pertahanan terhadap hallucination — masing-masing tidak cukup sendiri, kombinasi yang membuat acceptable.

| Layer | Mekanisme | Catch apa |
|---|---|---|
| 1. Prompt Engineering | System prompt strict + few-shot citation example | Mayoritas LLM patuh kalau prompt ketat |
| 2. Citation Whitelist | Regex extract Pasal, cek registry | "Pasal 99" yang tidak ada di doc → caught |
| 3. Entity Grounding | NER + check entity subset of context | "Kementerian X" yang tidak di-mention context → caught |
| 4. Relation Preservation | Dependency-pattern check | "X melakukan Y" yang tidak konsisten dengan context → flag |

**Kombinasi raw rate** (per literature estimate):
- LLM raw output: 25-45% imprecision di citation
- + Prompt strict: turun ke 15-25%
- + Citation whitelist: turun ke 8-15%
- + EG/RP scorer: turun ke 4-8%
- + HITL review pada flagged (~15% dari volume): production effective < 5%

Itu pencapaian realistic untuk zero-budget MVP. Bukan "sempurna" — tapi cukup untuk production single-tenant dengan disclaimer eksplisit.

---

## 7. Topologi Deployment

**Single-node deployment** di Oracle Cloud Always Free VM (ARM Ampere A1, 4 OCPU, 24GB RAM, 200GB block storage).

```
┌──────────────────────────────────────────────────────────────┐
│  Oracle Cloud Always Free VM (Ubuntu 24.04 ARM)              │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Docker Compose stack:                              │    │
│  │                                                     │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  │    │
│  │  │  Chainlit    │  │   Langfuse   │  │ Postgres │  │    │
│  │  │  (chat UI)   │  │ (observ)     │  │ (lf-db)  │  │    │
│  │  │  :8000       │  │ :3000        │  │ :5432    │  │    │
│  │  └──────┬───────┘  └──────────────┘  └──────────┘  │    │
│  │         │                                          │    │
│  │         ▼                                          │    │
│  │  ┌──────────────────────────────────────────────┐  │    │
│  │  │  App process:                                │  │    │
│  │  │   - ingest/retrieval/llm/validators modules  │  │    │
│  │  │   - e5-large ONNX INT8 (in-process)          │  │    │
│  │  │   - Chroma (data/chroma/)                    │  │    │
│  │  │   - BM25 (data/bm25/)                        │  │    │
│  │  │   - LiteLLM gateway → external APIs          │  │    │
│  │  └──────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  Backup: cron tar data/chroma/ → OCI Object Storage Free    │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼ HTTPS via Cloudflare Tunnel
                  Public URL: chatbot-permensos.your-domain
                       │
                       ▼
                  External APIs (free tier):
                       ├ Gemini 2.5 Flash
                       ├ Groq Llama 3.3 70B
                       ├ Cerebras Qwen 3 32B
                       └ OpenRouter DeepSeek R1
```

**Kenapa single-node**: skala 100 active users/hari = ~1500-5000 queries/hari = ~0.02-0.06 query/detik average. Single VM 24GB RAM cukup besar. Multi-node = over-engineering Phase 1.

**Kenapa Cloudflare Tunnel**: gratis, no port-opening di Oracle (security), built-in DDoS protection, custom domain bisa.

**Backup**: data Chroma + BM25 = beberapa MB sampai GB. Daily cron tar → OCI Object Storage 20GB free.

---

## 8. Roadmap 3 Fase

### Phase 1 (saat ini): MVP Single-Doc
- Permensos 8/2023 saja.
- Single VM Oracle Free.
- Manual test set 50 questions.
- Sukses kriteria: faithfulness ≥0.85, citation accuracy ≥90%, P95 latency ≤30s.

### Phase 2: Multi-Doc + KG-Augmented
- Tambah Permensos lain, UU 21/2007, UU 18/2017, dst (target 20-50 dokumen).
- **Knowledge Graph** dari struktur Pasal-ayat-rujukan + relasi antar-regulasi ("Pasal X di Permensos Y merujuk UU Z").
- GraphRAG retrieval pattern (Microsoft GraphRAG / LightRAG inspired).
- Eval framework di-extend untuk cross-document reasoning.
- Migration trigger ke paid tier (Gemini Flash paid $5-15/bulan) kalau free tier RPM exhausted.

### Phase 3: Agentic + HITL Production
- Multi-step reasoning planner (Self-RAG / ReAct inspired) — "AI as judge" dalam batas realistis.
- Full HITL dashboard untuk reviewer (bukan cuma Hilmi sendiri).
- A/B testing dengan ahli hukum.
- Compliance audit trail (PDP, log retention).
- Migration ke paid stack (Claude Haiku 4.5 / Sonnet 4.6 routing) — $30-100/bulan.

---

## 9. Trade-off Utama yang Diterima

Penting Hilmi (dan reviewer) tahu apa yang **secara sengaja TIDAK** dibangun di Phase 1, dan kenapa.

| Trade-off | Pilihan Phase 1 | Alternatif yang ditolak | Alasan |
|---|---|---|---|
| **Citation discipline** | Gemini Flash + validators | Claude Sonnet 4.6 | Sonnet 4.6 paid; free tier tidak ada |
| **Multi-doc reasoning** | Single doc dulu | GraphRAG langsung | Over-engineering untuk Phase 1; Phase 2 plan |
| **Reranker** | RRF saja | Cohere Rerank API | Cohere paid; RRF + parent expansion sudah cukup baik literature |
| **Embedding self-host** | e5-large CPU ONNX | Gemini embedding API | API free tier limited; self-host ARM CPU feasible untuk korpus kecil |
| **HITL** | Flag JSONL queue | Full reviewer dashboard | Personal MVP; Phase 2 expand |
| **Auth** | Tidak ada (single-tenant) | OAuth + multi-tenant | Personal/eksploratif; Phase 2+ expand |
| **Latency target** | P95 ≤30 detik | Sub-second | Free tier rate-limit mengharuskan queueing; honest target |

---

## 10. Decision Points Masih Terbuka

Beberapa pertanyaan masih perlu Hilmi putuskan saat hit milestone tertentu (di-track di `decisions.md`):

1. **Oracle region pilih mana?** — capacity Ampere A1 sering out-of-stock di Singapore/Tokyo. Cek Hyderabad/Osaka saat provision.
2. **PDF source kanonikal**: ada dua copy di project root (`Permensos_Nomor_8_Tahun_2023.pdf` dan `Permensos_Nomor_8_Tahun_2023 (1).pdf`). Verify identical via hash, pilih yang tanpa `(1)`.
3. **KG technology untuk Phase 2**: Neo4j vs Postgres-as-KG. Neo4j lebih native untuk graph queries, tapi Postgres lebih familiar + free tier Supabase cukup.
4. **Self-host LLM ada di roadmap?** — kalau Kemensos eventually butuh data residency, Sahabat-AI 70B / Qwen 3 32B self-host perlu GPU budget ($50-200/bulan H100 rental). Defer ke Phase 3.

---

## 11. Referensi

- Build-spec teknis lengkap: `D:\Research\Project Data\k1\research\chatbot-permensos-tppo\drafts\build-spec-phase1-zero-budget.md`
- Brief v1 (single-doc baseline): `D:\Research\Project Data\k1\research\chatbot-permensos-tppo\brief-tech-stack.md`
- Brief v2 (multi-doc legal reasoning): `D:\Research\Project Data\k1\research\chatbot-permensos-tppo\brief-v2-legal-reasoning.md`
- Addendum zero-budget: `D:\Research\Project Data\k1\research\chatbot-permensos-tppo\brief-v2-addendum-zero-budget.md`
- Keputusan committed: `D:\Research\Project Data\k1\research\chatbot-permensos-tppo\decisions.md`

---

## 12. Catatan untuk Pembaca Baru

Kalau Anda baru join proyek ini, urutan baca yang saya rekomendasikan:

1. **Dokumen ini (ARCHITECTURE.md)** — orientasi 10 menit.
2. **HANDOFF_DRA_PROMPT.md** — konteks bagaimana proyek di-orchestrate via Claude agents.
3. **Brief v2** — kalau ingin paham trade-off teknis (legal reasoning, GraphRAG, dll).
4. **Build-spec Phase 1** — kalau Anda yang akan implement.
5. **Brief v1** — referensi single-doc baseline (lebih ringan).

Selamat berkontribusi.
