# Inbox — drop PDFs here to auto-ingest

The `inbox_watcher` daemon polls this folder every 10s. **Just drop a PDF** — if its
`.meta.json` sidecar is missing, the watcher auto-generates one from the filename + first
pages (best-effort, flagged `auto_generated`/`needs_review`; review before trusting
citations). Supply your own sidecar for precise metadata, or run with `--no-auto-meta` to
require one. On each detected document it:

1. Runs the ingest pipeline (parse + chunk + register).
2. Embeds the new children with the configured backend (`EMBEDDER_BACKEND` env,
   default `onnx`) and upserts to Chroma.
3. Rebuilds BM25 from the full corpus union.
4. Merges parent_lookup.
5. Moves the PDF + sidecar to `data/raw/` (so the registry's `pdf_path` stays valid).

## How to add a document

The quick way: drop `data/raw/inbox/<name>.pdf` and let the watcher generate the sidecar.

To control the metadata yourself, also create `data/raw/inbox/<name>.pdf.meta.json` with:

```json
{
  "doc_id": "uu-21-2007",
  "title": "UU 21/2007",
  "nomor": "21",
  "tahun": 2007,
  "jenis_regulasi": "UU",
  "judul_lengkap": "Undang-Undang Nomor 21 Tahun 2007 tentang Pemberantasan Tindak Pidana Perdagangan Orang",
  "tentang": "Pemberantasan Tindak Pidana Perdagangan Orang",
  "tanggal_berlaku": "2007-04-19",
  "source_url": null,
  "summary_prefix": "UU 21/2007 TPPO"
}
```

`jenis_regulasi` must be one of: `PERMENSOS`, `UU`, `PP`, `PERPRES`, `PERBUP`,
`PERMENKES`, `PERMEN_KEMENAKER`, `PERBANK`, `PERMA`, `OTHER`.

**Scanned PDFs:** if the PDF has no text layer, the loader auto-falls back to
Tesseract OCR (Indonesian). Tesseract must be installed; the `ind` language pack
ships in `models/tessdata/`. The parser handles both "Pasal 5" and older
"Pasal 5." (trailing period) formats, and recovers Pasal that appear before the
first BAB (common in short administrative regulations).

3. Wait ~10–60s for the watcher to detect and process. Status logs in
   `processed.log`; failures move to `data/raw/failed/` with reason in
   `failures.log`.

## Running the watcher

```powershell
.\.venv\Scripts\python.exe -m tools.inbox_watcher
# or one-shot pass:
.\.venv\Scripts\python.exe -m tools.inbox_watcher --once
```

For Chainlit + watcher together, open two terminals.
