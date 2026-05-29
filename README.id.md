# Patih — Asisten Regulasi Kementerian Sosial RI

> 🌐 **Bahasa / Language:** **Bahasa Indonesia** · [English](./README.md)

**Patih** menjawab pertanyaan berbahasa natural seputar korpus regulasi Kementerian Sosial
(*Kemensos*) dengan **kutipan yang dapat ditelusuri hingga tingkat pasal** — setiap klaim
faktual menunjuk balik ke pasal sumbernya (mis. *"(Pasal 5 ayat (2) huruf a)"*). Patih
dibangun sebagai **Hybrid Parent-Document RAG dengan penegakan kutipan (citation-enforced)**,
**berjalan lokal di laptop Anda**, dan dirancang jujur untuk ranah hukum: jika jawabannya
tidak ada di korpus, ia **menolak menjawab** alih-alih mengarang.

Nama "Patih" merujuk pada penasihat senior di kerajaan Jawa klasik — personanya adalah
"asisten ahli yang membantu Anda memahami regulasi".

| Jawaban dengan kartu kutipan | Pertanyaan di luar cakupan → penolakan |
|---|---|
| ![Jawaban dengan badge keyakinan dan kartu kutipan Pasal](./assets/ui-answer-citations.png) | ![Penolakan pertanyaan di luar cakupan dengan tombol umpan balik](./assets/ui-refusal-feedback.png) |

---

## Fitur

- **Retrieval hibrida**: dense (multilingual-e5-large, ONNX, CPU) + sparse (BM25), digabung
  dengan Reciprocal Rank Fusion — menangkap baik *makna* maupun *nomor pasal literal*.
- **Parent-Document retrieval**: memeringkat unit kecil (ayat/huruf) tetapi menjawab dengan
  unit utuh (Pasal) — peringkat tajam, konteks lengkap.
- **Multi-dokumen** (22 dokumen, lihat di bawah) dengan perutean implisit plus resolusi
  rujukan-silang *bercakupan per-dokumen* dan pasal definisi yang selalu aktif (Pasal 1).
- **Penegakan kutipan & pertahanan halusinasi**: ekstraksi kutipan + whitelist + dua
  penilai bergaya HalluGraph (Entity-Grounding, Relation-Preservation); jawaban berisiko
  ditandai ke antrean human-in-the-loop dan diberi badge keyakinan 🟢/🟡/🔴.
- **Abstensi terkalibrasi**: menolak pertanyaan di luar korpus.
- **Dwibahasa ID/EN**: mendeteksi bahasa → menerjemahkan *kueri* (bukan korpus) → kutipan
  pasal selalu tetap dalam bahasa Indonesia aslinya.
- **Bertumbuh lewat folder watcher**: jatuhkan dokumen baru ke inbox dan ia otomatis
  diserap (ingest) serta diindeks.

> **Catatan privasi — "lokal" itu presisi, bukan absolut.** Data, embedding, retrieval, dan
> indeks semuanya lokal dan tidak pernah meninggalkan laptop. **Satu** langkah menyentuh
> jaringan: panggilan LLM mengirim *kueri* + teks pasal hasil retrieval ke penyedia cloud
> (Groq/Gemini/dll.) melalui HTTPS. Korpusnya adalah regulasi publik, jadi risikonya rendah
> — tetapi tetap perhatikan bila kueri Anda memuat data pribadi atau rahasia. Lihat
> `docs/patih_v3.pdf` §Privacy.

---

## Prasyarat

- **Python 3.11** (bukan 3.12+ — kompatibilitas wheel).
- **[Poetry](https://python-poetry.org/docs/#installation)** ≥ 1.8.
- **~3 GB ruang disk kosong** untuk model embedding (diunduh dan dibangun lokal).
- *(Opsional)* **Tesseract OCR** + paket bahasa `ind` — hanya bila ingin menyerap
  PDF hasil pindai (Windows: <https://github.com/UB-Mannheim/tesseract/wiki>).

## Instalasi

```bash
# 1. Klon dan masuk ke repo
git clone https://github.com/masterA88/patih.git
cd patih

# 2. Pasang dependensi
poetry install

# 3. Ambil model embedder (ONNX, ~2.2 GB) ke models/. Dua opsi — lihat
#    "Mengambil model embedder" di bawah; tercepat adalah unduh langsung, tanpa konversi.

# 4. Konfigurasi kunci dan pengaturan
cp .env.example .env
#    Sunting .env:
#      - Set  EMBEDDER_BACKEND=onnx
#      - Isi minimal SATU kunci LLM gratis (GROQ_API_KEY disarankan sebagai primer;
#        GEMINI/CEREBRAS/OPENROUTER opsional sebagai fallback). Lihat komentar di .env.example.

# 5. Bangun indeks (Chroma + BM25) dari 22 dokumen yang sudah ter-parse
poetry run python -m app.retrieval.indexer --rebuild
```

## Mengambil model embedder (rincian langkah 3)

Aplikasi memuat encoder `multilingual-e5-large` sebagai ONNX dari
`models/multilingual-e5-large-onnx-int8/` (nama folder bersifat historis — bobot FP32 tetap
berfungsi). Pilih **salah satu** dari dua opsi di bawah. Setelahnya folder harus berisi
`model.onnx` + `model.onnx_data` serta berkas tokenizer (`tokenizer.json`,
`tokenizer_config.json`, `sentencepiece.bpe.model`, `special_tokens_map.json`,
`config.json`).

### Opsi A — unduh langsung (disarankan, tanpa konversi)

Repo HuggingFace sudah menyediakan ONNX siap-pakai di bawah `onnx/`, jadi Anda tinggal
mengunduh dan memindahkannya. Ini lebih cepat dari konversi dan sepenuhnya menghindari bug
ekspor INT8 di Windows.

**PowerShell (Windows):**
```powershell
poetry run huggingface-cli download intfloat/multilingual-e5-large `
  --include "onnx/model.onnx" "onnx/model.onnx_data" "onnx/config.json" `
            "onnx/tokenizer.json" "onnx/tokenizer_config.json" `
            "onnx/special_tokens_map.json" "onnx/sentencepiece.bpe.model" `
  --local-dir models\_e5_dl
New-Item -ItemType Directory -Force models\multilingual-e5-large-onnx-int8 | Out-Null
Move-Item models\_e5_dl\onnx\* models\multilingual-e5-large-onnx-int8\
Remove-Item -Recurse -Force models\_e5_dl
```

**bash (macOS / Linux):**
```bash
poetry run huggingface-cli download intfloat/multilingual-e5-large \
  --include "onnx/model.onnx" "onnx/model.onnx_data" "onnx/config.json" \
            "onnx/tokenizer.json" "onnx/tokenizer_config.json" \
            "onnx/special_tokens_map.json" "onnx/sentencepiece.bpe.model" \
  --local-dir models/_e5_dl
mkdir -p models/multilingual-e5-large-onnx-int8
mv models/_e5_dl/onnx/* models/multilingual-e5-large-onnx-int8/
rm -rf models/_e5_dl
```

Folder `onnx/` yang sama di HuggingFace juga menyediakan build INT8 yang lebih kecil
(`model_qint8_avx512_vnni.onnx`, ~560 MB). Aplikasi memuat `model.onnx` secara default, dan
build INT8 itu memerlukan CPU AVX-512-VNNI, jadi unduhan FP32 di atas adalah default yang
aman.

### Opsi B — bangun sendiri

Mengonversi bobot PyTorch ke ONNX secara lokal (mencoba INT8, jatuh ke FP32 bila gagal).
Gunakan ini bila Anda lebih suka tidak menarik artefak siap-pakai, atau ingin build INT8
AVX2:
```bash
poetry run python deploy/scripts/quantize_e5.py
```

Apa pun pilihannya, pertahankan `EMBEDDER_BACKEND=onnx` di `.env` (langkah 4) agar
pengindeksan dan kueri berbagi ruang 1024-dim yang sama.

## Menjalankan

```bash
poetry run chainlit run app/ui/chainlit_app.py --port 8000
```

Buka <http://localhost:8000>. Kueri pertama lebih lambat (~7–9 dtk) karena model embedding
dimuat; kueri berikutnya sudah panas (~1,5–2,5 dtk).

Contoh pertanyaan:
- *Apa saja bentuk eksploitasi dalam TPPO?* → mengutip Permensos 8/2023.
- *Apa definisi anak?* → mengutip UU 35/2014 (Pasal 1 angka 1).
- *Apa itu hak asasi manusia?* → mengutip UU 39/1999.
- *Bagaimana cara membuat kue brownies?* → ditolak (di luar cakupan).

## Menambah dokumen

Jalankan watcher di terminal terpisah, lalu **cukup jatuhkan PDF** ke
`data/raw/inbox/`:

```bash
poetry run python -m tools.inbox_watcher
```

Watcher akan menyerap, mengindeks secara inkremental, dan memindahkan berkas ke `data/raw/`
(kegagalan dikarantina ke `data/raw/failed/` beserta alasannya). Lihat
`data/raw/inbox/README.md`.

### Bagaimana metadata ditangani

Setiap dokumen memerlukan berkas pendamping kecil `<nama>.pdf.meta.json` yang mencatat
identitas dan peruteannya — `doc_id`-nya, jenis/nomor/tahun regulasi, serta apakah
diperlakukan sebagai regulasi berbasis pasal atau dokumen `reference` bebas-bentuk (yang
dipotong per seksi/halaman alih-alih per pasal). Fakta-fakta ini tidak bisa dibaca andal
dari isi PDF, jadi pendamping ini wajib.

Anda tidak harus menulisnya: **bila pendamping tidak ada, watcher otomatis membuatnya** dari
nama berkas dan halaman-halaman awal (memakai inferensi yang sama seperti
`tools/triage_pdfs.py` / `tools/generate_meta_sidecars.py`). Hasilnya adalah tebakan
terbaik — ditandai `"_provenance": { "auto_generated": true }` dan, bila nomor/tahun tak
dapat ditentukan, `"needs_review": true`. **Tinjau** sebelum mengandalkan label kutipannya,
sebab nomor regulasi yang salah akan menyalahlabeli kutipan.

Untuk metadata presisi, sediakan pendamping Anda sendiri di sebelah PDF (lebih diutamakan
ketimbang hasil otomatis). Pendamping minimal untuk sebuah regulasi:

```json
{
  "doc_id": "permensos-8-2023",
  "title": "Permensos 8/2023",
  "nomor": "8",
  "tahun": 2023,
  "jenis_regulasi": "PERMENSOS",
  "judul_lengkap": "Peraturan Menteri Sosial Nomor 8 Tahun 2023",
  "tentang": "Penanganan Korban TPPO dan PMI Bermasalah"
}
```

Untuk dokumen bebas-bentuk (SOP, statistik, rencana) gunakan `"doc_type": "reference"`
alih-alih `jenis_regulasi`. Untuk menonaktifkan pembuatan otomatis dan mewajibkan pendamping
eksplisit, jalankan watcher dengan `--no-auto-meta`.

## Pengujian

```bash
poetry run pytest          # ~307 tes (unit + integrasi + golden)
```

---

## Korpus bawaan (22 dokumen)

Dikirim sebagai JSON ter-parse di bawah `data/parsed/` (sumber kebenaran; indeks dibangun
ulang darinya). **19 regulasi berstruktur pasal** + **3 dokumen referensi** (RPJMN, dua
SOP). Dokumen jangkar: **Permensos 8/2023** (TPPO & pekerja migran bermasalah). Termasuk
pula, antara lain, UU 39/1999 (Hak Asasi Manusia), UU 35/2014 (Perlindungan Anak), UU
13/2011 (Penanganan Fakir Miskin), PP 39/2012, beberapa Permensos, Perbup, dan satu
Permenkes. Daftar lengkap beserta jumlah pasal ada di `docs/patih_v3.pdf` §Current Corpus
Inventory.

## Dokumentasi

- **`docs/patih_v3.pdf`** — referensi lengkap: teori (keluarga RAG, embedding, BM25, RRF,
  parsing peraturan perundang-undangan, pertahanan halusinasi, metrik evaluasi, rujukan
  matematis) + dokumentasi teknis (arsitektur tujuh lapis, katalog komponen, alur
  online/offline, ekonomi tier-gratis, privasi & runbook, panduan bangun-dari-nol, serta
  bibliografi beranotasi).
- **`ARCHITECTURE.md`** — ikhtisar arsitektur ringkas.

## Catatan biaya & tier-gratis

Patih dirancang berjalan dengan **biaya nol** untuk pemakaian personal: embedding/retrieval
lokal, dan generasi memakai tier gratis. Untuk evaluasi yang andal atau target ~100
pengguna/hari, tier gratis tidak cukup (lihat `docs/patih_v3.pdf` §Free-Tier Economics) —
anggarkan ~$5/bulan (Groq Dev atau Gemini berbayar).

---

## Penafian (Disclaimer)

Patih adalah **alat bantu riset regulasi, bukan pengganti nasihat hukum**. Akurasinya adalah
target terkalibrasi dengan tinjauan manusia, bukan jaminan. Selalu verifikasi pasal yang
dikutip terhadap regulasi aslinya sebelum mengandalkannya untuk keputusan apa pun.

## Lisensi

**Kode** Patih dirilis di bawah [Lisensi MIT](LICENSE) — © 2026 Hilmi.

Teks regulasi yang dipakai sebagai korpus adalah **dokumen publik pemerintah Indonesia** dan
tidak tercakup oleh lisensi ini; status hukumnya sendiri yang berlaku. Verifikasi setiap
sumber sebelum digunakan ulang.
