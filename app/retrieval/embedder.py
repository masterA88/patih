"""
Multi-backend embedder: Gemini API → ONNX local → sentence-transformers fallback.

Priority (auto-detected):
  1. Gemini API (gemini-embedding-001 / text-embedding-004) — if GEMINI_API_KEY env var set.
     - Multilingual, free tier 1500 RPD / 100 RPM, dim 768.
     - Zero local disk needed.
     - task_type controls query/passage semantics (instead of e5 prefix).
  2. ONNX local (multilingual-e5-large) — if models/multilingual-e5-large-onnx-int8/ exists.
     - Multilingual, dim 1024. Requires ~600 MB disk for INT8 model.
  3. sentence-transformers fallback (all-MiniLM-L6-v2) — last resort, English-only.
     - For pipeline validation only — poor quality on Indonesian text.

e5 prefix convention:
  - Queries:  text_for_embed should NOT have prefix; encode_query() adds "query: " (for ONNX/ST).
  - Passages: text_for_embed from Step 2 has "passage: " prefix baked in.
  - For Gemini: e5 prefix is stripped before API call; task_type carries the semantic.

See build-spec Section 5.2 (dense retrieval, encode query line 699).
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "models/multilingual-e5-large-onnx-int8"
_ST_FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Gemini config
_GEMINI_MODEL = "models/gemini-embedding-001"  # GA, dim configurable, free tier 100 RPM
_GEMINI_DIM = 768  # balance quality vs Chroma storage (gemini-embedding-001 supports 768/1536/3072)
_GEMINI_RPM_LIMIT = 100
_GEMINI_BATCH_PAUSE_SEC = 0.65  # ~92 RPM ceiling, safe under 100

# Tokenizer limits for e5-large
MAX_LENGTH = 512
BATCH_SIZE_DEFAULT = 16


def _strip_e5_prefix(text: str) -> str:
    """Strip 'query: ' or 'passage: ' prefix added for e5 convention."""
    for prefix in ("query: ", "passage: "):
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


class Embedder:
    """
    Encoder supporting three backends:
      1. Gemini API (primary): google.generativeai.embed_content
      2. ONNX local: ORTModelForFeatureExtraction
      3. sentence-transformers (fallback, dev only)

    Thread-safety: not thread-safe.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        allow_st_fallback: bool = True,
        prefer_gemini: bool = True,
    ) -> None:
        self._backend: str = ""
        model_path = Path(model_path)
        gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()

        # EMBEDDER_BACKEND env override — pins the backend regardless of API key.
        # Values: "onnx" | "gemini" | "st" | "" (auto)
        # Use this when index and runtime must share a dim (e.g. ONNX dim=1024 vs Gemini dim=768).
        forced_backend = os.environ.get("EMBEDDER_BACKEND", "").strip().lower()
        if forced_backend == "onnx":
            prefer_gemini = False
        elif forced_backend == "gemini":
            # keep prefer_gemini True; fall through to gemini branch
            pass
        elif forced_backend == "st":
            prefer_gemini = False
            # We'll force ST below by short-circuiting the ONNX path

        if prefer_gemini and gemini_key:
            try:
                self._load_gemini(gemini_key)
                return
            except Exception as e:
                logger.warning("Gemini embedder init failed (%s) — falling back to local", e)

        if forced_backend == "st":
            if not allow_st_fallback:
                raise RuntimeError("EMBEDDER_BACKEND=st but allow_st_fallback=False")
            self._load_st_fallback()
            return

        if model_path.exists() and any(model_path.glob("*.onnx")):
            self._load_onnx(model_path)
        elif allow_st_fallback:
            logger.warning(
                "No Gemini API key and ONNX model not found at %s — using ST fallback (%s). "
                "Set GEMINI_API_KEY env var for multilingual quality. "
                "MiniLM is English-only (dim=384) — low quality for Indonesian.",
                model_path,
                _ST_FALLBACK_MODEL,
            )
            self._load_st_fallback()
        else:
            raise FileNotFoundError(
                f"No embedding backend available. Set GEMINI_API_KEY env var, "
                f"or download ONNX model to {model_path.resolve()}."
            )

    def _load_gemini(self, api_key: str) -> None:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        # Smoke test with output_dimensionality for gemini-embedding-001
        probe = genai.embed_content(
            model=_GEMINI_MODEL,
            content="probe",
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=_GEMINI_DIM,
        )
        if "embedding" not in probe or not probe["embedding"]:
            raise RuntimeError("Gemini embed probe returned empty result")
        self._genai = genai
        self._backend = "gemini"
        self._dim = len(probe["embedding"])
        self._last_call_ts = 0.0
        logger.info("Gemini embedder ready — model=%s dim=%d", _GEMINI_MODEL, self._dim)

    def _load_onnx(self, model_path: Path) -> None:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer

        logger.info("Loading ONNX model from %s", model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self._model = ORTModelForFeatureExtraction.from_pretrained(str(model_path))
        self._backend = "onnx"
        probe = self._encode_onnx_texts(["probe"])
        self._dim = probe.shape[1]
        logger.info("ONNX embedder ready — dim=%d", self._dim)

    def _load_st_fallback(self) -> None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading sentence-transformers fallback: %s", _ST_FALLBACK_MODEL)
        self._st_model = SentenceTransformer(_ST_FALLBACK_MODEL)
        self._backend = "sentence_transformers"
        self._dim = self._st_model.get_sentence_embedding_dimension()
        logger.info("ST fallback embedder ready — dim=%d", self._dim)

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def backend(self) -> str:
        return self._backend

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string. Returns L2-normalized float32 vector."""
        if self._backend == "gemini":
            return self._encode_gemini_single(query, task_type="RETRIEVAL_QUERY")
        prefixed = f"query: {query}"
        return self._encode_single(prefixed)

    def encode_passage(self, text: str) -> np.ndarray:
        """Encode a single passage. Step 2 chunker bakes 'passage: ' prefix into text_for_embed."""
        if self._backend == "gemini":
            stripped = _strip_e5_prefix(text)
            return self._encode_gemini_single(stripped, task_type="RETRIEVAL_DOCUMENT")
        return self._encode_single(text)

    def encode_batch(
        self, texts: list[str], is_query: bool = False, batch_size: int = BATCH_SIZE_DEFAULT
    ) -> np.ndarray:
        """Batch encode for indexing."""
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        if self._backend == "gemini":
            task_type = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
            return self._encode_gemini_batch(texts, task_type=task_type)

        if is_query:
            texts = [f"query: {t}" for t in texts]

        if self._backend == "sentence_transformers":
            return self._encode_st_batch(texts, batch_size)

        # ONNX path
        all_vecs: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vecs = self._encode_onnx_texts(batch)
            all_vecs.append(vecs)
        return np.vstack(all_vecs)

    # ------------------------------------------------------------------
    # Gemini
    # ------------------------------------------------------------------

    def _gemini_throttle(self) -> None:
        """Sleep if needed to stay under RPM ceiling."""
        elapsed = time.monotonic() - self._last_call_ts
        if elapsed < _GEMINI_BATCH_PAUSE_SEC:
            time.sleep(_GEMINI_BATCH_PAUSE_SEC - elapsed)
        self._last_call_ts = time.monotonic()

    def _encode_gemini_single(self, text: str, task_type: str) -> np.ndarray:
        self._gemini_throttle()
        result = self._genai.embed_content(
            model=_GEMINI_MODEL,
            content=text,
            task_type=task_type,
            output_dimensionality=_GEMINI_DIM,
        )
        vec = np.asarray(result["embedding"], dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / max(norm, 1e-9)

    def _encode_gemini_batch(self, texts: list[str], task_type: str) -> np.ndarray:
        """Batch via sequential calls (Gemini SDK has batch_embed_contents but free-tier compat varies)."""
        stripped = [_strip_e5_prefix(t) for t in texts] if task_type == "RETRIEVAL_DOCUMENT" else texts
        vecs = []
        for i, text in enumerate(stripped):
            if (i + 1) % 10 == 0:
                logger.info("Gemini embedding %d/%d", i + 1, len(stripped))
            vec = self._encode_gemini_single(text, task_type=task_type)
            vecs.append(vec)
        return np.vstack(vecs)

    # ------------------------------------------------------------------
    # Local backends
    # ------------------------------------------------------------------

    def _encode_single(self, text: str) -> np.ndarray:
        return self.encode_batch([text], is_query=False)[0]

    def _encode_onnx_texts(self, texts: list[str]) -> np.ndarray:
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="np",
        )
        outputs = self._model(**encoded)
        last_hidden = outputs.last_hidden_state

        attention_mask = encoded["attention_mask"]
        mask_expanded = attention_mask[:, :, None].astype(np.float32)
        sum_hidden = (last_hidden * mask_expanded).sum(axis=1)
        sum_mask = mask_expanded.sum(axis=1)
        mean_pooled = sum_hidden / np.maximum(sum_mask, 1e-9)

        norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
        normalized = mean_pooled / np.maximum(norms, 1e-9)
        return normalized.astype(np.float32)

    def _encode_st_batch(self, texts: list[str], batch_size: int) -> np.ndarray:
        embeddings = self._st_model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 50,
        )
        return embeddings.astype(np.float32)
