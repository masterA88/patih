"""
One-time quantization of multilingual-e5-large to ONNX INT8.
Output: models/multilingual-e5-large-onnx-int8/  (~600MB vs 2.2GB FP32)

See build-spec Section 2.3.

Usage (from project root):
    .venv\\Scripts\\python.exe deploy\\scripts\\quantize_e5.py

ARM Ampere note: uses arm64 config, NOT avx512_vnni (x86-only).
Windows dev note: avx2 is used (avx512_vnni requires AVX-512 which is not
                  universal on modern Windows laptops — avx2 is safe everywhere).
                  For Oracle ARM, switch to arm64 config (uncomment below).

Fallback path: if INT8 quantization fails (e.g. ONNX opset issues on Windows),
the script falls back to full-precision ONNX export to the same output dir.
Correctness is identical; inference will be ~2x slower and RAM ~2x higher.
"""

import sys
import time
from pathlib import Path

MODEL_ID = "intfloat/multilingual-e5-large"
OUTPUT_DIR = Path("models/multilingual-e5-large-onnx-int8")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[quantize_e5] Downloading and exporting {MODEL_ID} to ONNX...")
print("  This downloads ~2.3 GB on first run. Subsequent runs use HF cache.")
t0 = time.time()

try:
    from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    ort_model = ORTModelForFeatureExtraction.from_pretrained(MODEL_ID, export=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    quantizer = ORTQuantizer.from_pretrained(ort_model)

    # avx2: safe for all modern x86-64 CPUs (avx512_vnni requires specific ISA)
    # For Oracle Cloud ARM Ampere A1 — swap to:
    #   qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)

    print(f"[quantize_e5] Quantizing to INT8 (avx2, dynamic) → {OUTPUT_DIR} ...")
    quantizer.quantize(save_dir=str(OUTPUT_DIR), quantization_config=qconfig)
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    elapsed = time.time() - t0
    # Report output size
    total_bytes = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file())
    print(f"[quantize_e5] INT8 quantization done in {elapsed:.0f}s")
    print(f"[quantize_e5] Output dir: {OUTPUT_DIR.resolve()}")
    print(f"[quantize_e5] Total size: {total_bytes / 1e6:.0f} MB")
    print("[quantize_e5] RAM at serving: ~1.2 GB steady-state (per build-spec Section 2.3)")

except Exception as e:
    print(f"[quantize_e5] INT8 quantization failed: {e}", file=sys.stderr)
    print("[quantize_e5] Falling back to full-precision ONNX export...", file=sys.stderr)
    print("[quantize_e5] NOTE: FP32 fallback — ~2.2 GB on disk, ~2 GB RAM, ~2x slower inference",
          file=sys.stderr)

    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    ort_model = ORTModelForFeatureExtraction.from_pretrained(MODEL_ID, export=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    ort_model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    elapsed = time.time() - t0
    total_bytes = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file())
    print(f"[quantize_e5] FP32 export done in {elapsed:.0f}s", file=sys.stderr)
    print(f"[quantize_e5] Output dir: {OUTPUT_DIR.resolve()}", file=sys.stderr)
    print(f"[quantize_e5] Total size: {total_bytes / 1e6:.0f} MB", file=sys.stderr)
