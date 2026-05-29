"""Multi-doc smoke test after Phase B reindex.

Verifies:
  1. Chroma has chunks from multiple doc_ids (not just permensos-8-2023).
  2. BM25 + Chroma indexes load.
  3. parent_lookup spans multiple docs.
  4. Retrieval for queries about non-Permensos-8/2023 docs returns the
     correct source doc.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))


def check_chroma() -> dict:
    import chromadb
    client = chromadb.PersistentClient(path="data/chroma")
    coll = client.get_collection("permensos_chunks")
    total = coll.count()
    sample = coll.get(limit=total, include=["metadatas"])
    doc_counts = Counter(m.get("doc_id", "?") for m in sample["metadatas"])
    return {"total_chunks": total, "doc_counts": dict(doc_counts)}


def check_parent_lookup() -> dict:
    path = Path("data/bm25/parent_lookup.json")
    if not path.exists():
        return {"error": "parent_lookup.json missing"}
    data = json.loads(path.read_text(encoding="utf-8"))
    doc_counts = Counter(v.get("doc_id", "?") for v in data.values())
    return {"total_parents": len(data), "doc_counts": dict(doc_counts)}


def check_registry() -> dict:
    path = Path("data/registry/documents.json")
    if not path.exists():
        return {"error": "registry missing"}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"n_docs": len(data), "doc_ids": sorted(data.keys())}


def smoke_retrievals() -> list[dict]:
    """Run a few queries with expected source docs."""
    from app.retrieval.pipeline import retrieve

    queries = [
        ("apa itu korban TPPO", "permensos-8-2023"),  # original doc
        ("siapa pekerja migran indonesia bermasalah", "permensos-8-2023"),
        ("apa itu perdagangan orang menurut UU 21 2007", "uu-13-2011"),  # may not hit if UU TPPO not loaded
        ("perlindungan anak", "uu-35-2014"),  # UU Perlindungan Anak
        ("hak asasi manusia", "uu-39-1999"),  # UU HAM
        ("bantuan langsung tunai", "permensos-3-2025"),  # plausibly recent permensos
    ]
    results = []
    for q, expected_doc in queries:
        try:
            res = retrieve(q, top_k_fused=5)
            top_docs = [p.get("doc_id") for p in res.parent_chunks[:3]]
            results.append({
                "query": q,
                "expected_doc": expected_doc,
                "top3_docs": top_docs,
                "expected_hit": expected_doc in top_docs,
            })
        except Exception as e:
            results.append({"query": q, "error": str(e)})
    return results


def main() -> None:
    print("=" * 70)
    print("MULTI-DOC SMOKE TEST")
    print("=" * 70)

    print("\n[1/4] Registry")
    reg = check_registry()
    print(f"  n_docs: {reg.get('n_docs')}")
    if reg.get("doc_ids"):
        for d in reg["doc_ids"]:
            print(f"    - {d}")

    print("\n[2/4] Chroma collection")
    chroma = check_chroma()
    print(f"  total_chunks: {chroma['total_chunks']}")
    for d, n in sorted(chroma["doc_counts"].items(), key=lambda x: -x[1]):
        print(f"    {d}: {n} children")

    print("\n[3/4] parent_lookup")
    pl = check_parent_lookup()
    print(f"  total_parents: {pl.get('total_parents')}")
    if pl.get("doc_counts"):
        for d, n in sorted(pl["doc_counts"].items(), key=lambda x: -x[1]):
            print(f"    {d}: {n} parents")

    print("\n[4/4] Retrieval smoke")
    rets = smoke_retrievals()
    for r in rets:
        if "error" in r:
            print(f"  X {r['query'][:50]}: ERROR {r['error']}")
        else:
            mark = "OK" if r["expected_hit"] else "??"
            print(
                f"  {mark} {r['query'][:50]:50} -> top3={r['top3_docs']} "
                f"(expected={r['expected_doc']})"
            )


if __name__ == "__main__":
    main()
