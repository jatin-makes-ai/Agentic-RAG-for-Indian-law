#!/usr/bin/env python3
"""
build_faiss_from_jsonl.py

Builds a FAISS index from an embeddings JSONL file.

Writes:
- FAISS index file (args.faiss_out)
- idmap jsonl mapping faiss_internal_id -> chunk_id (args.idmap_out)

Usage:
  python build_faiss_from_jsonl.py \
    --emb_jsonl data/embeddings/constitution_embeddings.jsonl \
    --faiss_out data/embeddings/constitution.faiss \
    --idmap_out data/embeddings/constitution.idmap.jsonl \
    --index_type flat   # or ip for IndexFlatIP
"""
import argparse, json, sys
from pathlib import Path

def load_embeddings_jsonl(path):
    rows = []
    dim = None
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip(): continue
            obj = json.loads(ln)
            emb = obj.get("embedding")
            if emb is None:
                raise ValueError(f"No 'embedding' field found in record: {obj.get('chunk_id')}")
            if dim is None:
                dim = len(emb)
            elif len(emb) != dim:
                raise ValueError(f"Embedding dimension mismatch for chunk {obj.get('chunk_id')}: {len(emb)} vs {dim}")
            rows.append((obj.get("chunk_id"), emb, obj))
    return rows, dim

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_jsonl", required=True)
    parser.add_argument("--faiss_out", required=True)
    parser.add_argument("--idmap_out", required=True)
    parser.add_argument("--index_type", choices=["flat","ip"], default="flat",
                        help="flat = IndexFlatL2 (L2). ip = IndexFlatIP (inner product; normalize for cosine).")
    args = parser.parse_args()

    emb_jsonl = Path(args.emb_jsonl)
    if not emb_jsonl.exists():
        print("embeddings JSONL not found:", emb_jsonl, file=sys.stderr)
        sys.exit(2)

    rows, dim = load_embeddings_jsonl(str(emb_jsonl))
    print(f"Loaded {len(rows)} embeddings, dim={dim}")

    try:
        import numpy as np
        import faiss
    except Exception as e:
        print("Please install faiss-cpu (or faiss-gpu) and numpy. pip install faiss-cpu numpy", file=sys.stderr)
        raise

    # prepare matrix (float32)
    mat = np.array([r[1] for r in rows], dtype="float32")
    n, d = mat.shape
    assert d == dim

    # choose index
    if args.index_type == "flat":
        index = faiss.IndexFlatL2(dim)
    else:
        # IndexFlatIP for inner product (good for cosine if you normalize vectors first)
        index = faiss.IndexFlatIP(dim)

    # add
    index.add(mat)
    print("Index trained/added. ntotal=", index.ntotal)

    # write index to disk
    faiss.write_index(index, str(args.faiss_out))
    print("Wrote FAISS index to", args.faiss_out)

    # write idmap (mapping internal id -> chunk_id)
    with open(args.idmap_out, "w", encoding="utf-8") as f:
        for i, (chunk_id, emb, orig) in enumerate(rows):
            f.write(json.dumps({"faiss_internal_id": i, "chunk_id": chunk_id}, ensure_ascii=False) + "\n")
    print("Wrote idmap to", args.idmap_out)

if __name__ == "__main__":
    main()
