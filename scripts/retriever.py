#!/usr/bin/env python3
"""
Minimal retriever:
- FAISS IndexFlatL2 (preferred) with idmap jsonl (faiss_internal_id -> chunk_id)
- fallback: linear scan over embeddings jsonl
- uses OpenAI v1 embeddings (from .env or env)
"""

import os, json, argparse
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError("openai>=1.0.0 required") from e

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else OpenAI()

import numpy as np
_HAS_FAISS = True
try:
    import faiss
except Exception:
    faiss = None
    _HAS_FAISS = False

def load_embeddings_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip(): continue
            r = json.loads(ln)
            rows.append(r)
    emb_matrix = np.array([np.asarray(r["embedding"], dtype="float32") for r in rows], dtype="float32")
    id_to_rec = {r["chunk_id"]: r for r in rows}
    ordered_ids = [r["chunk_id"] for r in rows]
    return emb_matrix, id_to_rec, ordered_ids

def load_faiss_index(index_path: str, idmap_path: Optional[str]=None):
    idx = faiss.read_index(index_path)
    id_map = None
    if idmap_path and Path(idmap_path).exists():
        id_map = []
        with open(idmap_path, "r", encoding="utf-8") as f:
            for ln in f:
                id_map.append(json.loads(ln)["chunk_id"])
    return idx, id_map

def embed_query(text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    resp = client.embeddings.create(model=model, input=[text])
    return resp.data[0].embedding

def l2_to_score(d: float) -> float:
    return float(1.0 / (1.0 + d))

class Retriever:
    def __init__(
        self,
        embeddings_jsonl: str,
        faiss_index: Optional[str] = None,
        faiss_idmap: Optional[str] = None,
        use_faiss: bool = True,
    ):
        self.emb_jsonl = embeddings_jsonl
        self.faiss_index_path = faiss_index
        self.faiss_idmap = faiss_idmap
        self.use_faiss = use_faiss and _HAS_FAISS and faiss_index is not None
        self.emb_matrix, self.id_to_rec, self.ordered_ids = load_embeddings_jsonl(embeddings_jsonl)
        self.dim = self.emb_matrix.shape[1]
        self.index = None
        self.id_map = None
        if self.use_faiss:
            self.index, self.id_map = load_faiss_index(faiss_index, faiss_idmap)
            # check dim compatibility
            try:
                if getattr(self.index, "d", None) and self.index.d != self.dim:
                    raise ValueError("FAISS index dim != embeddings dim")
            except Exception:
                pass

    def retrieve(self, query: str, top_k: int = 5, min_score: Optional[float] = None) -> List[Dict]:
        q_emb = np.array(embed_query(query), dtype="float32").reshape(1, -1)
        results = []
        if self.index is not None:
            D, I = self.index.search(q_emb, top_k)
            dists = D[0]
            ids = I[0]
            for dist, iid in zip(dists, ids):
                if iid < 0:
                    continue
                chunk_id = None
                if self.id_map:
                    chunk_id = self.id_map[iid] if iid < len(self.id_map) else None
                else:
                    chunk_id = self.ordered_ids[iid] if iid < len(self.ordered_ids) else None
                rec = self.id_to_rec.get(chunk_id, {})
                score = l2_to_score(dist)
                if min_score is not None and score < min_score:
                    continue
                results.append({"chunk_id": chunk_id, "score": score, "distance": float(dist), "text": rec.get("text",""), "metadata": rec.get("metadata",{})})
        else:
            # linear L2 scan
            dif = self.emb_matrix - q_emb
            dists = np.sum(dif * dif, axis=1)
            idxs = np.argsort(dists)[:top_k]
            for idx in idxs:
                chunk_id = self.ordered_ids[idx]
                rec = self.id_to_rec.get(chunk_id, {})
                dist = float(dists[idx])
                score = l2_to_score(dist)
                if min_score is not None and score < min_score:
                    continue
                results.append({"chunk_id": chunk_id, "score": score, "distance": dist, "text": rec.get("text",""), "metadata": rec.get("metadata",{})})
        return results

    def retrieve_by_threshold(self, query: str, min_score: float, max_results: int = 200) -> List[Dict]:
        """
        Retrieve all chunks above a score threshold.
        For FAISS: searches up to max_results, then filters by threshold.
        For linear scan: calculates all distances, filters by threshold, sorts by score.
        """
        q_emb = np.array(embed_query(query), dtype="float32").reshape(1, -1)
        results = []
        
        if self.index is not None:
            # FAISS: search with high top_k, then filter by threshold
            D, I = self.index.search(q_emb, max_results)
            dists = D[0]
            ids = I[0]
            for dist, iid in zip(dists, ids):
                if iid < 0:
                    continue
                chunk_id = None
                if self.id_map:
                    chunk_id = self.id_map[iid] if iid < len(self.id_map) else None
                else:
                    chunk_id = self.ordered_ids[iid] if iid < len(self.ordered_ids) else None
                rec = self.id_to_rec.get(chunk_id, {})
                score = l2_to_score(dist)
                if score >= min_score:
                    results.append({"chunk_id": chunk_id, "score": score, "distance": float(dist), "text": rec.get("text",""), "metadata": rec.get("metadata",{})})
        else:
            # Linear scan: calculate all distances, filter by threshold
            dif = self.emb_matrix - q_emb
            dists = np.sum(dif * dif, axis=1)
            for idx, dist in enumerate(dists):
                chunk_id = self.ordered_ids[idx]
                rec = self.id_to_rec.get(chunk_id, {})
                score = l2_to_score(float(dist))
                if score >= min_score:
                    results.append({"chunk_id": chunk_id, "score": score, "distance": float(dist), "text": rec.get("text",""), "metadata": rec.get("metadata",{})})
        
        # Sort by score (descending) and return
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

# CLI
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", required=True)
    p.add_argument("--faiss_index", default=None)
    p.add_argument("--faiss_idmap", default=None)
    p.add_argument("--query", required=True)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--min_score", type=float, default=None)
    args = p.parse_args()

    r = Retriever(args.embeddings, faiss_index=args.faiss_index, faiss_idmap=args.faiss_idmap, use_faiss=True)
    out = r.retrieve(args.query, top_k=args.top_k, min_score=args.min_score)
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
