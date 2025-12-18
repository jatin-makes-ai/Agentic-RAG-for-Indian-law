#!/usr/bin/env python3
"""
chunks_to_embeddings.py

Convert chunk jsonl -> embeddings (OpenAI by default) and optionally write a FAISS index.

Features:
- Loads OPENAI_API_KEY and EMBEDDING_MODEL from .env (dotenv) or environment.
- Defaults to OpenAI embedding model: text-embedding-3-large
- Streams input chunk jsonl to avoid memory pressure
- Deduplicates by chunk_hash (if present) to save embedding cost
- Batches embedding requests with retry/backoff
- Writes:
    - embeddings JSONL with embedding vector (floats)
    - metadata JSONL mapping chunk_id -> metadata
    - optional FAISS index file (if faiss is installed)
- CLI flags for tuning

Example:
python chunks_to_embeddings.py \
  --chunks_file data/chunks/constitution_article_chunks.jsonl \
  --output_jsonl data/embeddings/constitution_embeddings.jsonl \
  --meta_out data/embeddings/constitution_metadata.jsonl \
  --faiss_index data/embeddings/constitution.faiss \
  --batch_size 32

Requirements:
- pip install openai tiktoken python-dotenv tqdm
- pip install faiss-cpu (optional, for FAISS support)
"""

import os
import sys
import json
import time
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Iterable, Tuple

# Robust imports
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv is optional — env vars can be set directly
    pass

try:
    import openai
    from openai import OpenAI as OpenAIClient
except Exception as e:
    print("ERROR: openai package not installed. pip install openai", file=sys.stderr)
    raise

try:
    import tiktoken
except Exception:
    tiktoken = None

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False

from tqdm import tqdm
import math
import random
import pandas as pd

# -------------------------
# Configuration / defaults
# -------------------------
DEFAULT_OPENAI_MODEL = os.environ.get("EMBEDDING_MODEL", os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"))
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_APIKEY") or os.environ.get("OPENAI_KEY")

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# -------------------------
# Helpers
# -------------------------
def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)

def write_jsonl(path: str, rows: Iterable[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def md5_text(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

# Exponential backoff
def backoff_sleep(attempt: int, base: float = 1.0, jitter: float = 0.1):
    # deterministic-ish backoff with jitter
    sleep = base * (2 ** attempt)
    sleep = sleep + random.random() * jitter
    time.sleep(min(sleep, 60.0))

# -------------------------
# Embedding wrapper (OpenAI)
# -------------------------

class OpenAIEmbeddingClient:
    def __init__(self, model_name: str = DEFAULT_OPENAI_MODEL, api_key: str = None, timeout: int = 60):
        self.model_name = model_name
        # prefer explicit api_key then env var
        if api_key:
            self.client = OpenAIClient(api_key=api_key, timeout=timeout)
        else:
            # OpenAIClient will read from OPENAI_API_KEY env var if present
            self.client = OpenAIClient(timeout=timeout)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts using the new openai>=1.0.0 interface.
        Retries on transient errors.
        """
        max_retries = 5
        attempt = 0
        while True:
            try:
                resp = self.client.embeddings.create(model=self.model_name, input=texts)
                # resp.data is a list of objects; each has .embedding (a list of floats)
                embeddings = [item.embedding for item in resp.data]
                return embeddings
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    raise
                print(f"Embedding request failed (attempt {attempt}/{max_retries}): {e}", file=sys.stderr)
                backoff_sleep(attempt)
# -------------------------
# Core processing
# -------------------------
def stream_chunks_jsonl(chunks_file: str):
    """
    Yield (chunk_id, chunk_text, metadata) for each item in a chunks JSONL file.
    Expects each json line to be a dict with at least chunk_id and text fields.
    """
    for rec in read_jsonl(chunks_file):
        chunk_id = rec.get("chunk_id") or rec.get("id") or rec.get("doc_id", "") + "_" + md5_text(rec.get("text", "")[:64])
        text = rec.get("text", "")
        # metadata: keep everything except text and embedding if present
        metadata = {k: v for k, v in rec.items() if k not in ("text", "embedding")}
        yield chunk_id, text, metadata


def stream_chunks_csv(csv_file: str):
    """
    Yield (chunk_id, chunk_text, metadata) for each row in a structured CSV.

    Assumes the CSV has the following columns:
      - Part_Number
      - Part_Title
      - Article_Number
      - Article_Text

    Each row (Article) becomes a single chunk.
    - chunk_id: Article_Number (as string)
    - text: Article_Text
    - metadata: all other columns (including Part_Number, Part_Title, Article_Number)
    """
    df = pd.read_csv(csv_file)

    required_cols = {"Article_Number", "Article_Text"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    for _, row in df.iterrows():
        chunk_id = str(row["Article_Number"])
        text = str(row["Article_Text"])
        # keep all other columns as metadata
        metadata = {col: row[col] for col in df.columns if col not in ("Article_Text",)}
        yield chunk_id, text, metadata


def stream_chunks(chunks_file: str):
    """
    Dispatch to the appropriate chunk stream based on file extension.

    - If chunks_file ends with .csv  -> interpret as structured article CSV
    - Otherwise                      -> interpret as JSONL chunks (original behavior)
    """
    suffix = Path(chunks_file).suffix.lower()
    if suffix == ".csv":
        yield from stream_chunks_csv(chunks_file)
    else:
        yield from stream_chunks_jsonl(chunks_file)

def dedupe_by_hash(items: Iterable[Tuple[str,str,dict]]) -> Tuple[List[Tuple[str,str,dict]], Dict[str,str]]:
    """
    Deduplicate by chunk_hash in metadata (if present); otherwise by text md5.
    Returns list of unique items and mapping from deduped_chunk_id -> representative_chunk_id.
    """
    seen = {}
    out = []
    mapping = {}
    for chunk_id, text, meta in items:
        chash = None
        if meta and "chunk_hash" in meta:
            chash = meta.get("chunk_hash")
        else:
            chash = md5_text(text)
        if chash in seen:
            # map this chunk_id to existing
            mapping[chunk_id] = seen[chash]
            continue
        seen[chash] = chunk_id
        mapping[chunk_id] = chunk_id
        out.append((chunk_id, text, meta))
    return out, mapping

def batchify(iterable: List, batch_size: int):
    batch = []
    for it in iterable:
        batch.append(it)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

# -------------------------
# FAISS helpers
# -------------------------
def build_faiss_index(vectors, dim, index_path=None, use_gpu=False, index_factory=None):
    """
    vectors: numpy array float32 shape (n,dim)
    index_path: if provided, write index to this path (faiss.write_index)
    index_factory: optional faiss index description (e.g., "Flat", "IVF100,Flat", "HNSW32")
    Returns: (index, index_path_written_or_None)
    """
    import numpy as np
    if index_factory:
        index = faiss.index_factory(dim, index_factory)
    else:
        # default: exact flat L2 index (or we can use inner product)
        index = faiss.IndexFlatL2(dim)
    # if GPU requested, try to move to GPU
    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        except Exception as e:
            print("Warning: requested FAISS GPU but couldn't move index to GPU:", e, file=sys.stderr)
    index.add(np.asarray(vectors, dtype='float32'))
    if index_path:
        try:
            # if GPU index, move back to CPU before writing
            if faiss.get_num_gpus() > 0 and index.is_trained and hasattr(faiss, "index_gpu_to_cpu"):
                try:
                    index = faiss.index_gpu_to_cpu(index)
                except Exception:
                    pass
            faiss.write_index(index, index_path)
            return index, index_path
        except Exception as e:
            print("Failed to write FAISS index:", e, file=sys.stderr)
            return index, None
    return index, None

# -------------------------
# Main CLI function
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks_file", required=True, help="Input chunks jsonl (from chunker)")
    parser.add_argument("--output_jsonl", required=True, help="Output embeddings jsonl (writes each record with embedding and metadata)")
    parser.add_argument("--meta_out", required=True, help="Output metadata jsonl mapping chunk_id -> metadata (no embedding)")
    parser.add_argument("--faiss_index", default=None, help="Optional path to write FAISS index (.faiss). If omitted, FAISS index not written.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding API calls")
    parser.add_argument("--model", default=DEFAULT_OPENAI_MODEL, help="Embedding model name (env EMBEDDING_MODEL overrides)")
    parser.add_argument("--openai_api_key", default=None, help="Optional override for OPENAI_API_KEY")
    parser.add_argument("--dedupe", action="store_true", help="Deduplicate chunks by chunk_hash/text before embedding")
    parser.add_argument("--skip_faiss_if_missing", action="store_true", help="If faiss not installed, skip instead of failing")
    parser.add_argument("--max_chunks", type=int, default=None, help="Optional: limit to this many unique chunks for testing")
    args = parser.parse_args()

    # env overrides
    if args.openai_api_key:
        openai.api_key = args.openai_api_key
    elif OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY

    # model selection
    model_name = os.environ.get("EMBEDDING_MODEL", args.model)

    print("Embedding configuration:")
    print("  model:", model_name)
    print("  batch_size:", args.batch_size)
    print("  chunks_file:", args.chunks_file)
    print("  output_jsonl:", args.output_jsonl)
    print("  meta_out:", args.meta_out)
    if args.faiss_index:
        print("  faiss_index:", args.faiss_index)
    print("  dedupe:", args.dedupe)

    client = OpenAIEmbeddingClient(model_name=model_name, api_key=openai.api_key)

    # Stream chunks
    items = list(stream_chunks(args.chunks_file))
    print(f"Loaded {len(items)} chunks from {args.chunks_file}")

    if args.dedupe:
        unique_items, mapping = dedupe_by_hash(items)
        print(f"Deduplicated -> {len(unique_items)} unique chunks (mapping size: {len(mapping)})")
    else:
        unique_items = items
        mapping = {cid: cid for cid, _, _ in items}

    if args.max_chunks:
        unique_items = unique_items[:args.max_chunks]

    # Prepare outputs
    out_dir = Path(args.output_jsonl).parent
    ensure_dir = Path(out_dir)
    ensure_dir.mkdir(parents=True, exist_ok=True)

    out_f = open(args.output_jsonl, "w", encoding="utf-8")
    meta_f = open(args.meta_out, "w", encoding="utf-8")

    # We'll collect vectors and ids for faiss if requested
    ids_for_faiss = []
    vectors_for_faiss = []

    # Process in batches
    total = len(unique_items)
    pbar = tqdm(total=total, desc="Embedding chunks")
    processed = 0
    for batch in batchify(unique_items, args.batch_size):
        chunk_ids = [b[0] for b in batch]
        texts = [b[1] for b in batch]
        metas = [b[2] for b in batch]

        # Call embedding API
        try:
            embeddings = client.embed_batch(texts)
        except Exception as e:
            print("Fatal error embedding batch:", e, file=sys.stderr)
            raise

        # Ensure shape alignment
        if len(embeddings) != len(texts):
            raise RuntimeError(f"Embedding count mismatch: {len(embeddings)} embeddings for {len(texts)} texts")

        # Write outputs
        for cid, txt, meta, emb in zip(chunk_ids, texts, metas, embeddings):
            # convert to float32 list
            emb_f32 = [float(x) for x in emb]
            # record to embeddings jsonl: keep text, metadata, and embedding
            out_rec = {
                "chunk_id": cid,
                "embedding": emb_f32,
                "text": txt,
                "metadata": meta
            }
            out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

            # metadata mapping (without embedding)
            meta_rec = {
                "chunk_id": cid,
                "metadata": meta
            }
            meta_f.write(json.dumps(meta_rec, ensure_ascii=False) + "\n")

            # prepare for faiss
            ids_for_faiss.append(cid)
            vectors_for_faiss.append(emb_f32)

        processed += len(batch)
        pbar.update(len(batch))

    pbar.close()
    out_f.close()
    meta_f.close()

    print(f"Wrote embeddings for {processed} chunks to {args.output_jsonl}")
    print(f"Wrote metadata for {processed} chunks to {args.meta_out}")

    # Build FAISS index if requested
    if args.faiss_index:
        if not _HAS_FAISS:
            if args.skip_faiss_if_missing:
                print("FAISS not installed; skipping FAISS index build.", file=sys.stderr)
            else:
                raise RuntimeError("FAISS not installed. Install faiss-cpu or faiss-gpu to build index.")
        else:
            import numpy as np
            vecs = np.array(vectors_for_faiss, dtype="float32")
            dim = vecs.shape[1]
            # default to FlatL2 index. If you want inner product (cosine) use IndexFlatIP
            print(f"Building FAISS IndexFlatL2 with dim={dim} and n={len(vecs)} vectors ...")
            index = faiss.IndexFlatL2(dim)
            index.add(vecs)
            # save textual mapping separately for ids -> chunk_id
            # FAISS only stores numeric ids; we will rely on ordering or store mapping file
            faiss.write_index(index, args.faiss_index)
            # write id mapping file
            id_map_path = str(Path(args.faiss_index).with_suffix(".idmap.jsonl"))
            with open(id_map_path, "w", encoding="utf-8") as mid:
                for i, cid in enumerate(ids_for_faiss):
                    mid.write(json.dumps({"faiss_internal_id": i, "chunk_id": cid}, ensure_ascii=False) + "\n")
            print(f"FAISS index written to {args.faiss_index} and id map to {id_map_path}")

if __name__ == "__main__":
    main()
