# Agents/conversation_agent.py
"""
Prompt builder for conversation LLM.
"""
from typing import List, Dict

SYSTEM_TEMPLATE = (
    "You are a legal assistant. Answer concisely and cite the chunk_ids used. "
    "If the answer cannot be found in the provided context, say \"I don't know\"."
)

def build_messages(question: str, retrieved_chunks: List[Dict], max_context_chars: int = 4000):
    # prepare context: join top chunks, truncated to max_context_chars
    pieces = []
    total = 0
    for c in retrieved_chunks:
        txt = c.get("text", "") or ""
        meta = c.get("metadata", {}) or {}
        header = f"CHUNK_ID: {c.get('chunk_id')} METADATA: {meta}"
        part = header + "\n" + txt + "\n\n"
        if total + len(part) > max_context_chars:
            # truncate remaining
            remain = max(0, max_context_chars - total)
            part = part[:remain]
            pieces.append(part)
            break
        pieces.append(part)
        total += len(part)
    context = "\n---\n".join(pieces)

    system_msg = {"role": "system", "content": SYSTEM_TEMPLATE}
    user_content = (
        "Use only the following retrieved chunks to answer. "
        "Cite chunk ids in your answer when you reference them.\n\n"
        f"RETRIEVED_CHUNKS:\n{context}\n\nQUESTION: {question}"
    )
    user_msg = {"role": "user", "content": user_content}
    return [system_msg, user_msg]
