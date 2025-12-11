#!/usr/bin/env python3
# app.py
import os, json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from agents import Runner, trace
from ai_agents.enhancer_agent import enhancer_agent
from ai_agents.conversation_agent import conversation_agent
from scripts.retriever import Retriever

EMB_JSONL = os.getenv("EMB_JSONL", "data/embeddings/the_indian_constitution_embeddings.jsonl")
FAISS_INDEX = os.getenv("FAISS_INDEX", "data/embeddings/the_indian_constitution.faiss")
FAISS_IDMAP = os.getenv("FAISS_IDMAP", "data/embeddings/the_indian_constitution.idmap.jsonl")

retriever = Retriever(
    embeddings_jsonl=EMB_JSONL,
    faiss_index=FAISS_INDEX,
    faiss_idmap=FAISS_IDMAP,
    use_faiss=True
)

def format_context(retrieved_chunks, max_context_chars=4000):
    """Format retrieved chunks into context string."""
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
    return "\n---\n".join(pieces)

def answer_query(user_query, top_k=5):
    # Step 1: Enhance the query using enhancer agent
    try:
        enhanced_query_result = Runner.run_sync(enhancer_agent, user_query)
        # Try different possible return formats
        if hasattr(enhanced_query_result, 'final_output'):
            enhanced_query = enhanced_query_result.final_output
        elif hasattr(enhanced_query_result, 'messages') and enhanced_query_result.messages:
            enhanced_query = enhanced_query_result.messages[-1].content
        else:
            enhanced_query = str(enhanced_query_result) if enhanced_query_result else user_query
    except Exception as e:
        # Fallback to original query if enhancement fails
        print(f"Warning: Query enhancement failed: {e}")
        enhanced_query = user_query
    
    # Step 2: Retrieve relevant chunks using the enhanced query
    retrieved = retriever.retrieve(enhanced_query, top_k=top_k)
    
    # Step 3: Format context from retrieved chunks
    context = format_context(retrieved)
    
    # Step 4: Generate answer using conversation agent
    conversation_prompt = (
        f"Use only the following retrieved chunks to answer. "
        f"Cite chunk ids in your answer when you reference them.\n\n"
        f"RETRIEVED_CHUNKS:\n{context}\n\nQUESTION: {user_query}"
    )
    try:
        answer_result = Runner.run_sync(conversation_agent, conversation_prompt)
        # Try different possible return formats
        if hasattr(answer_result, 'final_output'):
            answer = answer_result.final_output
        elif hasattr(answer_result, 'messages') and answer_result.messages:
            answer = answer_result.messages[-1].content
        else:
            answer = str(answer_result) if answer_result else "I don't know."
    except Exception as e:
        print(f"Warning: Answer generation failed: {e}")
        answer = "I don't know."
    
    # Prepare simple sources list
    sources = [{"chunk_id": r["chunk_id"], "score": r["score"], "metadata": r.get("metadata")} for r in retrieved]
    return answer, json.dumps(sources, ensure_ascii=False, indent=2)

with gr.Blocks(title="Simple RAG Chat") as demo:
    gr.Markdown("## Simple RAG Chat (FAISS retriever + OpenAI LLM)")
    with gr.Row():
        with gr.Column(scale=3):
            query = gr.Textbox(label="Your question", placeholder="Ask about the constitution...", lines=3)
            top_k = gr.Slider(label="Top K (retrieval)", minimum=1, maximum=10, value=5, step=1)
            submit = gr.Button("Ask")
        with gr.Column(scale=2):
            sources_out = gr.Textbox(label="Sources (chunk ids + metadata)", lines=10)
    answer_out = gr.Markdown()
    def on_ask(q, k):
        ans, src = answer_query(q, top_k=k)
        return ans, src
    submit.click(on_ask, inputs=[query, top_k], outputs=[answer_out, sources_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
