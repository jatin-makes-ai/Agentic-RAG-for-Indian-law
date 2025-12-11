#!/usr/bin/env python3
# app.py
import os, json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from openai import OpenAI
from agents.conversation_agent import build_messages

from scripts.retriever import Retriever

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CONV_MODEL = os.getenv("CONVERSATION_LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else OpenAI()

EMB_JSONL = os.getenv("EMB_JSONL", "data/embeddings/constitution_embeddings.jsonl")
FAISS_INDEX = os.getenv("FAISS_INDEX", "data/embeddings/constitution.faiss")
FAISS_IDMAP = os.getenv("FAISS_IDMAP", "data/embeddings/constitution.idmap.jsonl")

retriever = Retriever(
    embeddings_jsonl=EMB_JSONL,
    faiss_index=FAISS_INDEX,
    faiss_idmap=FAISS_IDMAP,
    use_faiss=True
)

def answer_query(user_query, top_k=5):
    retrieved = retriever.retrieve(user_query, top_k=top_k)
    messages = build_messages(user_query, retrieved)
    # call chat completion
    resp = client.chat.completions.create(model=CONV_MODEL, messages=messages, temperature=0.0, max_tokens=512)
    answer = resp.choices[0].message.content
    # prepare simple sources list
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
