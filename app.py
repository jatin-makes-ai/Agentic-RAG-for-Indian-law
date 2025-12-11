#!/usr/bin/env python3
# app.py
import os, json
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
load_dotenv()

import gradio as gr
from agents import Runner, trace
from ai_agents.enhancer_agent import enhancer_agent
from ai_agents.conversation_agent import conversation_agent
from ai_agents.retrieve_checker_agent import retrieve_checker_agent, RetrieveCheckerResponse
from scripts.retriever import Retriever

EMB_JSONL = os.getenv("EMB_JSONL", "data/embeddings/the_indian_constitution_embeddings.jsonl")
FAISS_INDEX = os.getenv("FAISS_INDEX", "data/embeddings/the_indian_constitution.faiss")
FAISS_IDMAP = os.getenv("FAISS_IDMAP", "data/embeddings/the_indian_constitution.idmap.jsonl")
MAX_RETRIEVAL_TRIES = int(os.getenv("MAX_RETRIEVAL_TRIES", "3"))

retriever = Retriever(
    embeddings_jsonl=EMB_JSONL,
    faiss_index=FAISS_INDEX,
    faiss_idmap=FAISS_IDMAP,
    use_faiss=True
)

trace_name = f"Legal-Agentic-RAG_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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

def extract_agent_response(result):
    """Extract text content from agent result."""
    if hasattr(result, 'final_output'):
        return result.final_output
    elif hasattr(result, 'messages') and result.messages:
        return result.messages[-1].content
    else:
        return str(result) if result else ""

def extract_structured_output(result, expected_type=None):
    """Extract structured output (Pydantic model) from agent result."""
    if hasattr(result, 'final_output'):
        output = result.final_output
        # If it's already the expected type, return it
        if expected_type and isinstance(output, expected_type):
            return output
        # If it's a dict and we have an expected type, try to instantiate it
        if expected_type and isinstance(output, dict):
            return expected_type(**output)
        return output
    elif hasattr(result, 'messages') and result.messages:
        # Try to extract from last message if it contains structured data
        last_msg = result.messages[-1]
        if hasattr(last_msg, 'content') and isinstance(last_msg.content, expected_type):
            return last_msg.content
    return None

def check_retrieved_chunks(user_query, retrieved_chunks):
    """Check retrieved chunks using the checker agent."""
    context = format_context(retrieved_chunks, max_context_chars=2000)  # Smaller context for checking
    checker_prompt = (
        f"USER_QUERY: {user_query}\n\n"
        f"RETRIEVED_CHUNKS:\n{context}\n\n"
        f"Evaluate these chunks and output JSON with sufficiency, correctness, and feedback fields."
    )
    
    with trace(trace_name):
        checker_result = Runner.run_sync(retrieve_checker_agent, checker_prompt)
        # Extract Pydantic model directly
        checker_response = extract_structured_output(checker_result, RetrieveCheckerResponse)
        
        # Fallback if extraction fails
        if checker_response is None:
            print("Warning: Failed to extract structured output from checker, using defaults")
            return RetrieveCheckerResponse(
                sufficiency=False,
                correctness=False,
                feedback="Failed to extract checker response"
            )
        
        return checker_response

def answer_query(user_query, top_k=5):
    """Answer query with retry loop: enhancer → retriever → checker → (retry if needed)."""
    original_query = user_query
    enhanced_query = user_query
    retrieved = []
    feedback = ""
    checker_response = None
    
    # Retry loop: enhance → retrieve → check
    for attempt in range(MAX_RETRIEVAL_TRIES):
        # Step 1: Enhance the query (with feedback if retrying)
        try:
            with trace(trace_name):
                if attempt == 0:
                    # First attempt: enhance original query
                    enhancer_input = original_query
                else:
                    # Retry: enhance with feedback
                    enhancer_input = f"Original query: {original_query}\n\nFeedback from previous retrieval: {feedback}\n\nPlease enhance the query based on this feedback."
                
                enhanced_query_result = Runner.run_sync(enhancer_agent, enhancer_input)
                enhanced_query = extract_agent_response(enhanced_query_result)
        except Exception as e:
            print(f"Warning: Query enhancement failed: {e}")
            if attempt == 0:
                enhanced_query = original_query
            # Continue with previous enhanced_query if retry fails
        
        # Step 2: Retrieve relevant chunks
        retrieved = retriever.retrieve(enhanced_query, top_k=top_k)
        
        # Step 3: Check retrieved chunks
        checker_response = check_retrieved_chunks(original_query, retrieved)
        
        # Step 4: Check if we should retry
        if checker_response.sufficiency and checker_response.correctness:
            # Both flags are true, proceed to answer generation
            break
        else:
            # Flags are false, prepare feedback for retry
            feedback = checker_response.feedback or "Retrieved chunks are insufficient or incorrect."
            if attempt < MAX_RETRIEVAL_TRIES - 1:
                print(f"Retrieval attempt {attempt + 1} failed. Retrying with feedback: {feedback}")
            else:
                print(f"Max retrieval tries ({MAX_RETRIEVAL_TRIES}) reached. Proceeding with current chunks.")
    
    # Step 5: Generate answer using conversation agent
    context = format_context(retrieved)
    conversation_prompt = (
        f"Use only the following retrieved chunks to answer. "
        f"Cite chunk ids in your answer when you reference them.\n\n"
        f"RETRIEVED_CHUNKS:\n{context}\n\nQUESTION: {original_query}"
    )
    try:
        with trace(trace_name):
            answer_result = Runner.run_sync(conversation_agent, conversation_prompt)
            answer = extract_agent_response(answer_result)
            if not answer:
                answer = "I don't know."
    except Exception as e:
        print(f"Warning: Answer generation failed: {e}")
        answer = "I don't know."
    
    # Prepare sources list with checker flags
    sources = [{"chunk_id": r["chunk_id"], "score": r["score"], "metadata": r.get("metadata")} for r in retrieved]
    
    # Ensure checker_response is available (should always be set, but handle edge case)
    if checker_response is None:
        checker_response = RetrieveCheckerResponse(
            sufficiency=False,
            correctness=False,
            feedback="Checker response not available"
        )
    
    sources_info = {
        "sources": sources,
        "checker_flags": checker_response.model_dump(),  # Convert Pydantic model to dict
        "enhanced_query": enhanced_query,
        "retrieval_attempts": min(attempt + 1, MAX_RETRIEVAL_TRIES)
    }
    return answer, json.dumps(sources_info, ensure_ascii=False, indent=2)

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
