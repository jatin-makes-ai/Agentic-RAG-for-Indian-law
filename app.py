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
from ai_agents.evaluator_agent import evaluator_agent, EvaluatorResponse
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
        
        # Format header with new CSV-based metadata structure
        article_num = c.get('chunk_id') or meta.get('Article_Number', 'Unknown')
        part_num = meta.get('Part_Number', '')
        part_title = meta.get('Part_Title', '')
        
        # Build readable header
        header_parts = [f"Article {article_num}"]
        if part_num:
            header_parts.append(f"Part {part_num}")
        if part_title:
            header_parts.append(f"({part_title})")
        
        header = " | ".join(header_parts)
        
        part = f"{header}\n{txt}\n\n"
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

def evaluate_response(user_query, answer, retrieved_chunks):
    """Evaluate the conversation agent's response using the evaluator agent."""
    context = format_context(retrieved_chunks, max_context_chars=2000)  # Smaller context for evaluation
    evaluator_prompt = (
        f"USER_QUERY: {user_query}\n\n"
        f"RETRIEVED_CHUNKS:\n{context}\n\n"
        f"AGENT_RESPONSE:\n{answer}\n\n"
        f"Evaluate the agent's response for acceptability, completeness, accuracy, and proper citation."
    )
    
    evaluator_result = Runner.run_sync(evaluator_agent, evaluator_prompt)
    # Extract Pydantic model directly
    evaluator_response = extract_structured_output(evaluator_result, EvaluatorResponse)
    
    # Fallback if extraction fails
    if evaluator_response is None:
        print("Warning: Failed to extract structured output from evaluator, using defaults")
        return EvaluatorResponse(
            is_acceptable=False,
            completeness=False,
            accuracy=False,
            proper_citation=False,
            feedback="Failed to extract evaluator response"
        )
    
    return evaluator_response

def answer_query(user_query, top_k=5):
    """
    Answer query with complete workflow in a single trace:
    1. Retrieval loop: enhancer → retriever → checker → (retry if needed)
    2. Answer generation: conversation agent
    3. Evaluation loop: evaluator → (retry conversation if needed)
    """
    # Wrap entire workflow in a single trace
    with trace(trace_name):
        original_query = user_query
        enhanced_query = user_query
        retrieved = []
        retrieval_feedback = ""
        checker_response = None
        evaluator_response = None
        
        # ===== PHASE 1: RETRIEVAL LOOP =====
        # Retry loop: enhance → retrieve → check
        for retrieval_attempt in range(MAX_RETRIEVAL_TRIES):
            # Step 1: Enhance the query (with feedback if retrying)
            try:
                if retrieval_attempt == 0:
                    # First attempt: enhance original query
                    enhancer_input = original_query
                else:
                    # Retry: enhance with feedback
                    enhancer_input = (
                        f"Original query: {original_query}\n\n"
                        f"Feedback from previous retrieval: {retrieval_feedback}\n\n"
                        f"Please enhance the query based on this feedback."
                    )
                
                enhanced_query_result = Runner.run_sync(enhancer_agent, enhancer_input)
                enhanced_query = extract_agent_response(enhanced_query_result)
            except Exception as e:
                print(f"Warning: Query enhancement failed: {e}")
                if retrieval_attempt == 0:
                    enhanced_query = original_query
                # Continue with previous enhanced_query if retry fails
            
            # Step 2: Retrieve relevant chunks
            retrieved = retriever.retrieve(enhanced_query, top_k=top_k)
            
            # Step 3: Check retrieved chunks
            checker_response = check_retrieved_chunks(original_query, retrieved)
            
            # Step 4: Check if we should retry retrieval
            if checker_response.sufficiency and checker_response.correctness:
                # Both flags are true, proceed to answer generation
                break
            else:
                # Flags are false, prepare feedback for retry
                retrieval_feedback = checker_response.feedback or "Retrieved chunks are insufficient or incorrect."
                if retrieval_attempt < MAX_RETRIEVAL_TRIES - 1:
                    print(f"Retrieval attempt {retrieval_attempt + 1} failed. Retrying with feedback: {retrieval_feedback}")
                else:
                    print(f"Max retrieval tries ({MAX_RETRIEVAL_TRIES}) reached. Proceeding with current chunks.")
        
        # Ensure checker_response is available
        if checker_response is None:
            checker_response = RetrieveCheckerResponse(
                sufficiency=False,
                correctness=False,
                feedback="Checker response not available"
            )
        
        # ===== PHASE 2: ANSWER GENERATION WITH EVALUATION LOOP =====
        context = format_context(retrieved)
        answer = None
        answer_feedback = ""
        
        for answer_attempt in range(MAX_RETRIEVAL_TRIES):
            # Step 5: Generate answer using conversation agent
            try:
                if answer_attempt == 0:
                    # First attempt: generate answer
                    conversation_prompt = (
                        f"Use only the following retrieved articles from the Indian Constitution to answer. "
                        f"Cite Article numbers (e.g., Article 15, Article 21) in your answer when you reference them.\n\n"
                        f"RETRIEVED_ARTICLES:\n{context}\n\nQUESTION: {original_query}"
                    )
                else:
                    # Retry: generate answer with feedback
                    conversation_prompt = (
                        f"Use only the following retrieved articles from the Indian Constitution to answer. "
                        f"Cite Article numbers (e.g., Article 15, Article 21) in your answer when you reference them.\n\n"
                        f"RETRIEVED_ARTICLES:\n{context}\n\n"
                        f"QUESTION: {original_query}\n\n"
                        f"FEEDBACK ON PREVIOUS ATTEMPT: {answer_feedback}\n\n"
                        f"Please improve your answer based on this feedback."
                    )
                
                answer_result = Runner.run_sync(conversation_agent, conversation_prompt)
                answer = extract_agent_response(answer_result)
                if not answer:
                    answer = "I don't know."
            except Exception as e:
                print(f"Warning: Answer generation failed: {e}")
                answer = "I don't know."
            
            # Step 6: Evaluate the answer
            evaluator_response = evaluate_response(original_query, answer, retrieved)
            
            # Step 7: Check if answer is acceptable
            if (evaluator_response.is_acceptable and 
                evaluator_response.completeness and 
                evaluator_response.accuracy and 
                evaluator_response.proper_citation):
                # All flags are true, answer is good
                break
            else:
                # Flags are false, prepare feedback for retry
                answer_feedback = evaluator_response.feedback or "Response needs improvement."
                if answer_attempt < MAX_RETRIEVAL_TRIES - 1:
                    print(f"Answer attempt {answer_attempt + 1} failed. Retrying with feedback: {answer_feedback}")
                else:
                    print(f"Max answer tries ({MAX_RETRIEVAL_TRIES}) reached. Using current answer.")
        
        # Ensure evaluator_response is available
        if evaluator_response is None:
            evaluator_response = EvaluatorResponse(
                is_acceptable=False,
                completeness=False,
                accuracy=False,
                proper_citation=False,
                feedback="Evaluator response not available"
            )
        
        # Prepare sources list with all metadata
        sources = [{"chunk_id": r["chunk_id"], "score": r["score"], "metadata": r.get("metadata")} for r in retrieved]
        sources_info = {
            "sources": sources,
            "checker_flags": checker_response.model_dump(),
            "evaluator_flags": evaluator_response.model_dump(),
            "enhanced_query": enhanced_query,
            "retrieval_attempts": min(retrieval_attempt + 1, MAX_RETRIEVAL_TRIES),
            "answer_attempts": min(answer_attempt + 1, MAX_RETRIEVAL_TRIES)
        }
        return answer, json.dumps(sources_info, ensure_ascii=False, indent=2)

with gr.Blocks(title="The Indian AI Lawyer") as demo:
    gr.Markdown("## The Indian AI Lawyer")
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
