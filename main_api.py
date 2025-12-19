#!/usr/bin/env python3
"""
FastAPI application for the Indian AI Lawyer system.
Uses threshold-based retrieval instead of fixed top_k.
"""

import os
import json
import asyncio
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import Optional, List, Dict
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents import Runner, trace
from ai_agents.enhancer_agent import enhancer_agent
from ai_agents.conversation_agent import conversation_agent
from ai_agents.retrieve_checker_agent import retrieve_checker_agent, RetrieveCheckerResponse
from ai_agents.evaluator_agent import evaluator_agent, EvaluatorResponse
from scripts.retriever import Retriever

# Configuration
EMB_JSONL = os.getenv("EMB_JSONL", "data/embeddings/the_indian_constitution_embeddings.jsonl")
FAISS_INDEX = os.getenv("FAISS_INDEX", "data/embeddings/the_indian_constitution.faiss")
FAISS_IDMAP = os.getenv("FAISS_IDMAP", "data/embeddings/the_indian_constitution.idmap.jsonl")
MAX_RETRIEVAL_TRIES = int(os.getenv("MAX_RETRIEVAL_TRIES", "3"))
DEFAULT_MIN_SCORE = float(os.getenv("DEFAULT_MIN_SCORE", "0.7"))
MAX_RETRIEVAL_RESULTS = int(os.getenv("MAX_RETRIEVAL_RESULTS", "200"))

# Initialize retriever
retriever = Retriever(
    embeddings_jsonl=EMB_JSONL,
    faiss_index=FAISS_INDEX,
    faiss_idmap=FAISS_IDMAP,
    use_faiss=True
)

trace_name = f"Legal-Agentic-RAG_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Thread pool executor for running sync agent calls in async context
executor = ThreadPoolExecutor(max_workers=4)

async def run_sync_in_thread(func, *args, **kwargs):
    """Run a synchronous function in a thread pool to avoid event loop conflicts."""
    loop = asyncio.get_running_loop()
    # Use functools.partial to create a callable with arguments
    return await loop.run_in_executor(executor, partial(func, *args, **kwargs))

# FastAPI app
app = FastAPI(
    title="Indian AI Lawyer API",
    description="API for querying the Indian Constitution using AI agents",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QueryRequest(BaseModel):
    query: str = Field(..., description="User query about the Indian Constitution")
    min_score: Optional[float] = Field(
        default=DEFAULT_MIN_SCORE,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold for retrieval (0.0-1.0)"
    )
    max_results: Optional[int] = Field(
        default=MAX_RETRIEVAL_RESULTS,
        ge=1,
        le=500,
        description="Maximum number of results to retrieve above threshold"
    )

class ArticleMetadata(BaseModel):
    article_number: str
    part_number: Optional[str] = None
    part_title: Optional[str] = None
    score: float
    distance: float

class QueryResponse(BaseModel):
    answer: str = Field(..., description="AI-generated answer to the query")
    articles: List[ArticleMetadata] = Field(..., description="List of cited articles with metadata")
    enhanced_query: Optional[str] = Field(None, description="Enhanced query used for retrieval")
    retrieval_attempts: int = Field(..., description="Number of retrieval attempts made")
    answer_attempts: int = Field(..., description="Number of answer generation attempts made")
    checker_flags: Optional[Dict] = Field(None, description="Retrieval checker evaluation flags")
    evaluator_flags: Optional[Dict] = Field(None, description="Answer evaluator flags")

# Helper functions (reused from app.py)
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
        if expected_type and isinstance(output, expected_type):
            return output
        if expected_type and isinstance(output, dict):
            return expected_type(**output)
        return output
    elif hasattr(result, 'messages') and result.messages:
        last_msg = result.messages[-1]
        if hasattr(last_msg, 'content') and isinstance(last_msg.content, expected_type):
            return last_msg.content
    return None

async def check_retrieved_chunks(user_query, retrieved_chunks):
    """Check retrieved chunks using the checker agent."""
    context = format_context(retrieved_chunks, max_context_chars=2000)
    checker_prompt = (
        f"USER_QUERY: {user_query}\n\n"
        f"RETRIEVED_CHUNKS:\n{context}\n\n"
        f"Evaluate these chunks and output JSON with sufficiency, correctness, and feedback fields."
    )
    
    checker_result = await run_sync_in_thread(Runner.run_sync, retrieve_checker_agent, checker_prompt)
    checker_response = extract_structured_output(checker_result, RetrieveCheckerResponse)
    
    if checker_response is None:
        return RetrieveCheckerResponse(
            sufficiency=False,
            correctness=False,
            feedback="Failed to extract checker response"
        )
    
    return checker_response

async def evaluate_response(user_query, answer, retrieved_chunks):
    """Evaluate the conversation agent's response using the evaluator agent."""
    context = format_context(retrieved_chunks, max_context_chars=2000)
    evaluator_prompt = (
        f"USER_QUERY: {user_query}\n\n"
        f"RETRIEVED_CHUNKS:\n{context}\n\n"
        f"AGENT_RESPONSE:\n{answer}\n\n"
        f"Evaluate the agent's response for acceptability, completeness, accuracy, and proper citation."
    )
    
    evaluator_result = await run_sync_in_thread(Runner.run_sync, evaluator_agent, evaluator_prompt)
    evaluator_response = extract_structured_output(evaluator_result, EvaluatorResponse)
    
    if evaluator_response is None:
        return EvaluatorResponse(
            is_acceptable=False,
            completeness=False,
            accuracy=False,
            proper_citation=False,
            feedback="Failed to extract evaluator response"
        )
    
    return evaluator_response

async def answer_query_with_threshold(user_query, min_score: float = DEFAULT_MIN_SCORE, max_results: int = MAX_RETRIEVAL_RESULTS):
    """
    Answer query with threshold-based retrieval:
    1. Retrieval loop: enhancer → retriever (threshold-based) → checker → (retry if needed)
    2. Answer generation: conversation agent
    3. Evaluation loop: evaluator → (retry conversation if needed)
    """
    with trace(trace_name):
        original_query = user_query
        enhanced_query = user_query
        retrieved = []
        retrieval_feedback = ""
        checker_response = None
        evaluator_response = None
        
        # ===== PHASE 1: RETRIEVAL LOOP =====
        for retrieval_attempt in range(MAX_RETRIEVAL_TRIES):
            # Step 1: Enhance the query
            try:
                if retrieval_attempt == 0:
                    enhancer_input = original_query
                else:
                    enhancer_input = (
                        f"Original query: {original_query}\n\n"
                        f"Feedback from previous retrieval: {retrieval_feedback}\n\n"
                        f"Please enhance the query based on this feedback."
                    )
                
                enhanced_query_result = await run_sync_in_thread(Runner.run_sync, enhancer_agent, enhancer_input)
                enhanced_query = extract_agent_response(enhanced_query_result)
            except Exception as e:
                print(f"Warning: Query enhancement failed: {e}")
                if retrieval_attempt == 0:
                    enhanced_query = original_query
            
            # Step 2: Retrieve relevant chunks using threshold
            retrieved = retriever.retrieve_by_threshold(
                enhanced_query,
                min_score=min_score,
                max_results=max_results
            )
            
            # Step 3: Check retrieved chunks
            checker_response = await check_retrieved_chunks(original_query, retrieved)
            
            # Step 4: Check if we should retry retrieval
            if checker_response.sufficiency and checker_response.correctness:
                break
            else:
                retrieval_feedback = checker_response.feedback or "Retrieved chunks are insufficient or incorrect."
                if retrieval_attempt < MAX_RETRIEVAL_TRIES - 1:
                    print(f"Retrieval attempt {retrieval_attempt + 1} failed. Retrying with feedback: {retrieval_feedback}")
                else:
                    print(f"Max retrieval tries ({MAX_RETRIEVAL_TRIES}) reached. Proceeding with current chunks.")
        
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
            try:
                if answer_attempt == 0:
                    conversation_prompt = (
                        f"Use only the following retrieved articles from the Indian Constitution to answer. "
                        f"Cite Article numbers (e.g., Article 15, Article 21) in your answer when you reference them.\n\n"
                        f"RETRIEVED_ARTICLES:\n{context}\n\nQUESTION: {original_query}"
                    )
                else:
                    conversation_prompt = (
                        f"Use only the following retrieved articles from the Indian Constitution to answer. "
                        f"Cite Article numbers (e.g., Article 15, Article 21) in your answer when you reference them.\n\n"
                        f"RETRIEVED_ARTICLES:\n{context}\n\n"
                        f"QUESTION: {original_query}\n\n"
                        f"FEEDBACK ON PREVIOUS ATTEMPT: {answer_feedback}\n\n"
                        f"Please improve your answer based on this feedback."
                    )
                
                answer_result = await run_sync_in_thread(Runner.run_sync, conversation_agent, conversation_prompt)
                answer = extract_agent_response(answer_result)
                if not answer:
                    answer = "I don't know."
            except Exception as e:
                print(f"Warning: Answer generation failed: {e}")
                answer = "I don't know."
            
            # Step 6: Evaluate the answer
            evaluator_response = await evaluate_response(original_query, answer, retrieved)
            
            # Step 7: Check if answer is acceptable
            if (evaluator_response.is_acceptable and 
                evaluator_response.completeness and 
                evaluator_response.accuracy and 
                evaluator_response.proper_citation):
                break
            else:
                answer_feedback = evaluator_response.feedback or "Response needs improvement."
                if answer_attempt < MAX_RETRIEVAL_TRIES - 1:
                    print(f"Answer attempt {answer_attempt + 1} failed. Retrying with feedback: {answer_feedback}")
                else:
                    print(f"Max answer tries ({MAX_RETRIEVAL_TRIES}) reached. Using current answer.")
        
        if evaluator_response is None:
            evaluator_response = EvaluatorResponse(
                is_acceptable=False,
                completeness=False,
                accuracy=False,
                proper_citation=False,
                feedback="Evaluator response not available"
            )
        
        # Format articles metadata
        articles = []
        for r in retrieved:
            meta = r.get("metadata", {}) or {}
            articles.append(ArticleMetadata(
                article_number=r.get("chunk_id") or meta.get("Article_Number", "Unknown"),
                part_number=meta.get("Part_Number"),
                part_title=meta.get("Part_Title"),
                score=r.get("score", 0.0),
                distance=r.get("distance", 0.0)
            ))
        
        return QueryResponse(
            answer=answer or "I don't know.",
            articles=articles,
            enhanced_query=enhanced_query,
            retrieval_attempts=min(retrieval_attempt + 1, MAX_RETRIEVAL_TRIES),
            answer_attempts=min(answer_attempt + 1, MAX_RETRIEVAL_TRIES),
            checker_flags=checker_response.model_dump() if checker_response else None,
            evaluator_flags=evaluator_response.model_dump() if evaluator_response else None
        )

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Indian AI Lawyer API",
        "version": "1.0.0",
        "description": "API for querying the Indian Constitution using AI agents",
        "endpoints": {
            "/query": "POST - Query the constitution (uses threshold-based retrieval)",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "retriever_loaded": retriever is not None,
        "embeddings_dim": retriever.dim if retriever else None
    }

@app.post("/query", response_model=QueryResponse)
async def query_constitution(request: QueryRequest):
    """
    Query the Indian Constitution.
    
    Uses threshold-based retrieval to fetch all articles above the specified
    similarity score threshold, rather than a fixed number of results.
    """
    try:
        response = await answer_query_with_threshold(
            user_query=request.query,
            min_score=request.min_score,
            max_results=request.max_results
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

