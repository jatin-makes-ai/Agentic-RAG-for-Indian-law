# AI Lawyer - Agentic RAG System

An intelligent legal assistant system that uses Retrieval-Augmented Generation (RAG) with multiple AI agents to answer questions about the Indian Constitution.

## Overview

This project implements an agentic RAG system that combines:
- **Vector Search**: FAISS-based semantic search over the Indian Constitution
- **AI Agents**: Multiple specialized agents working together to enhance queries, validate retrieval, and generate answers
- **Iterative Improvement**: Automatic retry mechanism that refines queries based on retrieval quality feedback

## Architecture

### System Flow

1. **Query Enhancement** → User query is enhanced by the enhancer agent for better retrieval
2. **Retrieval** → FAISS vector search retrieves relevant chunks from the Indian Constitution
3. **Quality Check** → Retrieve checker agent evaluates if chunks are sufficient and correct
4. **Iterative Refinement** → If quality is insufficient, feedback is sent back to enhancer for retry (max 3 attempts)
5. **Answer Generation** → Conversation agent generates final answer using validated chunks
6. **Final Evaluator** → The Evaluator Agent checks for hallucination and completeness of the generated response and can decide to send for retries

### Agents

- **Enhancer Agent**: Optimizes user queries for better semantic search results
- **Retrieve Checker Agent**: Validates retrieval quality using Pydantic-structured output (sufficiency, correctness, feedback)
- **Conversation Agent**: Generates concise, cited answers from retrieved context
- **Evaluator Agent**: Evaluates the generated response to the user's query

## Features

- **Multi-Agent Architecture**: Specialized agents for different tasks
- **Quality Assurance**: Automatic validation of retrieved chunks
- **Iterative Refinement**: Self-improving query enhancement loop
- **Structured Output**: Pydantic models for type-safe agent responses
- **Tracing**: OpenAI platform integration for monitoring LLM calls
- **Web Interface**: Gradio-based UI for easy interaction
- **Website Integration**: Currently working on integrating with a publically hosted website

## Project Structure

```
AI-Lawyer/
├── ai_agents/              # AI agent definitions
│   ├── enhancer_agent.py
│   ├── conversation_agent.py
│   └── retrieve_checker_agent.py
├── scripts/                 # Data processing scripts
│   ├── pdf_to_text.py      # PDF extraction
│   ├── text_to_chunks.py   # Text chunking
│   ├── embed_chunks.py     # Embedding generation
│   ├── build_faiss_from_jsonl.py  # FAISS index building
│   └── retriever.py        # Retrieval logic
├── data/                   # Data files
│   ├── raw/                # Original PDF
│   ├── text/               # Extracted text
│   ├── chunks/             # Text chunks
│   └── embeddings/         # Embeddings and FAISS index
├── app.py                  # Main application (Gradio UI)
└── requirements.txt        # Dependencies
```

## Setup

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables (create `.env` file):
   ```env
   OPENAI_API_KEY=your_api_key_here
   CONVERSATION_LLM_MODEL=gpt-4o-mini
   EMBEDDING_MODEL=text-embedding-3-large
   RETRIEVE_CHECKER_AGENT_MODEL=gpt-4o-mini
   MAX_RETRIEVAL_TRIES=3
   EMB_JSONL=data/embeddings/the_indian_constitution_embeddings.jsonl
   FAISS_INDEX=data/embeddings/the_indian_constitution.faiss
   FAISS_IDMAP=data/embeddings/the_indian_constitution.idmap.jsonl
   PORT=7860
   ```

### Data Pipeline (if starting from scratch)

1. **Extract text from PDF**:
   ```bash
   python scripts/pdf_to_text.py
   ```

2. **Chunk the text**:
   ```bash
   python scripts/text_to_chunks.py
   ```

3. **Generate embeddings**:
   ```bash
   python scripts/embed_chunks.py
   ```

4. **Build FAISS index**:
   ```bash
   python scripts/build_faiss_from_jsonl.py
   ```

## Usage

### Running the Application

```bash
python app.py
```

The Gradio interface will be available at `http://localhost:7860` (or the port specified in your `.env`).

### Querying

1. Enter your question about the Indian Constitution
2. Adjust the `Top K` slider to control how many chunks to retrieve
3. Click "Ask" to get an answer with source citations

## Technology Stack

- **OpenAI Agents SDK**: Multi-agent orchestration
- **FAISS**: Vector similarity search
- **Gradio**: Web interface
- **Pydantic**: Structured data validation
- **OpenAI API**: Embeddings and LLM inference

## Key Components

### Retriever (`scripts/retriever.py`)
- FAISS-based vector search with fallback to linear scan
- L2 distance calculation with score normalization
- Supports metadata filtering

### Agents (`ai_agents/`)
- Built using OpenAI Agents framework
- Structured outputs with Pydantic models
- Integrated tracing for monitoring

### Main Application (`app.py`)
- Orchestrates the multi-agent workflow
- Implements retry loop with quality checking
- Provides Gradio web interface

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `CONVERSATION_LLM_MODEL` | Model for conversation agent | `gpt-4o-mini` |
| `EMBEDDING_MODEL` | Model for embeddings | `text-embedding-3-large` |
| `RETRIEVE_CHECKER_AGENT_MODEL` | Model for checker agent | `gpt-4o-mini` |
| `MAX_RETRIEVAL_TRIES` | Max retry attempts | `3` |
| `EMB_JSONL` | Path to embeddings JSONL | `data/embeddings/...` |
| `FAISS_INDEX` | Path to FAISS index | `data/embeddings/...` |
| `FAISS_IDMAP` | Path to FAISS ID map | `data/embeddings/...` |
| `PORT` | Gradio server port | `7860` |

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
