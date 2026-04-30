# RAG for AI Governance Documents

## Project Overview

This project investigates how to improve Retrieval-Augmented Generation (RAG) systems for long, complex AI governance and policy documents.

Unlike standard RAG pipelines, this project focuses on **strategy-level optimizations** rather than low-level parameter tuning. It aims to address challenges such as:
- heterogeneous document structures
- dispersed information across sections
- semantic and terminological mismatch

The goal is to improve retrieval quality and generate more accurate, grounded answers.

---

## Pipeline

Query → Query Rewriting → Hybrid Retrieval → Reranking → LLM Generation

---

## Key Features

### PDF Preprocessing
- Extracts and cleans text from PDFs in the `pdf/` directory
- Removes noise (headers, footers, references, etc.)
- Preserves page numbers and metadata for source attribution

### Advanced Chunking
- `fixed_size_chunking`: fixed-length chunks with overlap
- `sentence_boundary_chunking`: sentence-based chunks that preserve semantic coherence

### Retrieval Strategies
- Semantic retrieval (vector search)
- BM25 keyword retrieval
- Hybrid retrieval (semantic + BM25 with RRF fusion)

### Query Rewriting
- Uses LLM to generate retrieval-oriented query variants
- Improves alignment with document terminology

### Reranking
- Cross-encoder reranker (sentence-transformers)
- Improves relevance of retrieved chunks

### Answer Generation
- Generates answers strictly based on retrieved context
- Includes source citations to reduce hallucination

### Evaluation
- Metrics: answer-supported recall, precision, groundedness
- Supports JSON / Markdown / Excel outputs

---

## Implemented Configurations

Defined in `main.py`:

- `baseline`: Fixed chunk + Semantic retrieval
- `advanced_chunking`: Sentence chunking + Semantic retrieval
- `query_rewrite`: Fixed chunk + Query rewriting
- `bm25`: Fixed chunk + BM25 retrieval
- `hybrid`: Fixed chunk + Hybrid retrieval
- `full`: Sentence chunking + Query rewriting + Hybrid retrieval

---

## Full System (Final Model)

The final optimized RAG system includes:

- Sentence-boundary chunking
- LLM-based query rewriting
- Hybrid retrieval (BM25 + semantic + RRF)
- Cross-encoder reranking

This configuration is used for both evaluation and interactive Q&A.

---

## Key Insights

- Chunking quality is critical for retrieval performance
- Query rewriting improves precision more than recall
- Hybrid retrieval increases coverage but introduces noise
- Best performance comes from combining multiple components

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
### 2. Configure LLM API

Set up your API key (e.g., OpenAI / Zhipu / Anthropic) in environment variables or in llm.py.

### 3. Run Full Pipeline

```bash
python main.py --configs full --question-set all --sample-size 30
```

### 4. Interactive Q&A

```bash
python chat.py
```

### 5. Single Query
```bash
python chat.py --query "Explain AI governance principles"
```

## chat.py Options
--quiet: reduce logs
--fast: faster but lower accuracy
--no-rerank: skip reranker
--api-key: manually specify API key

## Directory Structure
pdf/: source documents
rag_cache/: processed text and chunks
chroma_db/: vector database
output/: evaluation results
question/: question sets

## Future Work
stronger reranking models
multi-turn conversation support
improved retrieval strategies
better handling of long-context reasoning
