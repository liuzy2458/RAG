# RAG Project

This is a Retrieval-Augmented Generation (RAG) project that implements an end-to-end pipeline for PDF document retrieval, vector retrieval, BM25, hybrid retrieval, query rewriting, reranking, and answer generation.

## Key Features and Improvements

- **PDF Preprocessing**
  - Reads and cleans text from multiple PDFs in the `pdf/` directory
  - Removes noise such as tables of contents, headers/footers, and references
  - Preserves page numbers and document metadata for evidence location

- **Advanced Chunking Strategies**
  - `fixed_size_chunking`: Fixed-length chunks with overlap
  - `sentence_boundary_chunking`: Sentence-based chunking that maintains sentence integrity and page information

- **Vector Retrieval and Caching**
  - Persistent vector indexing using `ChromaDB`
  - Supports multiple vector collections; different chunking strategies can reuse existing indices
  - Automatic cache checking and index rebuilding

- **Multiple Retrieval Strategies**
  - Semantic Retrieval
  - BM25 Keyword Retrieval
  - Hybrid Retrieval: Semantic + BM25 with RRF fusion

- **Query Rewriting and Enhanced Retrieval**
  - `QueryRewriteRRFPipeline`: Uses LLM to generate retrieval-friendly queries
  - Weighted RRF with original + rewritten queries to improve hit rates
  - Identifies "general chat" questions for direct LLM responses

- **Reranking**
  - Cross-encoder reranker using `sentence-transformers`
  - Re-ranks retrieval results to improve upstream retrieval quality

- **Answer Generation with Source Constraints**
  - Generates answers using only retrieved context
  - Includes source citations to reduce hallucinations
  - Interactive `chat.py`: Direct RAG Q&A for user queries

- **Evaluation and Output**
  - Implements metrics like recall@k, precision, groundedness
  - Supports JSON, Markdown, Excel reports
  - Checkpoint caching and resume functionality

## File Descriptions

- `main.py`: Main entry point for RAG experiments; supports building, evaluating, and comparing multiple configurations
- `chat.py`: Interactive Q&A script based on the `full` RAG configuration
- `preprocess.py`: PDF text extraction and cleaning
- `chunking.py`: Text chunking, chunk caching, and vector index building
- `retrieval.py`: BM25 / Semantic / Hybrid retrieval implementations
- `query_rewrite.py`: Query rewriting and weighted RRF retrieval
- `reranker.py`: Cross-encoder result reranking
- `llm.py`: LLM calls, context building, and answer generation
- `evaluation.py`: Evaluation metrics and reporting

## Implemented RAG Configurations

`main.py` defines multiple configurations for comparative experiments:

- `baseline`: Fixed chunk + Semantic retrieval
- `advanced_chunking`: Sentence boundary chunking + Semantic retrieval
- `query_rewrite`: Fixed chunk + Semantic retrieval + Query rewriting
- `bm25`: Fixed chunk + BM25 retrieval
- `hybrid`: Fixed chunk + Semantic + BM25 hybrid retrieval
- `full`: Sentence boundary chunking + Query rewriting + Hybrid retrieval

Currently, `chat.py` uses the `full` configuration, which includes:
- Sentence boundary chunking
- LLM query rewriting
- Semantic + BM25 hybrid retrieval
- LLM answer generation

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure LLM API

**Important**: You need to configure the LLM API yourself. This project requires access to an LLM service (e.g., OpenAI, Anthropic, or similar). Please set up your API keys and configuration in the appropriate files or environment variables as required by your LLM provider. Refer to `llm.py` for details on how to integrate your LLM API.

### 3. Run Full RAG Processing Pipeline

```bash
python main.py --configs full --question-set set1 --sample-size 0
```

### 4. Run Interactive Q&A

```bash
python chat.py
```

### 5. Single Query Q&A

```bash
python chat.py --query "Briefly explain the core principles of AI governance in this document?"
```

## `chat.py` Options

- `--quiet`: Hide most logs, HF warnings, and progress bars
- `--fast`: Reduce retrieval candidates, shorten query rewriting, skip reranker for faster response
- `--no-rerank`: Skip cross-encoder reranker for even faster speed
- `--api-key`: Explicitly specify `ZHIPUAI_API_KEY`

## Usage Recommendations

- For first use, ensure documents in `pdf/` are preprocessed and indexed
- If text is cached, reuse existing `rag_cache/` and `chroma_db/`
- Use `--rebuild-index` or `--reprocess-pdf` to force cache refresh after data changes

## Directory Structure Overview

- `pdf/`: Source PDF documents
- `rag_cache/`: Preprocessed text and chunk cache
- `chroma_db/`: Chroma vector index files
- `output/`: Evaluation results, sample outputs, checkpoint files
- `question/`: Question set JSON

## Version Notes

This project covers a relatively complete RAG system:

1. Document preprocessing with page number tracking
2. Sentence-level chunk splitting
3. Vector retrieval + BM25 retrieval
4. Language model query rewriting
5. Weighted RRF retrieval fusion
6. Cross-encoder reranking
7. Context-constrained LLM answer generation
8. Evaluation and result export

If you want to expand further, consider adding:

- Stronger retrieval rerankers
- Direct QA and retrieval hybrid strategies
- Multi-turn conversation context maintenance
- More robust Chinese text preprocessing and tokenization support
