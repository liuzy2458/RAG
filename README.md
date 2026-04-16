# RAG System for AI Governance QA

## Overview
This project is an end-to-end Retrieval-Augmented Generation (RAG) system designed for question answering over domain-specific AI governance documents, including reports from UNESCO, NIST, and OECD.

The system aims to improve factual accuracy and reduce hallucination by grounding large language model (LLM) outputs in retrieved external knowledge.

---

## Motivation
Large language models are powerful but often generate ungrounded or hallucinated responses. This project explores how retrieval-augmented methods can enhance reliability when working with authoritative policy and governance documents.

The system is designed with a focus on:
- factual correctness
- completeness of answers
- groundedness in source documents

---

## System Pipeline

The overall pipeline consists of four main stages:

1. **Document Processing**
   - PDF parsing using PyMuPDF
   - Text cleaning and preprocessing
   - Semantic chunking with overlap

2. **Embedding & Storage**
   - Text embeddings generated using BGE models
   - Stored in a Chroma vector database for efficient retrieval

3. **Retrieval & Reranking**
   - Top-k semantic similarity retrieval
   - Cross-encoder reranking to improve relevance

4. **Answer Generation**
   - LLM generates answers based only on retrieved content
   - Prompt design enforces grounded responses

---

## Key Features

- End-to-end RAG pipeline implementation  
- Domain-specific QA over AI governance documents  
- Semantic retrieval with embedding models  
- Retrieval comparison (with and without reranking)  
- Cross-encoder reranking for improved relevance  
- LLM-based answer generation with controlled prompts  
- Evaluation framework based on correctness, completeness, and groundedness  

---

## Project Structure
├── rag.py # Main RAG pipeline

├── query_only.py # Baseline retrieval + generation

├── README.md

└── requirements.txt

---

## Tech Stack

- Python  
- PyMuPDF (PDF parsing)  
- Chroma (vector database)  
- BGE Embeddings (semantic representation)  
- Sentence Transformers / Cross-Encoder (reranking)  
- LLM API (e.g., GLM / OpenAI)  

---

## Example Use Case

**Question:**  
What are the core values in the UNESCO Recommendation on the Ethics of AI?

**Approach:**  
The system retrieves relevant document chunks and generates an answer grounded in the source material, improving factual accuracy compared to LLM-only responses.

---

## Evaluation

The system is evaluated based on:

- **Correctness** – factual accuracy of the answer  
- **Completeness** – coverage of key information  
- **Groundedness** – alignment with retrieved sources  

This evaluation framework is used to compare different retrieval strategies and system configurations.

---

## Notes & Future Work

This project is still ongoing. Planned improvements include integrating advanced retrieval strategies such as hybrid retrieval (combining semantic and keyword search) and section-aware retrieval, as well as further optimization of the retrieval and reranking pipeline to improve accuracy and robustness.

---

## Author

Ziyang Liu  
BSc in Statistics, The Chinese University of Hong Kong  

---

## Disclaimer

This project was developed for academic and research purposes and is intended to demonstrate practical experience in building and evaluating RAG systems.
