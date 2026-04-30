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

## `chat.py` 可选项

- `--quiet`：隐藏大部分日志、HF 警告和进度条
- `--fast`：减少检索候选数、缩减查询重写、跳过 reranker，加快响应
- `--no-rerank`：跳过 cross-encoder reranker，进一步提高速度
- `--api-key`：显式指定 `ZHIPUAI_API_KEY`

## 运行建议

- 首次使用先确保 `pdf/` 中的文档已完成预处理与索引构建
- 若文本已缓存，可直接重复使用现有 `rag_cache/` 和 `chroma_db/`
- `--rebuild-index` 或 `--reprocess-pdf` 可在数据变更后强制刷新缓存

## 目录结构简介

- `pdf/`：源 PDF 文档
- `rag_cache/`：预处理文本与 chunk 缓存
- `chroma_db/`：Chroma 向量索引文件
- `output/`：评估结果、样本输出、检查点文件
- `question/`：问题集 JSON

## 版本说明

该项目已经覆盖了一个较完整的 RAG 体系：

1. 文档预处理与页码追踪
2. 句子级 chunk 切分
3. 向量检索 + BM25 检索
4. 语言模型查询重写
5. 加权 RRF 检索融合
6. cross-encoder 重排
7. 上下文限制的 LLM 回答生成
8. 评价与结果导出

如果你想继续拓展，可考虑加入：

- 更强的检索 reranker
- 直接问答与检索混合策略
- 多轮对话上下文维护
- 更健壮的中文文本预处理和分词支持
