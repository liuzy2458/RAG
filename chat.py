import argparse
import builtins
import contextlib
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

# Suppress common HuggingFace verbosity and tqdm progress bars globally.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_WARNING", "1")
os.environ.setdefault("DISABLE_TQDM", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from transformers import logging as transformers_logging
    transformers_logging.set_verbosity_error()
except Exception:
    pass

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")
warnings.filterwarnings("ignore", message=".*set a HF_TOKEN.*")

with open(os.devnull, "w") as devnull:
    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        from main import (
            RAG_CONFIGS,
            build_retriever,
            prepare_chunks_and_indexes,
            retrieve_documents,
            generate_chat_answer,
            RAGLLM,
            Reranker,
        )

DEFAULT_API_KEY_ENV = "ZHIPUAI_API_KEY"
DEFAULT_GENERATION_MODEL = "glm-4.7"
DEFAULT_REWRITE_MODEL = "glm-4-flash"
DEFAULT_EMBED_MODEL = "BAAI/bge-large-en-v1.5"
FULL_CONFIG_NAME = "full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive chat using the tuned full RAG pipeline."
    )

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="User query. If omitted, enters interactive chat mode.",
    )
    parser.add_argument(
        "--pdf-dir",
        default="pdf",
        help="PDF directory containing source documents.",
    )
    parser.add_argument(
        "--cache-dir",
        default="./rag_cache",
        help="Cache directory for chunking and preprocessing.",
    )
    parser.add_argument(
        "--chroma-dir",
        default="./chroma_db",
        help="Chroma DB persistence directory.",
    )
    parser.add_argument(
        "--preprocess-cache",
        default=None,
        help="Preprocessed document cache path. Defaults to <cache-dir>/preprocessed_documents.json.",
    )
    parser.add_argument(
        "--reprocess-pdf",
        action="store_true",
        help="Re-read PDFs and overwrite the preprocessing cache.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild Chroma index even if it already exists.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=f"LLM API key. Defaults to ${DEFAULT_API_KEY_ENV} environment variable.",
    )
    parser.add_argument(
        "--generation-model",
        default=DEFAULT_GENERATION_MODEL,
        help="LLM model name used for answer generation.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to generate in response. If not specified, uses model default (4096 for glm-4.7, 2048 for others).",
    )
    parser.add_argument(
        "--rewrite-model",
        default=DEFAULT_REWRITE_MODEL,
        help="LLM model name used for query rewriting.",
    )
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help="Embedding model name for vector index creation.",
    )
    parser.add_argument(
        "--top-k-retrieve",
        type=int,
        default=10,
        help="Number of documents retrieved by the hybrid retriever.",
    )
    parser.add_argument(
        "--top-k-rerank",
        type=int,
        default=5,
        help="Number of documents to rerank and include in context.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=None,
        help="Optional maximum number of characters to include in the context.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Fixed chunk size (unused for full config but kept for compatibility).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Fixed chunk overlap (unused for full config but kept for compatibility).",
    )
    parser.add_argument(
        "--advanced-min-chunk-size",
        type=int,
        default=250,
        help="Advanced chunking minimum chunk size.",
    )
    parser.add_argument(
        "--advanced-max-chunk-size",
        type=int,
        default=800,
        help="Advanced chunking maximum chunk size.",
    )
    parser.add_argument(
        "--advanced-sentence-overlap",
        type=int,
        default=1,
        help="Advanced chunking sentence overlap.",
    )
    parser.add_argument(
        "--rewrite-count",
        type=int,
        default=3,
        help="Number of rewritten queries generated by the query rewrite pipeline.",
    )
    parser.add_argument(
        "--rewrite-top-k-per-query",
        type=int,
        default=5,
        help="Top-k retrieval per rewritten query in the query rewrite RRF pipeline.",
    )
    parser.add_argument(
        "--hybrid-rrf-k",
        type=int,
        default=60,
        help="RRF k-value used by hybrid retrieval.",
    )
    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=1.0,
        help="Semantic retriever weight in hybrid fusion.",
    )
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=0.5,
        help="BM25 retriever weight in hybrid fusion.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Run quietly by suppressing progress prints and most HF warnings.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use a faster retrieval configuration with fewer candidates and reduced query rewrite overhead.",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip cross-encoder reranking to answer faster.",
    )
    return parser.parse_args()


def format_doc_source(doc: Dict[str, Any], rank: int) -> str:
    metadata = doc.get("metadata", {}) or {}
    source = metadata.get("file_name") or metadata.get("source") or "Unknown source"
    page_start = metadata.get("page_start") or metadata.get("page_number") or "N/A"
    page_end = metadata.get("page_end") or page_start
    score = doc.get("retrieval_score")
    rerank_score = doc.get("rerank_score")
    score_parts = []
    if score is not None:
        score_parts.append(f"retrieval={score:.4f}")
    if rerank_score is not None:
        score_parts.append(f"rerank={rerank_score:.4f}")

    return (
        f"{rank}. {source}, pages {page_start}-{page_end}"
        + (f" ({', '.join(score_parts)})" if score_parts else "")
    )


def _run_silently(func, *args, **kwargs):
    real_print = builtins.print
    builtins.print = lambda *args, **kwargs: None
    try:
        return func(*args, **kwargs)
    finally:
        builtins.print = real_print


def build_rag_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    api_key = args.api_key or os.getenv(DEFAULT_API_KEY_ENV)
    if not api_key:
        raise ValueError(
            f"LLM API key is required. Set --api-key or export {DEFAULT_API_KEY_ENV}."
        )

    if args.fast:
        args.top_k_retrieve = min(args.top_k_retrieve, 5)
        args.top_k_rerank = min(args.top_k_rerank, 3)
        args.rewrite_count = min(args.rewrite_count, 1)
        args.rewrite_top_k_per_query = min(args.rewrite_top_k_per_query, 3)
        args.no_rerank = True

    # Force quiet mode in chat so dataset loading and intermediate progress are hidden.
    args.quiet = True

    config = RAG_CONFIGS[FULL_CONFIG_NAME]
    preprocess_cache = args.preprocess_cache or str(Path(args.cache_dir) / "preprocessed_documents.json")

    build_func = _run_silently if args.quiet else lambda func, *a, **k: func(*a, **k)
    prepared_by_strategy = build_func(
        prepare_chunks_and_indexes,
        selected_configs=[config],
        pdf_dir=args.pdf_dir,
        cache_dir=args.cache_dir,
        preprocess_cache=preprocess_cache,
        reprocess_pdf=args.reprocess_pdf,
        chroma_dir=args.chroma_dir,
        embed_model_name=args.embed_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        advanced_min_chunk_size=args.advanced_min_chunk_size,
        advanced_max_chunk_size=args.advanced_max_chunk_size,
        advanced_sentence_overlap=args.advanced_sentence_overlap,
        rebuild=args.rebuild_index,
    )

    component_cache: Dict[str, Any] = {}
    retriever = build_retriever(
        config=config,
        prepared_entry=prepared_by_strategy[config.chunking],
        chroma_dir=args.chroma_dir,
        embed_model_name=args.embed_model,
        hybrid_rrf_k=args.hybrid_rrf_k,
        semantic_weight=args.semantic_weight,
        bm25_weight=args.bm25_weight,
        component_cache=component_cache,
    )

    generator = RAGLLM(api_key=api_key, model=args.generation_model, max_tokens=args.max_tokens)
    reranker = Reranker()

    if args.quiet:
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["HF_HUB_DISABLE_PROGRESS"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["DISABLE_TQDM"] = "1"
        os.environ["TQDM_DISABLE"] = "1"

    return {
        "config": config,
        "retriever": retriever,
        "generator": generator,
        "reranker": reranker,
        "component_cache": component_cache,
    }


def answer_user_query(
    query: str,
    config: Any,
    retriever: Any,
    generator: RAGLLM,
    reranker: Reranker,
    component_cache: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    retrieval_result = retrieve_documents(
        query=query,
        config=config,
        retriever=retriever,
        component_cache=component_cache,
        api_key=args.api_key or os.getenv(DEFAULT_API_KEY_ENV),
        rewrite_model=args.rewrite_model,
        top_k_retrieve=args.top_k_retrieve,
        rewrite_count=args.rewrite_count,
        rewrite_top_k_per_query=args.rewrite_top_k_per_query,
    )

    query_type = retrieval_result.get("query_type")
    rewritten_queries = retrieval_result.get("rewritten_queries", [])
    retrieved_docs = retrieval_result.get("retrieved_docs", [])

    if query_type == "general_chat":
        answer = generate_chat_answer(generator=generator, query=query)
        reranked_docs = []
        context = ""
    else:
        if args.no_rerank:
            reranked_docs = retrieved_docs[: args.top_k_rerank]
        else:
            reranked_docs = reranker.rerank(
                query=query,
                documents=retrieved_docs,
                top_k=args.top_k_rerank,
            )
        context = generator.build_context(
            docs=reranked_docs,
            max_context_chars=args.max_context_chars,
        )
        answer = generator.generate_answer_from_context(
            query=query,
            context=context,
        )

    if not answer.strip():
        answer = generate_chat_answer(generator=generator, query=query)

    return {
        "query": query,
        "query_type": query_type,
        "rewritten_queries": rewritten_queries,
        "retrieved_docs": retrieved_docs,
        "reranked_docs": reranked_docs,
        "context": context,
        "answer": answer,
    }


def print_result(result: Dict[str, Any]) -> None:
    answer_text = result["answer"].strip() or "[No answer produced.]"
    print("\n" + "=" * 80)
    print(answer_text)
    print("=" * 80 + "\n")


def run_interactive(pipeline: Dict[str, Any], args: argparse.Namespace) -> None:
    print("Loading dataset...")
    while True:
        try:
            query = input("User> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("Exiting chat.")
            break

        result = answer_user_query(
            query=query,
            config=pipeline["config"],
            retriever=pipeline["retriever"],
            generator=pipeline["generator"],
            reranker=pipeline["reranker"],
            component_cache=pipeline["component_cache"],
            args=args,
        )
        print_result(result)


def main() -> None:
    args = parse_args()
    pipeline = build_rag_pipeline(args)

    if args.query:
        result = answer_user_query(
            query=args.query,
            config=pipeline["config"],
            retriever=pipeline["retriever"],
            generator=pipeline["generator"],
            reranker=pipeline["reranker"],
            component_cache=pipeline["component_cache"],
            args=args,
        )
        print_result(result)
    else:
        run_interactive(pipeline=pipeline, args=args)


if __name__ == "__main__":
    main()
