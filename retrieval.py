# retrieval.py
# Semantic, BM25, and hybrid retrieval with metadata support

import chromadb
import heapq
import re
from typing import List, Tuple, Dict, Any, Optional, Union
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


DocumentInput = Union[str, Dict[str, Any], Any]
RetrievedDoc = Dict[str, Any]


def add_page_fields(doc: RetrievedDoc) -> RetrievedDoc:
    metadata = doc.get("metadata", {}) or {}

    for field in ("page_number", "page_start", "page_end"):
        if field in metadata and field not in doc:
            doc[field] = metadata[field]

    return doc


class BM25Retriever:
    """Keyword-based retrieval using BM25."""

    def __init__(self, documents: List[DocumentInput]):
        try:
            from rank_bm25 import BM25Okapi

        except ImportError:
            raise ImportError("BM25 requires: pip install rank-bm25")

        self.documents = []

        for i, doc in enumerate(documents):
            if isinstance(doc, dict):
                text = doc.get("text", "")
                metadata = doc.get("metadata", {})
            elif hasattr(doc, "text"):
                text = doc.text
                metadata = getattr(doc, "metadata", {}) or {}
            else:
                text = str(doc)
                metadata = {}

            self.documents.append(
                {
                    "id": metadata.get("chunk_id", f"bm25_doc_{i}"),
                    "text": text,
                    "metadata": metadata,
                }
            )

        self.tokenized_docs = [
            self._tokenize(doc["text"])
            for doc in self.documents
        ]

        self.bm25 = BM25Okapi(self.tokenized_docs) if self.tokenized_docs else None

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9_]+", text.lower())

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievedDoc]:
        """Retrieve documents using BM25."""
        if self.bm25 is None:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_items = heapq.nlargest(
            min(top_k, len(self.documents)),
            zip(self.documents, scores),
            key=lambda item: item[1],
        )

        return [
            add_page_fields({
                "id": doc["id"],
                "text": doc["text"],
                "metadata": doc["metadata"],
                "retrieval_score": float(score),
                "retrieval_method": "bm25",
            })
            for doc, score in top_items
        ]

    def search(self, query: str, top_k: int = 10) -> List[RetrievedDoc]:
        """Alias for compatibility."""
        return self.retrieve(query, top_k=top_k)


class SemanticRetriever:
    """Semantic retrieval using vector similarity."""

    def __init__(
        self,
        collection_name: str,
        chroma_dir: str = "./chroma_db",
        embed_model_name: str = "BAAI/bge-large-en-v1.5",
    ):
        self.collection_name = collection_name
        self.chroma_dir = chroma_dir
        Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
        self.index = self._load_index()

    def _load_index(self):
        chroma_client = chromadb.PersistentClient(path=self.chroma_dir)
        chroma_collection = chroma_client.get_collection(self.collection_name)

        vector_store = ChromaVectorStore(
            chroma_collection=chroma_collection
        )

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )

        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
        )

        return index

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievedDoc]:
        """Retrieve documents using semantic similarity."""
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)

        results = []

        for i, node in enumerate(nodes):
            node_id = getattr(node.node, "node_id", None) or getattr(node, "node_id", None)
            text = node.text
            metadata = getattr(node.node, "metadata", {}) or {}
            score = node.score if getattr(node, "score", None) is not None else 1.0

            results.append(
                add_page_fields({
                    "id": metadata.get("chunk_id", node_id or f"semantic_doc_{i}"),
                    "text": text,
                    "metadata": metadata,
                    "retrieval_score": float(score),
                    "retrieval_method": "semantic",
                })
            )

        return results

    def search(self, query: str, top_k: int = 10) -> List[RetrievedDoc]:
        """Alias for compatibility."""
        return self.retrieve(query, top_k=top_k)


class HybridRetriever:
    """Hybrid retrieval combining semantic search and BM25 using RRF."""

    def __init__(
        self,
        semantic_retriever: SemanticRetriever,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
        semantic_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ):
        self.semantic = semantic_retriever
        self.bm25 = bm25_retriever
        self.rrf_k = rrf_k
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight

    def _get_doc_id(self, doc: RetrievedDoc) -> str:
        if doc.get("id"):
            return str(doc["id"])

        metadata = doc.get("metadata", {})
        if metadata.get("chunk_id"):
            return str(metadata["chunk_id"])

        if metadata.get("source") and metadata.get("page"):
            return f"{metadata['source']}_{metadata['page']}_{hash(doc.get('text', ''))}"

        return str(hash(doc.get("text", "")))

    def _rrf_fuse(
        self,
        ranked_lists: List[Tuple[List[RetrievedDoc], float, str]],
        final_top_k: int,
        rrf_k: Optional[int] = None,
    ) -> List[RetrievedDoc]:
        """Fuse ranked lists using weighted RRF."""
        if rrf_k is None:
            rrf_k = self.rrf_k

        doc_store: Dict[str, RetrievedDoc] = {}
        rrf_scores: Dict[str, float] = {}

        for results, weight, method_name in ranked_lists:
            for rank, doc in enumerate(results, start=1):
                doc_id = self._get_doc_id(doc)

                if doc_id not in doc_store:
                    doc_store[doc_id] = add_page_fields({
                        "id": doc_id,
                        "text": doc.get("text", ""),
                        "metadata": doc.get("metadata", {}),
                        "retrieval_score": 0.0,
                        "retrieval_method": "hybrid",
                        "source_scores": {},
                    })

                rrf_score = weight * (1 / (rrf_k + rank))
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_score

                doc_store[doc_id]["source_scores"][method_name] = doc.get(
                    "retrieval_score", 0.0
                )

        ranked_doc_ids = sorted(
            rrf_scores.keys(),
            key=lambda doc_id: rrf_scores[doc_id],
            reverse=True,
        )

        final_docs = []

        for doc_id in ranked_doc_ids[:final_top_k]:
            doc = doc_store[doc_id]
            doc["retrieval_score"] = float(rrf_scores[doc_id])
            final_docs.append(doc)

        return final_docs

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        semantic_top_k: Optional[int] = None,
        bm25_top_k: Optional[int] = None,
        rrf_k: Optional[int] = None,
    ) -> List[RetrievedDoc]:
        """
        Hybrid retrieval with RRF.

        top_k is the final output size.
        semantic_top_k and bm25_top_k control internal retrieval size.
        """
        if semantic_top_k is None:
            semantic_top_k = top_k

        if bm25_top_k is None:
            bm25_top_k = top_k

        semantic_results = self.semantic.retrieve(
            query=query,
            top_k=semantic_top_k,
        )

        bm25_results = self.bm25.retrieve(
            query=query,
            top_k=bm25_top_k,
        )

        return self._rrf_fuse(
            ranked_lists=[
                (semantic_results, self.semantic_weight, "semantic"),
                (bm25_results, self.bm25_weight, "bm25"),
            ],
            final_top_k=top_k,
            rrf_k=rrf_k,
        )

    def search(self, query: str, top_k: int = 10) -> List[RetrievedDoc]:
        """Alias for query rewrite pipeline."""
        return self.retrieve(query=query, top_k=top_k)



