# reranker.py
# Cross-encoder reranker module with metadata support

from sentence_transformers import CrossEncoder
from typing import List, Tuple, Dict, Any, Union


RetrievedDoc = Dict[str, Any]
DocumentInput = Union[str, Tuple[str, float], RetrievedDoc]


class Reranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 16,
    ):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace cross-encoder model name.
            batch_size: Batch size for inference.
        """
        self.model = CrossEncoder(model_name)
        self.batch_size = batch_size

    def _normalize_documents(
        self,
        documents: List[DocumentInput],
    ) -> List[RetrievedDoc]:
        """Convert different input formats into a unified dictionary format."""
        normalized_docs = []

        for i, doc in enumerate(documents):
            if isinstance(doc, dict):
                metadata = doc.get("metadata", {}) or {}
                normalized_doc = {
                    "id": doc.get("id", f"doc_{i}"),
                    "text": doc.get("text", ""),
                    "metadata": metadata,
                    "retrieval_score": float(doc.get("retrieval_score", 0.0)),
                    "retrieval_method": doc.get("retrieval_method", "unknown"),
                    "source_scores": doc.get("source_scores", {}),
                }

                for field in ("page_number", "page_start", "page_end"):
                    if field in doc:
                        normalized_doc[field] = doc[field]
                    elif field in metadata:
                        normalized_doc[field] = metadata[field]

                normalized_docs.append(
                    normalized_doc
                )

            elif isinstance(doc, tuple):
                normalized_docs.append(
                    {
                        "id": f"doc_{i}",
                        "text": doc[0],
                        "metadata": {},
                        "retrieval_score": float(doc[1]),
                        "retrieval_method": "unknown",
                        "source_scores": {},
                    }
                )

            else:
                normalized_docs.append(
                    {
                        "id": f"doc_{i}",
                        "text": str(doc),
                        "metadata": {},
                        "retrieval_score": 0.0,
                        "retrieval_method": "unknown",
                        "source_scores": {},
                    }
                )

        return normalized_docs

    def rerank(
        self,
        query: str,
        documents: List[DocumentInput],
        top_k: int = 5,
    ) -> List[RetrievedDoc]:
        """
        Rerank documents using cross-encoder.

        Args:
            query: Original user query.
            documents:
                - List[str]
                - List[(text, retrieval_score)]
                - List[dict] with text, metadata, and retrieval_score.
            top_k: Number of top results to return.

        Returns:
            List of dictionaries containing:
                id, text, metadata, retrieval_score, rerank_score, retrieval_method.
        """
        if not documents:
            return []

        docs = self._normalize_documents(documents)

        pairs = [
            (query, doc["text"])
            for doc in docs
        ]

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        for i, score in enumerate(scores):
            docs[i]["rerank_score"] = float(score)

        docs.sort(
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        return docs[:top_k]

    def rerank_only_text(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Simplified rerank method for compatibility.

        Returns:
            List of (document_text, rerank_score).
        """
        results = self.rerank(
            query=query,
            documents=documents,
            top_k=top_k,
        )

        return [
            (doc["text"], doc["rerank_score"])
            for doc in results
        ]
