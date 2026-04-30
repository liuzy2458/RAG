# llm.py
# Handles LLM interactions for answer generation

import os
import time
from typing import Dict, Any, List, Optional

from llama_index.llms.zhipuai import ZhipuAI


def complete_with_retry(
    llm: Any,
    prompt: str,
    max_attempts: int = 5,
    initial_delay: float = 1.0,
    backoff: float = 2.0,
) -> Any:
    """
    Call llm.complete with retry for transient API failures such as timeout.
    """
    last_error = None
    delay = initial_delay

    for attempt in range(1, max_attempts + 1):
        try:
            return llm.complete(prompt)
        except Exception as exc:
            last_error = exc

            if attempt >= max_attempts:
                break

            print(
                f"[Warning] LLM call failed "
                f"(attempt {attempt}/{max_attempts}): {exc}. Retrying in {delay:.1f}s."
            )
            time.sleep(delay)
            delay *= backoff

    raise last_error


class RAGLLM:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "glm-4-flash",
        temperature: float = 0,
        max_tokens: Optional[int] = None,
    ):
        """
        Initialize ZhipuAI LLM.

        Args:
            api_key: ZhipuAI API key. If None, it will be read from environment variable.
            model: LLM model name.
            max_tokens: Maximum number of tokens to generate. If None, uses model default.
        """
        if api_key is None:
            api_key = os.getenv("ZHIPUAI_API_KEY")

        if not api_key:
            raise ValueError(
                "ZHIPUAI_API_KEY is missing. Pass api_key or set environment variable."
            )

        # Set default max_tokens based on model if not specified
        if max_tokens is None:
            if "glm-4.7" in model:
                max_tokens = 4096  # Longer context for newer models
            else:
                max_tokens = 2048  # Default for other models

        self.llm = ZhipuAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _extract_text_from_doc(self, doc: Any) -> str:
        """
        Extract text from different possible document formats.
        """
        if isinstance(doc, dict):
            return doc.get("text", "")

        if hasattr(doc, "text"):
            return doc.text

        if isinstance(doc, str):
            return doc

        return str(doc)

    def _extract_metadata_from_doc(self, doc: Any) -> Dict[str, Any]:
        """
        Extract metadata from different possible document formats.
        """
        if isinstance(doc, dict):
            return doc.get("metadata", {}) or {}

        if hasattr(doc, "metadata"):
            return doc.metadata or {}

        return {}

    def _format_source_label(self, metadata: Dict[str, Any], index: int) -> str:
        """
        Format source label from metadata for citation-like context display.
        """
        source = (
            metadata.get("file_name")
            or metadata.get("source_name")
            or metadata.get("source")
            or metadata.get("document")
            or "Unknown source"
        )

        page = self._format_page_label(metadata)

        section = (
            metadata.get("section_title")
            or metadata.get("section")
            or metadata.get("heading")
            or metadata.get("nearby_heading")
            or ""
        )

        label = f"[Source {index}: {source}, page {page}"

        if section:
            label += f", section: {section}"

        label += "]"

        return label

    def _format_page_label(self, metadata: Dict[str, Any]) -> str:
        if metadata.get("page_start") is not None and metadata.get("page_end") is not None:
            if metadata["page_start"] == metadata["page_end"]:
                return str(metadata["page_start"])
            return f"{metadata['page_start']}-{metadata['page_end']}"

        return (
            metadata.get("page_number")
            or metadata.get("page")
            or metadata.get("page_num")
            or "Unknown page"
        )

    def build_context(
        self,
        docs: List[Any],
        max_context_chars: Optional[int] = None,
    ) -> str:
        """
        Build context string from retrieved or reranked documents.

        Args:
            docs: List of retrieved/reranked document dictionaries.
            max_context_chars: Optional maximum character length for context.

        Returns:
            Formatted context string with source metadata.
        """
        context_blocks = []
        total_chars = 0

        for i, doc in enumerate(docs, start=1):
            text = self._extract_text_from_doc(doc).strip()
            metadata = self._extract_metadata_from_doc(doc)

            if not text:
                continue

            source_label = self._format_source_label(metadata, i)
            block = f"{source_label}\n{text}"

            if max_context_chars is not None:
                if total_chars + len(block) > max_context_chars:
                    remaining = max_context_chars - total_chars

                    if remaining <= 0:
                        break

                    block = block[:remaining]

                total_chars += len(block)

            context_blocks.append(block)

        return "\n\n".join(context_blocks)

    def generate_answer(
        self,
        query: str,
        docs: List[Any] | str,
        max_context_chars: Optional[int] = None,
    ) -> str:
        """
        Generate an answer using only retrieved/reranked documents.

        Args:
            query: Original user query.
            docs: Retrieved or reranked documents with text and metadata.
            max_context_chars: Optional maximum character length for context.

        Returns:
            Generated answer string.
        """
        if isinstance(docs, str):
            context = docs
        else:
            context = self.build_context(
                docs=docs,
                max_context_chars=max_context_chars,
            )

        prompt = f"""
You are a RAG-based question answering assistant.

Answer the question using ONLY the retrieved context below.
If the context does not contain enough information, say:
"I cannot answer this question based on the provided documents."

When using information from the context, cite the source using the source label, such as [Source 1], [Source 2].
Do not use external knowledge.

Retrieved Context:
{context}

Question:
{query}

Answer:
"""

        response = complete_with_retry(self.llm, prompt)
        return response.text.strip()

    def generate_answer_from_context(
        self,
        query: str,
        context: str,
    ) -> str:
        """
        Generate an answer from a pre-built context string.

        This method is kept for compatibility with older pipeline code.
        """
        prompt = f"""
You are a RAG-based question answering assistant.

Answer the question using ONLY the retrieved context below.
If the context does not contain enough information, say:
"I cannot answer this question based on the provided documents."

Do not use external knowledge.

Retrieved Context:
{context}

Question:
{query}

Answer:
"""

        response = complete_with_retry(self.llm, prompt)
        return response.text.strip()
    

