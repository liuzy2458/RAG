# Evaluation module for RAG system
# Metrics: Page Recall, Answer-Supported Recall, Precision, and Groundedness

import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from llama_index.llms.zhipuai import ZhipuAI
from llm import complete_with_retry


class RAGEvaluator:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "glm-4-flash",
        temperature: float = 0,
    ):
        if api_key is None:
            api_key = os.getenv("ZHIPUAI_API_KEY")

        if not api_key:
            raise ValueError("ZHIPUAI_API_KEY is missing.")

        self.llm = ZhipuAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
        )

    # -------------------------
    # Utility
    # -------------------------

    def _normalize_source(self, source: str) -> str:
        if not source:
            return ""

        basename = os.path.basename(str(source)).lower().strip()
        stem, _ = os.path.splitext(basename)
        return re.sub(r"[^a-z0-9]+", "", stem)

    def _extract_metadata(self, doc: Any) -> Dict[str, Any]:
        if isinstance(doc, dict):
            return doc.get("metadata", {}) or {}

        if hasattr(doc, "metadata"):
            return doc.metadata or {}

        return {}

    def _get_doc_source(self, doc: Any) -> str:
        metadata = self._extract_metadata(doc)
        return (
            metadata.get("file_name")
            or metadata.get("source_name")
            or metadata.get("source")
            or metadata.get("document")
            or ""
        )

    def _get_doc_text(self, doc: Any) -> str:
        if isinstance(doc, dict):
            return doc.get("text", "")

        if hasattr(doc, "text"):
            return doc.text

        return str(doc)

    def _parse_page_ranges(self, page_spec: Any) -> List[Tuple[int, int]]:
        ranges = []

        if page_spec is None:
            return ranges

        if isinstance(page_spec, int):
            return [(page_spec, page_spec)]

        if isinstance(page_spec, list):
            for item in page_spec:
                if isinstance(item, int):
                    ranges.append((item, item))
                elif isinstance(item, tuple) and len(item) == 2:
                    ranges.append((int(item[0]), int(item[1])))
                elif isinstance(item, str):
                    ranges.extend(self._parse_page_ranges(item))
            return ranges

        page_spec = str(page_spec)
        page_spec = page_spec.replace("–", "-").replace("—", "-").replace("−", "-")
        page_spec = re.sub(r"\s+to\s+", "-", page_spec, flags=re.IGNORECASE)
        parts = [part.strip() for part in page_spec.split(",") if part.strip()]

        for part in parts:
            if "-" in part:
                try:
                    start, end = part.split("-", 1)
                    ranges.append((int(start.strip()), int(end.strip())))
                except ValueError:
                    continue
            else:
                try:
                    page = int(part)
                    ranges.append((page, page))
                except ValueError:
                    continue

        return ranges

    def _get_doc_page_ranges(self, doc: Any) -> List[Tuple[int, int]]:
        metadata = self._extract_metadata(doc)

        if "page_start" in metadata and "page_end" in metadata:
            try:
                return [(int(metadata["page_start"]), int(metadata["page_end"]))]
            except (TypeError, ValueError):
                pass

        page_info = (
            metadata.get("page_number")
            or metadata.get("page")
            or metadata.get("page_num")
        )
        return self._parse_page_ranges(page_info)

    def _get_evidence_list(self, ground_truth: Dict[str, Any]) -> List[Dict[str, Any]]:
        evidence = ground_truth.get("evidence", [])

        if not isinstance(evidence, list):
            raise ValueError("'evidence' must be a list.")

        return evidence

    def _extract_json_object(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise
            parsed = json.loads(match.group(0))

        if not isinstance(parsed, dict):
            raise ValueError("Expected evaluator output to be a JSON object.")

        return parsed

    def _coerce_score(self, value: Any, default: float = 0.0) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return default

        return round(max(0.0, min(1.0, score)), 2)

    def _coerce_precision_scores(self, value: Any, expected_count: int) -> List[int]:
        if not isinstance(value, list):
            return [0] * expected_count

        scores = []
        for item in value:
            try:
                scores.append(1 if int(item) == 1 else 0)
            except (TypeError, ValueError):
                scores.append(0)

        if len(scores) < expected_count:
            scores.extend([0] * (expected_count - len(scores)))

        return scores[:expected_count]

    # -------------------------
    # 1. Page Recall
    # -------------------------

    def page_recall_at_k(
        self,
        docs: List[Any],
        ground_truth: Dict[str, Any],
        k: int = 5,
    ) -> float:
        """
        Coarse-grained source/page coverage.

        Returns the fraction of standard evidence pages covered by retrieved chunks.
        Source name and page range must both match.
        """
        target_pages = set()

        for evidence in self._get_evidence_list(ground_truth):
            true_source = self._normalize_source(evidence.get("source", ""))
            for start, end in self._parse_page_ranges(evidence.get("pages")):
                for page in range(start, end + 1):
                    target_pages.add((true_source, page))

        if not target_pages:
            return 0.0

        covered_pages = set()

        for doc in docs[:k]:
            doc_source = self._normalize_source(self._get_doc_source(doc))
            doc_ranges = self._get_doc_page_ranges(doc)

            for true_source, true_page in target_pages:
                if doc_source != true_source:
                    continue

                for doc_start, doc_end in doc_ranges:
                    if doc_start <= true_page <= doc_end:
                        covered_pages.add((true_source, true_page))
                        break

        return round(len(covered_pages) / len(target_pages), 2)

    # -------------------------
    # Single / Batch Evaluation
    # -------------------------

    def score_rag_result_once(
        self,
        query: str,
        result: Dict[str, Any],
        ground_truth: Dict[str, Any],
        k: int = 5,
    ) -> Dict[str, float]:
        retrieved_docs = (
            result.get("retrieved_docs")
            or result.get("docs")
            or result.get("retrieval_top_k")
            or result.get("reranked_docs")
            or []
        )
        reranked_docs = result.get("reranked_docs") or retrieved_docs
        standard_answer = ground_truth.get("answer", "")
        evaluation_docs = (reranked_docs or retrieved_docs)[:k]

        evaluation_chunks = "\n\n".join(
            f"Chunk {index}:\n{self._get_doc_text(doc)}"
            for index, doc in enumerate(evaluation_docs, start=1)
            if self._get_doc_text(doc)
        )

        prompt = f"""
You are an evaluator for a Retrieval-Augmented Generation (RAG) system.

Evaluate the following question once and return three scores:

1. Answer-Supported Recall@{k}
2. Precision@{k}
3. Groundedness

Please return the result in valid JSON format:

{{
  "answer_supported_recall": 0.0,
  "precision_scores": [1, 0, 1],
  "groundedness": 0.0
}}

Keep the output concise and in JSON format only.

---------------------
Answer-Supported Recall:
- Measures how much of the standard answer is supported by the retrieved chunks.
- Partial coverage is acceptable.

Scoring guideline:
- 1.0 → all key points clearly supported
- 0.7–0.9 → most key points supported
- 0.4–0.6 → partially supported
- 0.1–0.3 → very limited support
- 0.0 → no meaningful support

IMPORTANT:
- Do NOT require exact wording
- Paraphrased or semantically equivalent content SHOULD be counted as support
- Information can be distributed across multiple chunks
- If a key idea is implied or clearly related, count it as partial support (0.5)
- Do NOT be overly strict when judging support

---------------------
Precision:
- Evaluate each chunk independently.
- A chunk is useful (1) if it contains clear and specific answer-related information.
- Mark 0 if it is only general, vague, or loosely related.
- Return exactly {k} scores.
- If uncertain, prefer 0.

---------------------
Groundedness:
- Measures whether the RAG answer is supported by the retrieved chunks.
- Unsupported or hallucinated claims should reduce the score.
- Missing information does NOT reduce groundedness.
- If the answer correctly refuses due to insufficient context and does not hallucinate, groundedness should be high (around 0.85–1.0).

--------------------------------------------------
Example 1:

Question: What are the AI RMF core functions?
Standard Answer: GOVERN, MAP, MEASURE, MANAGE

Chunks:
Chunk 1: lists GOVERN, MAP
Chunk 2: describes general risk management ideas
Chunk 3: lists MEASURE and MANAGE

RAG Answer:
Mentions all four functions correctly, but only provides specific details for GOVERN and MEASURE, and vague content for MAP and MANAGE.

Output:
{{
  "answer_supported_recall": 0.8,
  "precision_scores": [1, 0, 1],
  "groundedness": 0.8
}}

--------------------------------------------------
Example 2:

Question: Compare UNESCO cultural diversity and ASEAN local context approaches.

Standard Answer:
UNESCO emphasizes cultural diversity, multilingualism, and local knowledge;
ASEAN emphasizes local context, regional implementation, and practical governance.

Chunks:
Chunk 1: describes ASEAN local context and regional implementation
Chunk 2: describes ASEAN national-level recommendations
Chunk 3: unrelated general AI governance content

RAG Answer:
The context does not provide enough information to compare both documents because UNESCO cultural diversity content is missing.

Output:
{{
  "answer_supported_recall": 0.5,
  "precision_scores": [1, 1, 0],
  "groundedness": 0.95
}}

--------------------------------------------------
Question:
{query}

Standard Answer:
{standard_answer}

Top-{k} Chunks:
{evaluation_chunks}

RAG Answer:
{result.get("answer", "")}

Return JSON only:
"""
        defaults = {
            "answer_supported_recall": 0.0,
            "precision": 0.0,
            "precision_scores": [0] * len(evaluation_docs),
            "groundedness": 0.0,
        }

        try:
            response = complete_with_retry(self.llm, prompt)
            parsed = self._extract_json_object(response.text)
        except Exception:
            return defaults

        precision_scores = self._coerce_precision_scores(
            parsed.get("precision_scores"),
            expected_count=len(evaluation_docs),
        )
        precision = (
            round(sum(precision_scores) / len(precision_scores), 2)
            if precision_scores
            else 0.0
        )

        return {
            "answer_supported_recall": self._coerce_score(
                parsed.get("answer_supported_recall"),
                default=0.0,
            ),
            "precision": precision,
            "precision_scores": precision_scores,
            "groundedness": self._coerce_score(parsed.get("groundedness"), default=0.0),
        }

    def evaluate_single(
        self,
        query: str,
        result: Dict[str, Any],
        ground_truth: Dict[str, Any],
        k: int = 5,
    ) -> Dict[str, Any]:
        retrieved_docs = (
            result.get("retrieved_docs")
            or result.get("docs")
            or result.get("retrieval_top_k")
            or result.get("reranked_docs")
            or []
        )

        page_recall = self.page_recall_at_k(
            docs=retrieved_docs,
            ground_truth=ground_truth,
            k=k,
        )
        llm_scores = self.score_rag_result_once(
            query=query,
            result=result,
            ground_truth=ground_truth,
            k=k,
        )

        return {
            "query": query,
            f"Page Recall@{k}": page_recall,
            f"Answer-Supported Recall@{k}": llm_scores["answer_supported_recall"],
            f"Precision@{k}": llm_scores["precision"],
            f"Precision Scores@{k}": llm_scores["precision_scores"],
            "Groundedness": llm_scores["groundedness"],
        }

    def evaluate_batch(
        self,
        dataset: List[Dict[str, Any]],
        results: List[Dict[str, Any]],
        k: int = 5,
    ) -> Dict[str, Any]:
        if len(dataset) != len(results):
            raise ValueError("dataset and results must have the same length.")

        if not dataset:
            return {
                f"Page Recall@{k}": 0.0,
                f"Answer-Supported Recall@{k}": 0.0,
                f"Precision@{k}": 0.0,
                "Groundedness": 0.0,
                "details": [],
            }

        detailed_results = []

        for data, result in zip(dataset, results):
            detailed_results.append(
                self.evaluate_single(
                    query=data["question"],
                    result=result,
                    ground_truth=data,
                    k=k,
                )
            )

        n = len(detailed_results)

        return {
            f"Page Recall@{k}": sum(item[f"Page Recall@{k}"] for item in detailed_results) / n,
            f"Answer-Supported Recall@{k}": (
                sum(item[f"Answer-Supported Recall@{k}"] for item in detailed_results) / n
            ),
            f"Precision@{k}": sum(item[f"Precision@{k}"] for item in detailed_results) / n,
            "Groundedness": sum(item["Groundedness"] for item in detailed_results) / n,
            "details": detailed_results,
        }
