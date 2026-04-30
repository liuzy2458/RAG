# query_rewrite.py
# Query rewriting + weighted RRF retrieval module

import os
import json
import re
from typing import List, Dict, Any, Optional
from llama_index.llms.zhipuai import ZhipuAI
from llm import complete_with_retry


QUERY_REWRITE_TEMPERATURE = 0.2


class QueryRewriter:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "glm-4-flash",
    ):
        if api_key is None:
            api_key = os.getenv("ZHIPUAI_API_KEY")

        if not api_key:
            raise ValueError(
                "ZHIPUAI_API_KEY is missing. Pass api_key or set environment variable."
            )

        self.llm = ZhipuAI(
            model=model,
            api_key=api_key,
            temperature=QUERY_REWRITE_TEMPERATURE,
        )

    def _deduplicate_queries(self, queries: List[str]) -> List[str]:
        final_queries = []
        seen = set()

        for q in queries:
            q = q.strip()

            if not q:
                continue

            normalized = " ".join(q.lower().split())

            if normalized not in seen:
                final_queries.append(q)
                seen.add(normalized)

        return final_queries

    def _stopwords(self) -> set[str]:
        return {
            "a",
            "an",
            "and",
            "are",
            "as",
            "by",
            "do",
            "does",
            "for",
            "from",
            "how",
            "in",
            "is",
            "it",
            "of",
            "on",
            "or",
            "should",
            "the",
            "to",
            "under",
            "what",
            "when",
            "where",
            "which",
            "who",
            "why",
            "with",
        }

    def _keyword_fallback_queries(
        self,
        query: str,
        query_type: str,
        num_queries: int,
    ) -> List[str]:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", query)
        keywords = [
            token
            for token in tokens
            if token.lower() not in self._stopwords() and len(token) > 2
        ]
        base = " ".join(keywords[:8]) or query

        expansions_by_type = {
            "comparison": [
                f"{base} comparison contrast criteria",
                f"{base} differences similarities framework",
                f"{base} comparative analysis governance",
            ],
            "policy": [
                f"{base} obligations recommendations principles",
                f"{base} Member States compliance implementation",
                f"{base} policy requirements governance",
            ],
            "structure": [
                f"{base} framework hierarchy components",
                f"{base} functions categories lifecycle",
                f"{base} process structure architecture",
            ],
            "definition": [
                f"{base} definition meaning types",
                f"{base} categories taxonomy examples",
                f"{base} terminology concept overview",
            ],
            "general_rag": [
                f"{base} evidence source context",
                f"{base} AI governance ethics framework",
                f"{base} policy principles requirements",
            ],
        }

        return expansions_by_type.get(query_type, expansions_by_type["general_rag"])[:num_queries]

    def _extract_json_object(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise

            return json.loads(match.group(0))

    def generate_query_plan(self, query: str, num_queries: int = 3) -> Dict[str, Any]:
        query = query.strip()

        if not query:
            return {
                "query_type": "general_chat",
                "queries": [],
            }

        prompt = f"""
You are a query planning assistant for a RAG retrieval system over AI governance and AI ethics documents.

Your task is to:
1. Classify the user query.
2. Generate retrieval-friendly search queries if needed.

Return ONLY valid JSON in this exact shape:
{{
  "query_type": "general_chat | comparison | policy | structure | definition | general_rag",
  "queries": ["...", "...", "..."]
}}

Do not include markdown, code fences, or any extra text. Output pure JSON only.

Query type definitions:
- general_chat: greetings, thanks, jokes, model identity, normal conversation, or anything unrelated to the document collection. No retrieval is needed.
- comparison: comparing two or more institutions, documents, frameworks, policies, regions, countries, or governance approaches.
- policy: asking recommendations, obligations, principles, requirements, what Member States/governments/organizations should do.
- structure: asking about structure, hierarchy, process, framework, functions, components, lifecycle, or categories.
- definition: asking definitions, types, lists, what is / what are.
- general_rag: other questions that need document retrieval.

Requirements:
- Do NOT answer the question.
- Do NOT merely paraphrase the original sentence.

- If query_type is general_chat:
  - Return "queries": [].

- If query_type is NOT general_chat:
  - Generate exactly {num_queries} retrieval-friendly search queries.

GENERAL QUERY RULES:
- Each query MUST be a short keyword-style phrase, not a full sentence.
- Each query should be concise (ideally 5–12 words).
- Do NOT start queries with question words such as "what", "how", "why", "when", "where".
- Avoid duplicate or near-duplicate queries.

KEYWORD EXPANSION:
- Use keyword expansion with entities, document terms, policy terms, section-heading terms, and domain-specific vocabulary.
- Always preserve exact named entities from the original query:
  including organization names, framework names, report names, acronyms, and edition names.
- If a specific document or framework is mentioned, include that exact name in EVERY generated query.

STRUCTURE / LIST QUESTIONS:
- If the query asks for a list (areas, components, functions, principles, etc.):
  - At least ONE query MUST include heading-style terms such as:
    "key areas", "broad areas", "components", "functions", "principles".

TYPE-SPECIFIC RULES:

For comparison:
- Decompose retrieval directions:
  - One query for object A
  - One query for object B
  - One query for comparison/contrast criteria
- Use terms such as: comparison, difference, governance approach, regional vs national.

For policy:
- Include terms such as:
  obligations, recommendations, principles, policy area, Member States, compliance, implementation.

For structure:
- Include terms such as:
  framework, hierarchy, functions, categories, components, lifecycle, process.

For definition:
- Generate short keyword queries and include synonyms or variants when useful.

GOOD QUERY STYLE EXAMPLES:
- NIST AI RMF GOVERN MAP MEASURE MANAGE core functions
- UNESCO AI ethics environment ecosystems carbon footprint energy use
- PDPC Model Artificial Intelligence Governance Framework four broad areas
- ASEAN AI governance human oversight risk categories

User query:
{query}
"""

        try:
            response = complete_with_retry(self.llm, prompt)
            plan = self._extract_json_object(response.text)
        except Exception:
            return {
                "query_type": "general_rag",
                "queries": self._keyword_fallback_queries(query, "general_rag", num_queries),
            }

        valid_types = {
            "general_chat",
            "comparison",
            "policy",
            "structure",
            "definition",
            "general_rag",
        }
        query_type = str(plan.get("query_type", "general_rag")).strip().lower()

        if query_type not in valid_types:
            query_type = "general_rag"

        if query_type == "general_chat":
            return {
                "query_type": query_type,
                "queries": [],
            }

        queries = plan.get("queries", [])
        if not isinstance(queries, list):
            queries = []

        queries = self._deduplicate_queries([str(item) for item in queries])

        for fallback_query in self._keyword_fallback_queries(query, query_type, num_queries):
            if len(queries) >= num_queries:
                break

            queries.append(fallback_query)
            queries = self._deduplicate_queries(queries)

        queries = queries[:num_queries]

        return {
            "query_type": query_type,
            "queries": queries if queries else self._keyword_fallback_queries(query, query_type, num_queries),
        }

    def rewrite_query(self, query: str) -> str:
        plan = self.generate_query_plan(
            query=query,
            num_queries=1,
        )
        queries = plan.get("queries", [])

        return queries[0] if queries else query.strip()


class WeightedRRFRetriever:
    def __init__(
        self,
        retriever,
        rrf_k: int = 60,
        original_weight: float = 1.5,
        rewrite_weight: float = 1.0,
        top_k_per_query: int = 5,
        final_top_k: int = 10,
    ):
        self.retriever = retriever
        self.rrf_k = rrf_k
        self.original_weight = original_weight
        self.rewrite_weight = rewrite_weight
        self.top_k_per_query = top_k_per_query
        self.final_top_k = final_top_k

    def _get_doc_id(self, doc: Any) -> str:
        if isinstance(doc, dict):
            if doc.get("id"):
                return str(doc["id"])

            metadata = doc.get("metadata", {}) or {}
            if metadata.get("chunk_id"):
                return str(metadata["chunk_id"])

            if metadata.get("source") and metadata.get("page_number"):
                return f"{metadata['source']}_{metadata['page_number']}_{hash(doc.get('text', ''))}"

            return str(hash(doc.get("text", str(doc))))

        if hasattr(doc, "id"):
            return str(doc.id)

        if hasattr(doc, "node_id"):
            return str(doc.node_id)

        if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
            if "chunk_id" in doc.metadata:
                return str(doc.metadata["chunk_id"])

            if "source" in doc.metadata and "page" in doc.metadata:
                text = getattr(doc, "text", str(doc))
                return f"{doc.metadata['source']}_{doc.metadata['page']}_{hash(text)}"

        return str(hash(str(doc)))

    def _search(self, query: str) -> List[Any]:
        if hasattr(self.retriever, "search"):
            return self.retriever.search(
                query,
                top_k=self.top_k_per_query,
            )

        if hasattr(self.retriever, "retrieve"):
            try:
                return self.retriever.retrieve(query, top_k=self.top_k_per_query)
            except TypeError:
                return self.retriever.retrieve(query)[: self.top_k_per_query]

        raise ValueError("Retriever must have either search() or retrieve().")

    def retrieve(
        self,
        original_query: str,
        rewritten_queries: List[str],
    ) -> List[Any]:
        original_query = original_query.strip()

        all_queries = [(original_query, self.original_weight)]

        for q in rewritten_queries:
            q = q.strip()

            if q and q != original_query:
                all_queries.append((q, self.rewrite_weight))

        doc_store: Dict[str, Any] = {}
        rrf_scores: Dict[str, float] = {}

        for query, query_weight in all_queries:
            results = self._search(query)

            for rank, doc in enumerate(results, start=1):
                doc_id = self._get_doc_id(doc)

                if doc_id not in doc_store:
                    if isinstance(doc, dict):
                        stored_doc = doc.copy()
                        stored_doc["metadata"] = (doc.get("metadata", {}) or {}).copy()
                        stored_doc["source_scores"] = doc.get("source_scores", {}).copy()
                        stored_doc["retrieval_method"] = "query_rewrite_rrf"
                        doc_store[doc_id] = stored_doc
                    else:
                        doc_store[doc_id] = doc

                rrf_score = query_weight * (1 / (self.rrf_k + rank))
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_score

                if isinstance(doc_store[doc_id], dict):
                    doc_store[doc_id]["source_scores"][query] = doc.get(
                        "retrieval_score", 0.0
                    ) if isinstance(doc, dict) else 0.0

        ranked_doc_ids = sorted(
            rrf_scores.keys(),
            key=lambda doc_id: rrf_scores[doc_id],
            reverse=True,
        )

        final_docs = []

        for doc_id in ranked_doc_ids[: self.final_top_k]:
            doc = doc_store[doc_id]

            if isinstance(doc, dict):
                doc["retrieval_score"] = float(rrf_scores[doc_id])

            final_docs.append(doc)

        return final_docs


class QueryRewriteRRFPipeline:
    def __init__(
        self,
        retriever,
        api_key: Optional[str] = None,
        rewrite_model: str = "glm-4-flash",
        num_rewrites: int = 3,
        rrf_k: int = 60,
        original_weight: float = 1.5,
        rewrite_weight: float = 1.2,
        top_k_per_query: int = 5,
        final_top_k: int = 10,
    ):
        self.query_rewriter = QueryRewriter(
            api_key=api_key,
            model=rewrite_model,
        )

        self.rrf_retriever = WeightedRRFRetriever(
            retriever=retriever,
            rrf_k=rrf_k,
            original_weight=original_weight,
            rewrite_weight=rewrite_weight,
            top_k_per_query=top_k_per_query,
            final_top_k=final_top_k,
        )

        self.num_rewrites = num_rewrites

    def retrieve(self, query: str) -> List[Any]:
        plan = self.query_rewriter.generate_query_plan(
            query=query,
            num_queries=self.num_rewrites,
        )

        if plan["query_type"] == "general_chat":
            return []

        return self.rrf_retriever.retrieve(
            original_query=query,
            rewritten_queries=plan["queries"],
        )

    def retrieve_with_queries(self, query: str) -> Dict[str, Any]:
        plan = self.query_rewriter.generate_query_plan(
            query=query,
            num_queries=self.num_rewrites,
        )

        if plan["query_type"] == "general_chat":
            return {
                "original_query": query,
                "query_type": plan["query_type"],
                "rewritten_queries": [],
                "candidate_docs": [],
            }

        candidate_docs = self.rrf_retriever.retrieve(
            original_query=query,
            rewritten_queries=plan["queries"],
        )

        return {
            "original_query": query,
            "query_type": plan["query_type"],
            "rewritten_queries": plan["queries"],
            "candidate_docs": candidate_docs,
        }




