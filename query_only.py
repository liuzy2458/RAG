Your_ZHIPU_API_key = ""
Your_DEEPSEEK_API_key = ""

import os

import chromadb
from openai import OpenAI as DeepSeekClient
from llama_index.core import PromptTemplate, Settings, VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.zhipuai import ZhipuAI
from llama_index.vector_stores.chroma import ChromaVectorStore


def log_step(message: str) -> None:
    print(f"\n[Progress] {message}")


def build_llm_only_prompt(question: str) -> str:
    return (
        "Do not use web search or any external tools. "
        "Answer only based on your internal knowledge.\n\n"
        f"Question: {question}"
    )


def build_evaluation_prompt(question: str, llm_only_answer: str, rag_answer: str) -> str:
    return (
        "You are an evaluator comparing two answers to the same question.\n"
        "Judge only from the information shown to you.\n"
        "Use these criteria: relevance, factual consistency, specificity, and completeness.\n"
        "Do not use web search or any external tools.\n"
        "Be concise and output exactly in this format:\n"
        "Winner: <LLM Only / RAG / Tie>\n"
        "Reason: <one short paragraph>\n"
        "Evaluator: This judgment was generated automatically by the DeepSeek deepseek-chat model.\n\n"
        f"Question: {question}\n\n"
        f"LLM Only Answer:\n{llm_only_answer}\n\n"
        f"RAG Answer:\n{rag_answer}\n"
    )


RAG_QA_PROMPT = PromptTemplate(
    "Answer based only on the provided context."
    "Prioritize information that is explicitly stated in the context."
    "If relevant information appears in multiple parts of the context, combine it into a complete answer."
    "Be concise and well-structured."
    "Do not introduce information that is not supported by the context."
    "Be honest and admit ot knowing if the context does not contain enough information to answer the question"
    "Provide a clear and complete answer."
    
    "Context:\n"
    "{context_str}\n\n"
    "Question: {query_str}\n\n"
    "Answer:"
)

def parse_winner(evaluation_text: str) -> str:
    text = evaluation_text.strip().lower()
    for line in text.splitlines():
        if line.startswith("winner:"):
            winner = line.split(":", 1)[1].strip().lower()
            if "rag" in winner:
                return "RAG"
            if "llm only" in winner:
                return "LLM Only"
            if "tie" in winner:
                return "Tie"
    return "Unknown"


def complete_with_deepseek(client: DeepSeekClient, prompt: str) -> str:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content or ""


def print_retrieved_chunks(rag_response) -> None:
    print(f"\nTop {len(rag_response.source_nodes)} Retrieved Chunks:")
    for idx, source_node in enumerate(rag_response.source_nodes, start=1):
        score = source_node.score
        text = source_node.node.get_content().strip().replace("\r\n", "\n")
        preview = text[:800]
        if len(text) > 800:
            preview += "..."

        print(f"\n[{idx}] Score: {score:.4f}" if score is not None else f"\n[{idx}] Score: N/A")
        print(preview)


CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "rag_demo"

TOP_K = 15
RERANK_TOP_N = 6
EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
EVALUATOR_NAME = "DeepSeek deepseek-chat acting as an automatic judge"

# Update the questions here before running the script.
QUESTIONS = [
    "According to the UNESCO recommendation, what are the core values that should guide the development and use of AI systems?"
]

results_summary = {
    "RAG": 0,
    "LLM Only": 0,
    "Tie": 0,
    "Unknown": 0,
}


log_step("Initializing LLM and query-time embedding model")

llm = ZhipuAI(
    model="glm-4-flash",
    api_key=os.getenv("ZHIPUAI_API_KEY", "Your_ZHIPU_API_key"),
)

evaluator_client = DeepSeekClient(
    api_key=os.getenv("DEEPSEEK_API_KEY", "Your_DEEPSEEK_API_key"),
    base_url="https://api.deepseek.com",
)

# This embedding model is used only to encode user questions for retrieval.
embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

Settings.llm = llm
Settings.embed_model = embed_model

log_step("Connecting to existing Chroma vector store")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
existing_collections = {collection.name for collection in chroma_client.list_collections()}
if COLLECTION_NAME not in existing_collections:
    raise ValueError(
        "The Chroma collection does not exist. Please run rag.py first to build the RAG knowledge base."
    )

chroma_collection = chroma_client.get_collection(COLLECTION_NAME)

if chroma_collection.count() == 0:
    raise ValueError(
        "The Chroma collection is empty. Please run rag.py first to build the RAG knowledge base."
    )

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

log_step("Loading existing vector index")
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=embed_model,
)

log_step("Initializing reranker")
reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L6-v2",
    top_n=RERANK_TOP_N,
)

log_step("Creating query engine")
query_engine = index.as_query_engine(
    similarity_top_k=TOP_K,
    node_postprocessors=[reranker],
    text_qa_template=RAG_QA_PROMPT,
)


def compare_answers(question: str) -> str:
    log_step(f"Starting comparison for question: {question}")
    llm_only_response = llm.complete(build_llm_only_prompt(question))
    log_step("Finished LLM-only generation")
    rag_response = query_engine.query(question)
    log_step("Finished RAG generation")
    log_step("Starting automatic evaluation")
    evaluation = complete_with_deepseek(
        evaluator_client,
        build_evaluation_prompt(
            question,
            str(llm_only_response),
            str(rag_response),
        )
    )
    log_step("Finished automatic evaluation")

    print("\nQuestion:", question)
    print("\nLLM Only Answer:")
    print(llm_only_response)
    print("\nRAG Answer:")
    print(rag_response)
    print_retrieved_chunks(rag_response)
    print("\nAutomatic Evaluation:")
    print(evaluation)
    print(f"\nEvaluator Note: {EVALUATOR_NAME}. This is an automatic model judgment, not a human review.")
    print("\n" + "=" * 80)
    return parse_winner(str(evaluation))


log_step(f"Running {len(QUESTIONS)} questions")

for question in QUESTIONS:
    winner = compare_answers(question)
    results_summary[winner] = results_summary.get(winner, 0) + 1

log_step("All questions finished")

print("\n=== Summary ===")
print(f"Total Questions: {len(QUESTIONS)}")
print(f"RAG Wins: {results_summary['RAG']}")
print(f"LLM Only Wins: {results_summary['LLM Only']}")
print(f"Ties: {results_summary['Tie']}")

if results_summary["Unknown"] > 0:
    print(f"Unknown Judgments: {results_summary['Unknown']}")
