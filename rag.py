Your_ZHIPU_API_KEY = ""

import os

import chromadb
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.zhipuai import ZhipuAI
from llama_index.vector_stores.chroma import ChromaVectorStore


def log_step(message: str) -> None:
    print(f"\n[Progress] {message}")


def load_pdf_documents_with_pymupdf(pdf_dir: str) -> list[Document]:
    try:
        import fitz
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required for PDF loading. Install it with: "
            "pip install pymupdf"
        ) from exc

    documents: list[Document] = []
    pdf_files = sorted(
        file_name for file_name in os.listdir(pdf_dir) if file_name.lower().endswith(".pdf")
    )

    for file_name in pdf_files:
        file_path = os.path.join(pdf_dir, file_name)
        pdf = fitz.open(file_path)
        page_texts = []

        for page_number, page in enumerate(pdf, start=1):
            page_text = page.get_text().strip()
            if page_text:
                page_texts.append(f"[Page {page_number}]\n{page_text}")

        pdf.close()

        if page_texts:
            documents.append(
                Document(
                    text="\n\n".join(page_texts),
                    metadata={"file_name": file_name, "source": file_path},
                )
            )

    return documents


PDF_DIR = "C:/Users/liuzy/Documents/yr3 sem2/seem2460/project/pdf-testing"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "rag_demo"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 100
EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"


log_step("Loading build configuration")

llm = ZhipuAI(
    model="glm-4-flash",
    api_key=os.getenv("ZHIPUAI_API_KEY", "Your_ZHIPU_API_KEY")
)

embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

Settings.llm = llm
Settings.embed_model = embed_model
Settings.text_splitter = SentenceSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

log_step(f"Reading source documents from {PDF_DIR}")
documents = load_pdf_documents_with_pymupdf(PDF_DIR)
log_step(f"Loaded {len(documents)} documents")

log_step("Connecting to Chroma")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

log_step("Resetting collection to match the current embedding dimension")
existing_collections = {collection.name for collection in chroma_client.list_collections()}
if COLLECTION_NAME in existing_collections:
    chroma_client.delete_collection(COLLECTION_NAME)
    log_step(f"Deleted existing collection: {COLLECTION_NAME}")
else:
    log_step("No previous collection found")

chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
log_step(f"Created fresh collection: {COLLECTION_NAME}")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

log_step("Building chunks, embeddings, and vector index")
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)
log_step("Vector index build finished")

doc_count = len(documents)
vector_count = chroma_collection.count()

print("\n=== Build Summary ===")
print(f"Documents Loaded: {doc_count}")
print(f"Chunk Size: {CHUNK_SIZE}")
print(f"Chunk Overlap: {CHUNK_OVERLAP}")
print(f"Embedding Model: {EMBED_MODEL_NAME}")
print(f"Collection Name: {COLLECTION_NAME}")
print(f"Stored Vector Count: {vector_count}")
print("\nRAG knowledge base is ready. Use query_only.py for QA and evaluation.")
