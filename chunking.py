# Chunking module
# Provides fixed-size chunking, simple sentence-boundary chunking, and embedding

import hashlib
import json
import os
import re

import chromadb
from llama_index.core import Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def log_step(message: str) -> None:
    print(f"\n[Progress] {message}")


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    # Remove incomplete text at the beginning (not starting with capital letter or number)
    text = re.sub(r'^[^A-Z0-9]*', '', text)
    return text


def build_page_metadata(page_numbers: list[int]) -> dict:
    unique_pages = sorted(set(page_numbers))

    if not unique_pages:
        return {}

    return {
        "page_number": unique_pages[0],
        "page_start": unique_pages[0],
        "page_end": unique_pages[-1],
    }


def page_numbers_for_span(metadata: dict, start: int, end: int) -> list[int]:
    page_numbers = []

    for span in metadata.get("page_spans", []) or []:
        try:
            page_number = int(span["page_number"])
            page_start = int(span["start"])
            page_end = int(span["end"])
        except (KeyError, TypeError, ValueError):
            continue

        if page_start < end and start < page_end:
            page_numbers.append(page_number)

    return page_numbers


def sanitize_metadata_for_chroma(metadata: dict) -> dict:
    sanitized = {}

    for key, value in metadata.items():
        if value is None:
            continue

        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, (list, tuple, set)):
            sanitized[key] = ",".join(str(item) for item in value)
        else:
            sanitized[key] = str(value)

    return sanitized


def create_chunk_id(file_name: str, strategy: str, chunk_index: int, chunk_text: str) -> str:
    raw = f"{file_name}_{strategy}_{chunk_index}_{chunk_text[:80]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def save_chunk_cache(chunks: list[Document], cache_path: str) -> None:
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    cache_data = [
        {
            "text": chunk.text,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]

    with open(cache_path, "w", encoding="utf-8") as file:
        json.dump(cache_data, file, ensure_ascii=False, indent=2)

    log_step(f"Saved chunk cache: {cache_path}, chunks={len(chunks)}")


def load_chunk_cache(cache_path: str) -> list[Document]:
    with open(cache_path, "r", encoding="utf-8") as file:
        cache_data = json.load(file)

    chunks = [
        Document(
            text=item["text"],
            metadata=item.get("metadata", {}),
        )
        for item in cache_data
    ]

    log_step(f"Loaded chunks from cache: {cache_path}, chunks={len(chunks)}")
    return chunks


def fixed_size_chunking(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[Document]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunked_documents = []
    strategy = "fixed_size"

    for doc in documents:
        text = doc.text
        file_name = doc.metadata.get("file_name", "unknown_file")

        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                page_numbers = page_numbers_for_span(doc.metadata, start, end)

                metadata = doc.metadata.copy()
                metadata.pop("page_spans", None)
                metadata.pop("page_numbers", None)
                metadata.update(
                    {
                        "chunk_id": create_chunk_id(file_name, strategy, chunk_index, chunk_text),
                        "chunking_strategy": strategy,
                        "chunk_index": chunk_index,
                    }
                )
                metadata.update(build_page_metadata(page_numbers))

                chunked_documents.append(Document(text=chunk_text, metadata=metadata))
                chunk_index += 1

            start += chunk_size - chunk_overlap

    log_step(f"Fixed-size chunking generated {len(chunked_documents)} chunks.")
    return chunked_documents


def load_spacy_model():
    try:
        import spacy
    except ImportError as exc:
        raise ImportError("spaCy is required. Install with: pip install spacy") from exc

    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:
        raise OSError(
            "spaCy model 'en_core_web_sm' is not installed. "
            "Please run: python -m spacy download en_core_web_sm"
        ) from exc


def split_text_with_page_tracking(text: str, metadata: dict, nlp) -> list[dict]:
    units = []
    parsed_doc = nlp(text)

    for sent in parsed_doc.sents:
        sentence = sent.text.strip()

        if not sentence:
            continue

        sentence_start = sent.start_char
        sentence_end = sent.end_char
        units.append(
            {
                "text": sentence,
                "page_numbers": page_numbers_for_span(
                    metadata,
                    sentence_start,
                    sentence_end,
                ),
            }
        )

    return units


def sentence_boundary_chunking(
    documents: list[Document],
    min_chunk_size: int = 250,
    max_chunk_size: int = 800,
    sentence_overlap: int = 1,
    include_context_prefix: bool = False,
    use_semantic_split: bool = False,
    semantic_threshold: float = 0.45,
    semantic_model_name: str = "BAAI/bge-large-en-v1.5",
    semantic_model=None,
) -> list[Document]:
    """
    Simple sentence-boundary chunking.

    Compatibility-only parameters are ignored:
    include_context_prefix, use_semantic_split, semantic_threshold,
    semantic_model_name, and semantic_model.
    """
    nlp = load_spacy_model()
    chunked_documents = []
    strategy = "sentence_boundary"
    global_chunk_index = 0

    for doc in documents:
        file_name = doc.metadata.get("file_name", "unknown_file")
        doc_chunk_start_index = len(chunked_documents)
        sentence_units = split_text_with_page_tracking(doc.text, doc.metadata, nlp)

        if not sentence_units:
            text = normalize_text(doc.text)
            if not text:
                continue
            sentence_units = [{"text": text, "page_numbers": []}]

        current_units = []
        local_chunk_index = 0

        def build_chunk(units: list[dict]):
            chunk_text = " ".join(unit["text"] for unit in units).strip()
            page_numbers = []
            for unit in units:
                page_numbers.extend(unit.get("page_numbers", []))
            page_metadata = build_page_metadata(page_numbers)
            return chunk_text, page_metadata

        def save_chunk(units: list[dict]):
            nonlocal global_chunk_index, local_chunk_index

            if not units:
                return

            chunk_text, page_metadata = build_chunk(units)

            metadata = doc.metadata.copy()
            metadata.pop("page_spans", None)
            metadata.pop("page_numbers", None)
            metadata.update(
                {
                    "chunk_id": create_chunk_id(
                        file_name=file_name,
                        strategy=strategy,
                        chunk_index=global_chunk_index,
                        chunk_text=chunk_text,
                    ),
                    "chunking_strategy": strategy,
                    "chunk_index": global_chunk_index,
                    "document_chunk_index": local_chunk_index,
                    "min_chunk_size": min_chunk_size,
                    "max_chunk_size": max_chunk_size,
                }
            )
            metadata.update(page_metadata)

            chunked_documents.append(Document(text=chunk_text, metadata=metadata))
            global_chunk_index += 1
            local_chunk_index += 1

        def merge_to_previous_chunk(units: list[dict]) -> bool:
            if not units:
                return True

            if len(chunked_documents) <= doc_chunk_start_index:
                return False

            short_text, page_metadata = build_chunk(units)
            previous_doc = chunked_documents[-1]
            merged_text = previous_doc.text.rstrip() + " " + short_text
            merged_metadata = previous_doc.metadata.copy()
            merged_metadata["merged_short_tail"] = True
            merged_metadata["merged_short_tail_chars"] = len(short_text)

            merged_page_values = []

            for key in ("page_start", "page_end", "page_number"):
                if merged_metadata.get(key) is not None:
                    merged_page_values.append(int(merged_metadata[key]))

            for key in ("page_start", "page_end", "page_number"):
                if page_metadata.get(key) is not None:
                    merged_page_values.append(int(page_metadata[key]))

            merged_metadata.update(build_page_metadata(merged_page_values))
            chunked_documents[-1] = Document(text=merged_text, metadata=merged_metadata)
            return True

        for unit in sentence_units:
            if len(unit["text"]) > max_chunk_size:
                if current_units:
                    save_chunk(current_units)

                save_chunk([unit])
                current_units = []
                continue

            candidate_units = current_units + [unit]
            candidate_text, _ = build_chunk(candidate_units)
            current_body = " ".join(u["text"] for u in current_units).strip()
            current_len = len(current_body)

            if len(candidate_text) > max_chunk_size:
                if current_units and current_len >= min_chunk_size:
                    save_chunk(current_units)
                    overlap_units = current_units[-sentence_overlap:] if sentence_overlap > 0 else []
                    current_units = overlap_units + [unit]
                else:
                    current_units.append(unit)
            else:
                current_units.append(unit)

        if current_units:
            final_text, _ = build_chunk(current_units)

            if len(final_text) >= min_chunk_size:
                save_chunk(current_units)
            else:
                merged = merge_to_previous_chunk(current_units)

                if not merged:
                    save_chunk(current_units)

    log_step(f"Sentence-boundary chunking generated {len(chunked_documents)} chunks.")
    return chunked_documents


def build_vector_index(
    documents: list[Document],
    collection_name: str,
    chunking_strategy: str | None = None,
    embed_model_name: str = "BAAI/bge-large-en-v1.5",
    chroma_dir: str = "./chroma_db",
    reset_collection: bool = True,
):
    if (
        chunking_strategy
        and ("/" in chunking_strategy or "\\" in chunking_strategy)
        and (embed_model_name.startswith(".") or embed_model_name.startswith("/") or "\\" in embed_model_name)
    ):
        chroma_dir = embed_model_name
        embed_model_name = chunking_strategy
        chunking_strategy = None

    embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
    Settings.embed_model = embed_model

    final_collection_name = (
        f"{collection_name}_{chunking_strategy}"
        if chunking_strategy
        else collection_name
    )

    log_step(f"Connecting to Chroma at {chroma_dir}")
    chroma_client = chromadb.PersistentClient(path=chroma_dir)

    existing_collections = {
        collection.name if hasattr(collection, "name") else str(collection)
        for collection in chroma_client.list_collections()
    }

    if reset_collection and final_collection_name in existing_collections:
        chroma_client.delete_collection(final_collection_name)
        log_step(f"Deleted existing collection: {final_collection_name}")

    if reset_collection:
        chroma_collection = chroma_client.create_collection(final_collection_name)
    else:
        chroma_collection = chroma_client.get_or_create_collection(final_collection_name)
    log_step(f"Using collection: {final_collection_name}")

    texts = [document.text or " " for document in documents]
    metadatas = [
        sanitize_metadata_for_chroma(document.metadata or {})
        for document in documents
    ]
    ids = []
    seen_ids = set()

    for index, document in enumerate(documents):
        metadata = document.metadata or {}
        raw_id = metadata.get("chunk_id") or create_chunk_id(
            file_name=str(metadata.get("file_name", "unknown_file")),
            strategy=str(metadata.get("chunking_strategy", "chunk")),
            chunk_index=index,
            chunk_text=document.text or "",
        )
        chunk_id = str(raw_id)

        if chunk_id in seen_ids:
            chunk_id = f"{chunk_id}_{index}"

        ids.append(chunk_id)
        seen_ids.add(chunk_id)

    if not (len(texts) == len(metadatas) == len(ids) == len(documents)):
        raise ValueError(
            "Chroma add payload length mismatch: "
            f"chunks={len(documents)}, documents={len(texts)}, "
            f"metadatas={len(metadatas)}, ids={len(ids)}"
        )

    log_step(
        "Building embeddings and vector index "
        f"(documents={len(texts)}, metadatas={len(metadatas)}, ids={len(ids)})"
    )

    batch_size = 128

    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]
        batch_embeddings = embed_model.get_text_embedding_batch(
            batch_texts,
            show_progress=False,
        )

        chroma_collection.add(
            ids=ids[start:end],
            documents=batch_texts,
            metadatas=metadatas[start:end],
            embeddings=batch_embeddings,
        )

    vector_count = chroma_collection.count()
    log_step(f"Stored {vector_count} vectors in collection '{final_collection_name}'")

    if vector_count != len(documents):
        raise RuntimeError(
            f"Chroma vector count mismatch after build: chunks={len(documents)}, "
            f"vectors={vector_count}"
        )

    return chroma_collection
