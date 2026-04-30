# Preprocessing module
# Handles PDF parsing, text cleaning, caching, and document preparation

import os
import re
import json
from llama_index.core import Document


def log_step(message: str) -> None:
    print(f"\n[Progress] {message}")


def clean_text(text: str) -> str:
    """
    Conservatively clean PDF text.

    Only removes page headers/footers, printed page numbers, table of contents
    entries, and reference sections. Copyright text is kept.
    """
    lines = text.split("\n")
    cleaned_lines = []

    reference_heading_pattern = re.compile(
        r"^\s*(references|bibliography|works cited)\s*$",
        re.IGNORECASE
    )

    toc_heading_pattern = re.compile(
        r"^\s*(table of contents|contents)\s*$",
        re.IGNORECASE
    )

    toc_line_pattern = re.compile(
        r"^.{3,120}\s+\.{2,}\s*\d+$|^.{3,120}\s+\d+$"
    )

    header_footer_patterns = [
        re.compile(r"^NIST AI 100-1\s*(AI RMF 1\.0)?$", re.IGNORECASE),
        re.compile(r"^AI RMF 1\.0$", re.IGNORECASE),
        re.compile(r"^Page\s+\d+$", re.IGNORECASE),
        re.compile(r"^\s*[-–—]?\s*\d+\s*[-–—]?\s*$"),
        re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),
        re.compile(r"^\d+$"),
    ]

    in_toc = False

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            cleaned_lines.append("")
            continue

        if any(pattern.match(line) for pattern in header_footer_patterns):
            continue

        if toc_heading_pattern.match(line):
            in_toc = True
            continue

        if in_toc:
            if toc_line_pattern.match(line):
                continue
            else:
                in_toc = False

        if reference_heading_pattern.match(line):
            break

        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"(?<=[A-Za-z])-\n(?=[a-z])", "", cleaned_text)
    cleaned_text = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned_text)
    cleaned_text = re.sub(r"[\u00a0\u2000-\u200b\u202f\u205f\u3000]", " ", cleaned_text)
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)

    return cleaned_text.strip()


def extract_page_text_by_position(page, y_tolerance: float = 3.0) -> str:
    """
    Extract page text by reading blocks from top to bottom, then left to right.
    """
    blocks = []

    for block in page.get_text("blocks"):
        if len(block) < 5:
            continue

        if len(block) > 6 and block[6] != 0:
            continue

        x0, y0, x1, y1, text = block[:5]
        text = str(text).strip()

        if not text:
            continue

        row_key = round(y0 / y_tolerance) * y_tolerance
        blocks.append((row_key, x0, y0, x1, y1, text))

    blocks.sort(key=lambda item: (item[0], item[1], item[2]))

    return "\n".join(block[-1] for block in blocks).strip()


def natural_sort_key(file_name: str) -> list:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", file_name)
    ]


def load_pdf_documents_with_pymupdf(
    pdf_dir: str,
) -> list[Document]:
    """
    Load PDF documents using PyMuPDF.

    Main logic:
    - Extract each page
    - Clean page text
    - Concatenate all pages from the same PDF into one Document
    - Store page spans in metadata for downstream chunk page tracking
    """
    try:
        import fitz
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required for PDF loading. Install it with: pip install pymupdf"
        ) from exc

    documents: list[Document] = []

    pdf_files = sorted(
        (
            file_name
            for file_name in os.listdir(pdf_dir)
            if file_name.lower().endswith(".pdf")
        ),
        key=natural_sort_key,
    )

    for file_name in pdf_files:
        file_path = os.path.join(pdf_dir, file_name)
        log_step(f"Processing PDF: {file_name}")

        pdf = fitz.open(file_path)

        full_text_parts = []
        page_spans = []
        page_numbers = []

        for page_number, page in enumerate(pdf, start=1):
            page_text = extract_page_text_by_position(page)
            page_text = clean_text(page_text)

            if page_text:
                start_offset = sum(len(part) for part in full_text_parts)
                if full_text_parts:
                    start_offset += 2 * len(full_text_parts)

                full_text_parts.append(page_text)
                end_offset = start_offset + len(page_text)
                page_spans.append(
                    {
                        "page_number": page_number,
                        "start": start_offset,
                        "end": end_offset,
                    }
                )
                page_numbers.append(page_number)

        pdf.close()

        full_text = "\n\n".join(full_text_parts).strip()

        if full_text:
            documents.append(
                Document(
                    text=full_text,
                    metadata={
                        "file_name": file_name,
                        "source": file_path,
                        "page_numbers": page_numbers,
                        "page_spans": page_spans,
                    },
                )
            )

    log_step(f"Loaded {len(documents)} full-document PDFs from {len(pdf_files)} PDF files.")
    return documents


def save_preprocessed_documents(documents: list[Document], cache_path: str) -> None:
    cache_data = [
        {
            "text": document.text,
            "metadata": document.metadata,
        }
        for document in documents
    ]

    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    with open(cache_path, "w", encoding="utf-8") as file:
        json.dump(cache_data, file, ensure_ascii=False, indent=2)

    log_step(f"Saved preprocessed PDF cache: {cache_path}")


def load_preprocessed_documents(cache_path: str) -> list[Document]:
    with open(cache_path, "r", encoding="utf-8") as file:
        cache_data = json.load(file)

    documents = [
        Document(
            text=item["text"],
            metadata=item.get("metadata", {}),
        )
        for item in cache_data
    ]

    missing_page_spans = [
        document.metadata.get("file_name", "unknown_file")
        for document in documents
        if not document.metadata.get("page_spans")
    ]

    if missing_page_spans:
        raise ValueError(
            "Preprocessed cache is missing page_spans. "
            "Please rerun with --reprocess-pdf. "
            f"First affected file: {missing_page_spans[0]}"
        )

    log_step(f"Loaded {len(documents)} preprocessed documents from cache: {cache_path}")
    return documents


def load_or_create_preprocessed_documents(
    pdf_dir: str,
    cache_path: str = "preprocessed_documents.json",
    reprocess_pdf: bool = False,
) -> list[Document]:
    if not reprocess_pdf and os.path.exists(cache_path):
        return load_preprocessed_documents(cache_path)

    documents = load_pdf_documents_with_pymupdf(
        pdf_dir=pdf_dir,
    )
    save_preprocessed_documents(documents, cache_path)
    return documents
