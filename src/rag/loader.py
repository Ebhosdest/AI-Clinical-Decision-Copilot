"""Document loader and section-aware chunker for clinical guidelines."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import PyPDF2
    _PDF_AVAILABLE = True
except ImportError:
    _PDF_AVAILABLE = False


@dataclass
class DocumentChunk:
    text: str
    source: str
    document_id: str
    section: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def _extract_header(text: str, key: str) -> str:
    """Extract a header value like 'SOURCE: value' from document text."""
    match = re.search(rf"^{key}:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_pdf(path: Path) -> str:
    if not _PDF_AVAILABLE:
        raise ImportError("PyPDF2 required for PDF loading: pip install PyPDF2")
    text_parts = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Return list of (section_name, section_text) pairs."""
    pattern = re.compile(r"^SECTION:\s*(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        return [("General", text)]

    sections = []
    for i, match in enumerate(matches):
        section_name = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections.append((section_name, section_text))
    return sections


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping character-level chunks."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        # Attempt to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind(". ")
            if last_period > chunk_size // 2:
                end = start + last_period + 1
                chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
    return [c for c in chunks if c]


def load_documents(
    guidelines_dir: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[DocumentChunk]:
    """Load all .txt and .pdf files from guidelines_dir, return chunked documents."""
    directory = Path(guidelines_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Guidelines directory not found: {guidelines_dir}")

    all_chunks: list[DocumentChunk] = []
    files = list(directory.glob("*.txt")) + list(directory.glob("*.pdf"))

    for file_path in sorted(files):
        raw_text = _load_pdf(file_path) if file_path.suffix == ".pdf" else _load_txt(file_path)
        source = _extract_header(raw_text, "SOURCE") or file_path.stem
        doc_id = _extract_header(raw_text, "DOCUMENT_ID") or file_path.stem

        sections = _split_into_sections(raw_text)
        chunk_index = 0
        for section_name, section_text in sections:
            for chunk_text in _chunk_text(section_text, chunk_size, overlap):
                all_chunks.append(
                    DocumentChunk(
                        text=chunk_text,
                        source=source,
                        document_id=doc_id,
                        section=section_name,
                        chunk_index=chunk_index,
                        metadata={"file": file_path.name},
                    )
                )
                chunk_index += 1

    return all_chunks
