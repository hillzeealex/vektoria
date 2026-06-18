"""
Extract plain text from an uploaded document, dispatched by file extension.

Everything works from bytes (no temp files). The heavy parsers — PyMuPDF for
PDF, python-docx for DOCX — are imported lazily so the base install stays light;
they ship with the ``vektoria[ingest]`` extra. Plain-text formats (txt/md/html/
csv) need only the standard library.
"""

from __future__ import annotations

import csv
import io
from html.parser import HTMLParser
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown", ".html", ".htm", ".csv"}


def extract_text(data: bytes, filename: str) -> str:
    """Return the document's text content. Raises ValueError on unknown types."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return _from_pdf(data)
    if ext == ".docx":
        return _from_docx(data)
    if ext in {".html", ".htm"}:
        return _from_html(_decode(data))
    if ext == ".csv":
        return _from_csv(_decode(data))
    if ext in {".txt", ".md", ".markdown"}:
        return _decode(data).strip()
    raise ValueError(
        f"Unsupported file type {ext!r}; supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )


def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _from_pdf(data: bytes) -> str:
    import fitz  # PyMuPDF — vektoria[ingest]

    doc = fitz.open(stream=data, filetype="pdf")
    try:
        return "\n\n".join(page.get_text() for page in doc).strip()
    finally:
        doc.close()  # PyMuPDF holds an fd — always release it (legacy bug fix)


def _from_docx(data: bytes) -> str:
    from docx import Document  # python-docx — vektoria[ingest]

    doc = Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip()).strip()


def _from_csv(text: str) -> str:
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return ""
    headers = rows[0]
    lines: list[str] = []
    for row in rows[1:]:
        # Pad short rows so no column is silently dropped (legacy zip() bug).
        cells = row + [""] * (len(headers) - len(row))
        pairs = [f"{h}: {v}" for h, v in zip(headers, cells) if v.strip()]
        if pairs:
            lines.append(" | ".join(pairs))
    return "\n".join(lines)


class _TextHTMLParser(HTMLParser):
    """Collect visible text, skipping <script>/<style> contents."""

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.parts.append(data.strip())


def _from_html(text: str) -> str:
    parser = _TextHTMLParser()
    parser.feed(text)
    return "\n".join(parser.parts)
