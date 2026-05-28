"""
Multi-format document ingestion. Auto-detects file type and routes
to the appropriate extractor.
"""

from pathlib import Path

from pdf_extractor import PdfExtractor
from pdf_extractor.models import StructuredDocument

from .extractors import CsvExtractor, DocxExtractor, HtmlExtractor, MarkdownExtractor

# Map extensions to extractor types
_EXTENSION_MAP: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "markdown",
    ".md": "markdown",
    ".html": "html",
    ".htm": "html",
    ".csv": "csv",
}

SUPPORTED_EXTENSIONS = set(_EXTENSION_MAP.keys())


class DocumentIngester:
    """
    Unified document ingestion. Accepts any supported file format,
    auto-detects type from extension, and returns a StructuredDocument.
    """

    def __init__(self):
        self._pdf = PdfExtractor()
        self._docx = DocxExtractor()
        self._markdown = MarkdownExtractor()
        self._html = HtmlExtractor()
        self._csv = CsvExtractor()

    def ingest(self, file_path: str) -> StructuredDocument:
        """Ingest a file and return a StructuredDocument."""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        kind = _EXTENSION_MAP.get(ext)

        if kind is None:
            raise ValueError(
                f"Unsupported file format: '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        if kind == "pdf":
            return self._pdf.extract(file_path)
        elif kind == "docx":
            return self._docx.extract(file_path)
        elif kind == "markdown":
            return self._markdown.extract(file_path)
        elif kind == "html":
            return self._html.extract(file_path)
        elif kind == "csv":
            return self._csv.extract(file_path)
        else:
            raise ValueError(f"No extractor for kind: {kind}")

    @staticmethod
    def supports(file_path: str) -> bool:
        """Check if a file format is supported."""
        ext = Path(file_path).suffix.lower()
        return ext in SUPPORTED_EXTENSIONS
