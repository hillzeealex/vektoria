"""
SwissExtract — PDF to structured Markdown extractor.

Replaces LlamaParse for native-text PDFs (cours de droit, etc.).
Uses PyMuPDF to analyze font sizes, bold flags, and positions to
reconstruct a heading hierarchy and produce clean Markdown.

Two complementary heading-detection strategies:
1. Font analysis — different font sizes/bold → heading levels
2. Numbering analysis — "1." → h1, "1.1." → h2, "1.1.1." → h3
   (kicks in when all headings share the same font style)

Zero external API calls. Everything runs locally.
"""

import re
import fitz  # PyMuPDF
from collections import Counter
from dataclasses import dataclass
from .models import StructuredDocument, Section


@dataclass
class _Span:
    text: str
    size: float
    is_bold: bool
    is_italic: bool
    font: str
    page: int
    y_pos: float
    x_pos: float


@dataclass
class _Line:
    spans: list[_Span]
    page: int
    y_pos: float

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.spans).strip()

    @property
    def max_size(self) -> float:
        return max(s.size for s in self.spans) if self.spans else 0

    @property
    def dominant_size(self) -> float:
        if not self.spans:
            return 0
        return max(self.spans, key=lambda s: len(s.text)).size

    @property
    def is_bold(self) -> bool:
        bold_chars = sum(len(s.text) for s in self.spans if s.is_bold)
        total_chars = sum(len(s.text) for s in self.spans)
        return total_chars > 0 and bold_chars / total_chars > 0.5

    @property
    def is_all_upper(self) -> bool:
        t = self.text
        alpha = [c for c in t if c.isalpha()]
        return len(alpha) > 3 and all(c.isupper() for c in alpha)

    @property
    def alpha_len(self) -> int:
        return sum(1 for c in self.text if c.isalpha())


# Regex for lines that are just section numbers: "1.", "2.4.3.1", "4.2", etc.
_BARE_NUMBER_RE = re.compile(r"^\d+(\.\d+)*\.?\s*$")

# Regex for TOC-style dotted leaders
_TOC_DOTS_RE = re.compile(r"\.{4,}")

# Regex for page-number-at-end pattern: "SOME TITLE ... 42"
_TOC_TRAILING_PAGE_RE = re.compile(r"[.\s]{3,}\d{1,3}\s*$")

# Regex to extract a leading section number: "1.2.3. Some Title" → ("1.2.3", depth=3)
_SECTION_NUM_RE = re.compile(
    r"^(\d+(?:\.\d+)*)\.?\s+"  # "1." or "1.2." or "1.2.3"
)

# Heading patterns common in legal/academic docs
_NAMED_HEADING_RE = re.compile(
    r"^(?:"
    r"(?:DOCUMENT|Document)\s+[Nn]°?\s*\d+|"   # Document n°1
    r"(?:COURS|Cours)\s+[Nn]°?\s*\d+|"          # Cours n°2
    r"(?:CHAPITRE|Chapitre)\s+\d+|"             # Chapitre 1
    r"(?:PARTIE|Partie)\s+\d+|"                 # Partie 1
    r"(?:TITRE|Titre)\s+\d+|"                   # Titre 1
    r"(?:SECTION|Section)\s+\d+|"               # Section 1
    r"§\s*\d+"                                  # § 1
    r")",
    re.IGNORECASE,
)


class PdfExtractor:
    def __init__(self, *, toc_pages_hint: int = 10):
        self.toc_pages_hint = toc_pages_hint

    def extract(self, source: str | bytes) -> StructuredDocument:
        if isinstance(source, str):
            doc = fitz.open(source)
        else:
            doc = fitz.open(stream=source, filetype="pdf")

        lines = self._extract_lines(doc)
        toc_pages = self._detect_toc_pages(lines)
        content_lines = [l for l in lines if l.page not in toc_pages]
        font_heading_map = self._detect_heading_levels_by_font(content_lines)
        use_numbering = self._should_use_numbering(font_heading_map)
        markdown, sections = self._build_markdown(
            content_lines, font_heading_map, use_numbering
        )
        title = self._detect_title(sections, doc)

        return StructuredDocument(
            markdown=markdown,
            page_count=len(doc),
            title=title,
            sections=sections,
            metadata={
                "parser": "swiss-extract",
                "toc_pages": sorted(toc_pages),
                "heading_strategy": "numbering" if use_numbering else "font",
                "font_heading_map": {
                    f"{k[0]}|{'bold' if k[1] else 'normal'}": f"h{v}"
                    for k, v in font_heading_map.items()
                },
            },
        )

    # ── Line extraction ─────────────────────────────────────────────

    def _extract_lines(self, doc: fitz.Document) -> list[_Line]:
        all_lines: list[_Line] = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if "lines" not in block:
                    continue
                for line_data in block["lines"]:
                    spans = []
                    for span in line_data["spans"]:
                        text = span["text"]
                        if not text:
                            continue
                        flags = span["flags"]
                        spans.append(_Span(
                            text=text,
                            size=round(span["size"], 1),
                            is_bold=bool(flags & (1 << 4)),
                            is_italic=bool(flags & (1 << 1)),
                            font=span.get("font", ""),
                            page=page_num,
                            y_pos=round(line_data["bbox"][1], 1),
                            x_pos=round(line_data["bbox"][0], 1),
                        ))

                    if spans:
                        line = _Line(spans=spans, page=page_num,
                                     y_pos=round(line_data["bbox"][1], 1))
                        text = line.text
                        if text and not self._is_page_number(text, page_num):
                            all_lines.append(line)

        return all_lines

    def _is_page_number(self, text: str, page_num: int) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        try:
            num = int(stripped)
            return abs(num - (page_num + 1)) <= 5
        except ValueError:
            return False

    # ── TOC detection ────────────────────────────────────────────────

    def _is_toc_line(self, text: str) -> bool:
        if _TOC_DOTS_RE.search(text):
            return True
        if _TOC_TRAILING_PAGE_RE.search(text):
            return True
        return False

    def _detect_toc_pages(self, lines: list[_Line]) -> set[int]:
        toc_pages: set[int] = set()
        page_line_counts: dict[int, int] = {}
        page_toc_counts: dict[int, int] = {}

        for line in lines:
            if line.page >= self.toc_pages_hint:
                continue
            page_line_counts[line.page] = page_line_counts.get(line.page, 0) + 1
            if self._is_toc_line(line.text):
                page_toc_counts[line.page] = page_toc_counts.get(line.page, 0) + 1

        for page, total in page_line_counts.items():
            toc_count = page_toc_counts.get(page, 0)
            if total > 0 and toc_count / total > 0.3:
                toc_pages.add(page)

        # Title page with very few lines
        if 0 in page_line_counts and page_line_counts[0] < 8:
            toc_pages.add(0)

        return toc_pages

    # ── Heading detection: font-based ────────────────────────────────

    def _detect_heading_levels_by_font(
        self, lines: list[_Line]
    ) -> dict[tuple[float, bool], int]:
        char_counts: Counter[tuple[float, bool]] = Counter()
        for line in lines:
            for span in line.spans:
                clean = span.text.strip()
                if len(clean) > 2:
                    key = (span.size, span.is_bold)
                    char_counts[key] += len(clean)

        if not char_counts:
            return {}

        body_key = char_counts.most_common(1)[0][0]
        body_size = body_key[0]

        heading_candidates: set[tuple[float, bool]] = set()
        for line in lines:
            text = line.text
            if not text or len(text) > 200:
                continue
            if self._is_toc_line(text):
                continue
            if line.alpha_len < 3:
                continue

            size = line.dominant_size
            bold = line.is_bold

            is_heading = (
                size > body_size + 0.5
                or (size >= body_size and bold and not body_key[1])
            )
            if is_heading:
                heading_candidates.add((round(size, 1), bold))

        sorted_candidates = sorted(
            heading_candidates,
            key=lambda k: (k[0], 1 if k[1] else 0),
            reverse=True,
        )

        heading_map: dict[tuple[float, bool], int] = {}
        level = 1

        for i, (size, bold) in enumerate(sorted_candidates):
            if level > 4:
                break
            if i > 0:
                prev_size = sorted_candidates[i - 1][0]
                if abs(size - prev_size) <= 0.3:
                    heading_map[(size, bold)] = heading_map.get(
                        (sorted_candidates[i - 1][0], sorted_candidates[i - 1][1]),
                        level - 1,
                    )
                    continue
            heading_map[(size, bold)] = level
            level += 1

        return heading_map

    def _should_use_numbering(
        self, font_heading_map: dict[tuple[float, bool], int]
    ) -> bool:
        """Use numbering-based hierarchy when font analysis gives only 1 level."""
        distinct_levels = len(set(font_heading_map.values()))
        return distinct_levels <= 1

    # ── Heading detection: numbering-based ───────────────────────────

    def _numbering_depth(self, text: str) -> int | None:
        """
        Extract heading depth from section numbering.
        "1. Introduction" → 1
        "1.1. Méthode" → 2
        "1.1.1. Détails" → 3
        Returns None if no numbering found.
        """
        m = _SECTION_NUM_RE.match(text)
        if not m:
            return None
        num_part = m.group(1)  # e.g. "1.2.3"
        depth = num_part.count(".") + 1  # "1" → 1, "1.2" → 2, "1.2.3" → 3
        return min(depth, 4)  # cap at h4

    def _is_named_heading(self, text: str) -> bool:
        """Detect named heading patterns like 'Document n°1', 'Cours n°2'."""
        return bool(_NAMED_HEADING_RE.match(text))

    # ── Markdown builder ─────────────────────────────────────────────

    def _build_markdown(
        self,
        lines: list[_Line],
        font_heading_map: dict[tuple[float, bool], int],
        use_numbering: bool,
    ) -> tuple[str, list[Section]]:
        parts: list[str] = []
        sections: list[Section] = []
        current_section: Section | None = None
        content_buffer: list[str] = []
        prev_page = -1

        for line in lines:
            text = line.text
            if not text:
                continue

            if line.page != prev_page and prev_page >= 0:
                parts.append("")
            prev_page = line.page

            heading_level = self._classify_heading(
                line, font_heading_map, use_numbering
            )

            if heading_level:
                if current_section and content_buffer:
                    current_section.content = "\n".join(content_buffer)
                    content_buffer = []

                clean_title = self._clean_heading(text)
                prefix = "#" * heading_level
                parts.append(f"\n{prefix} {clean_title}\n")

                current_section = Section(
                    title=clean_title,
                    level=heading_level,
                    page=line.page,
                )
                sections.append(current_section)
            else:
                formatted = self._format_body_line(text)
                parts.append(formatted)
                content_buffer.append(text)

        if current_section and content_buffer:
            current_section.content = "\n".join(content_buffer)

        markdown = "\n".join(parts)
        while "\n\n\n" in markdown:
            markdown = markdown.replace("\n\n\n", "\n\n")

        return markdown.strip(), sections

    def _classify_heading(
        self,
        line: _Line,
        font_heading_map: dict[tuple[float, bool], int],
        use_numbering: bool,
    ) -> int | None:
        text = line.text

        if len(text) > 150:
            return None
        if line.alpha_len < 3:
            return None
        if _BARE_NUMBER_RE.match(text):
            return None
        if self._is_toc_line(text):
            return None

        # Strategy 1: font-based (multiple font levels detected)
        size = line.dominant_size
        bold = line.is_bold
        key = (round(size, 1), bold)
        font_level = font_heading_map.get(key)

        if not use_numbering:
            # Pure font-based: use numbering to refine sub-levels within
            # the same font level
            if font_level:
                depth = self._numbering_depth(text)
                if depth:
                    return depth
                return font_level
            return None

        # Strategy 2: numbering-based (single font level for all headings)
        # Only classify as heading if the line is bold/styled as heading
        if font_level is None:
            return None

        # Named headings like "Document n°1", "Cours n°2" → h1
        if self._is_named_heading(text):
            return 1

        # Numbered headings: depth from numbering
        depth = self._numbering_depth(text)
        if depth:
            return depth

        # All-caps short lines → h1 (e.g. "INTRODUCTION")
        if line.is_all_upper and len(text) < 80:
            return 1

        # Fallback: bold heading without numbering → h2
        return 2

    def _clean_heading(self, text: str) -> str:
        text = _TOC_TRAILING_PAGE_RE.sub("", text)
        text = text.rstrip(". ")
        return text.strip()

    def _format_body_line(self, text: str) -> str:
        stripped = text.lstrip()
        if (
            stripped.startswith("- ")
            or stripped.startswith("• ")
            or stripped.startswith("– ")
            or stripped.startswith("▪ ")
        ):
            return f"- {stripped.lstrip('-•–▪ ').lstrip()}"
        return text

    # ── Title detection ──────────────────────────────────────────────

    def _detect_title(self, sections: list[Section], doc: fitz.Document) -> str:
        meta_title = doc.metadata.get("title", "").strip()
        if (meta_title and len(meta_title) > 5
                and not meta_title.endswith(".pdf")
                and " " in meta_title):
            return meta_title

        for s in sections:
            if s.level == 1:
                return s.title

        if sections:
            return sections[0].title

        return "Document"
