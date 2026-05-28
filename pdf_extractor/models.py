from dataclasses import dataclass, field


@dataclass
class Section:
    """A section detected from font analysis."""
    title: str
    level: int          # 1 = h1, 2 = h2, 3 = h3, etc.
    page: int
    content: str = ""


@dataclass
class StructuredDocument:
    """Result of PDF extraction: structured markdown + metadata."""
    markdown: str
    page_count: int
    title: str
    sections: list[Section] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
