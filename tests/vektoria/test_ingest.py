"""Ingestion pipeline tests: extraction (per format), chunking, and end-to-end."""

import io

import pytest

from vektoria import Index
from vektoria.embedding import HashEmbedder
from vektoria.ingest import Ingestor, chunk_text, extract_text


# ── chunking ─────────────────────────────────────────────────────────
def test_chunk_text_windows_with_overlap():
    words = " ".join(str(i) for i in range(10))
    chunks = chunk_text(words, max_words=4, overlap=1)
    assert chunks == ["0 1 2 3", "3 4 5 6", "6 7 8 9"]


def test_chunk_text_empty_and_validation():
    assert chunk_text("   ") == []
    with pytest.raises(ValueError):
        chunk_text("a b c", max_words=0)


def test_chunk_overlap_is_clamped():
    # overlap >= max_words would stall; it must be clamped so the window advances
    chunks = chunk_text("a b c d e", max_words=2, overlap=99)
    assert chunks and all(len(c.split()) <= 2 for c in chunks)
    assert "e" in chunks[-1]


# ── extraction (stdlib formats) ──────────────────────────────────────
def test_extract_txt_and_md():
    assert extract_text(b"hello world", "a.txt") == "hello world"
    assert "Title" in extract_text(b"# Title\n\nbody", "a.md")


def test_extract_csv_pads_short_rows():
    csv_bytes = b"name,age,city\nAlice,30,Paris\nBob,25\n"
    out = extract_text(csv_bytes, "people.csv")
    assert "name: Alice | age: 30 | city: Paris" in out
    assert "name: Bob | age: 25" in out  # short row didn't crash or drop silently


def test_extract_html_skips_script():
    html = b"<html><body><h1>Hi</h1><script>var x=1</script><p>Para</p></body></html>"
    out = extract_text(html, "a.html")
    assert "Hi" in out and "Para" in out and "var x" not in out


def test_extract_unsupported_raises():
    with pytest.raises(ValueError):
        extract_text(b"...", "a.xyz")


# ── extraction (optional heavy formats) ──────────────────────────────
def test_extract_pdf():
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "bonjour le monde")
    data = doc.tobytes()
    doc.close()
    assert "bonjour le monde" in extract_text(data, "x.pdf")


def test_extract_docx():
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_paragraph("paragraphe docx")
    buf = io.BytesIO()
    d.save(buf)
    assert "paragraphe docx" in extract_text(buf.getvalue(), "x.docx")


# ── end-to-end pipeline ──────────────────────────────────────────────
def test_ingest_end_to_end(tmp_path):
    emb = HashEmbedder(dimension=32)
    index = Index.create(tmp_path / "docs", dimension=32)
    ingestor = Ingestor(emb)

    text = (" ".join(f"word{i}" for i in range(120))).encode()
    result = ingestor.ingest(text, "doc.txt", index, max_words=50, overlap=10)

    assert result["source"] == "doc.txt"
    assert result["chunks"] == result["upserted"] >= 2
    assert index.count() == result["chunks"]

    # The stored chunks are queryable and carry provenance metadata.
    hit = index.query(emb.embed_query("word0"), top_k=1)[0]
    assert hit.metadata["source"] == "doc.txt"
    assert "chunk" in hit.metadata and "text" in hit.metadata
    index.close()


def test_ingest_dimension_mismatch_raises(tmp_path):
    index = Index.create(tmp_path / "docs", dimension=8)
    ingestor = Ingestor(HashEmbedder(dimension=32))
    with pytest.raises(ValueError):
        ingestor.ingest(b"hello", "a.txt", index)
    index.close()


def test_ingest_empty_document(tmp_path):
    index = Index.create(tmp_path / "docs", dimension=16)
    ingestor = Ingestor(HashEmbedder(dimension=16))
    result = ingestor.ingest(b"   ", "empty.txt", index)
    assert result == {"source": "empty.txt", "chunks": 0, "upserted": 0}
    assert index.count() == 0
    index.close()
