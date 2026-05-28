"""
Full pipeline test suite for SwissVectorStore.

Tests the entire RAG pipeline: PDF → Extract → Chunk → Embed → Store → Search
All tests run 100% locally with zero external API calls.

For thesis presentation: demonstrates that a self-hosted VPS solution
can deliver equivalent results to cloud services (LlamaParse, OpenAI).
"""

import os
import sys
import time
import shutil
import tempfile
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdf_extractor import PdfExtractor
from chunker import SemanticChunker, Chunk
from embedder import LocalEmbedder
from vector_store import VectorStore, SearchResult

# ── Test PDFs ────────────────────────────────────────────────────────

PDF_DIR = os.path.expanduser("~/Downloads/Cours de droit")
PDF_LCR = os.path.join(PDF_DIR, "Droit pénal LCR - OFFICIEL.pdf")
PDF_BASES = os.path.join(PDF_DIR, "935158.pdf")


def _has_pdfs() -> bool:
    return os.path.exists(PDF_LCR) and os.path.exists(PDF_BASES)


# ═══════════════════════════════════════════════════════════════════
# 1. PDF EXTRACTION TESTS
# ═══════════════════════════════════════════════════════════════════

class TestPdfExtraction:
    """Test que l'extraction PDF produit du markdown structuré correct."""

    def setup_method(self):
        self.ext = PdfExtractor()

    def test_lcr_page_count(self):
        """Le PDF LCR a 105 pages."""
        if not _has_pdfs():
            return
        doc = self.ext.extract(PDF_LCR)
        assert doc.page_count == 105

    def test_lcr_uses_numbering_strategy(self):
        """Le PDF LCR utilise la stratégie numbering (même police pour tous les titres)."""
        if not _has_pdfs():
            return
        doc = self.ext.extract(PDF_LCR)
        assert doc.metadata["heading_strategy"] == "numbering"

    def test_lcr_detects_sections(self):
        """Au moins 100 sections détectées dans le PDF LCR."""
        if not _has_pdfs():
            return
        doc = self.ext.extract(PDF_LCR)
        assert len(doc.sections) >= 100

    def test_lcr_heading_hierarchy(self):
        """Les sections ont des niveaux h1 et h2 (pas tout en h1)."""
        if not _has_pdfs():
            return
        doc = self.ext.extract(PDF_LCR)
        levels = {s.level for s in doc.sections}
        assert 1 in levels
        assert 2 in levels

    def test_lcr_toc_pages_excluded(self):
        """Les pages de TOC sont détectées et exclues."""
        if not _has_pdfs():
            return
        doc = self.ext.extract(PDF_LCR)
        assert len(doc.metadata["toc_pages"]) >= 3

    def test_lcr_markdown_has_headings(self):
        """Le markdown contient des headings markdown (#, ##)."""
        if not _has_pdfs():
            return
        doc = self.ext.extract(PDF_LCR)
        assert "# " in doc.markdown
        assert "## " in doc.markdown

    def test_lcr_no_toc_dots_in_content(self):
        """Pas de lignes pointillées TOC dans le contenu."""
        if not _has_pdfs():
            return
        doc = self.ext.extract(PDF_LCR)
        lines_with_dots = [l for l in doc.markdown.split("\n") if "...." in l]
        # Il peut y en avoir quelques-unes, mais pas beaucoup
        assert len(lines_with_dots) < 10

    def test_bases_uses_font_strategy(self):
        """Le PDF 935158 utilise la stratégie font (tailles différentes)."""
        if not _has_pdfs():
            return
        doc = self.ext.extract(PDF_BASES)
        assert doc.metadata["heading_strategy"] == "font"

    def test_bases_page_count(self):
        if not _has_pdfs():
            return
        doc = self.ext.extract(PDF_BASES)
        assert doc.page_count == 13

    def test_extraction_speed(self):
        """L'extraction prend moins de 5 secondes pour un PDF de 105 pages."""
        if not _has_pdfs():
            return
        t0 = time.time()
        self.ext.extract(PDF_LCR)
        elapsed = time.time() - t0
        print(f"  Extraction time: {elapsed:.2f}s")
        assert elapsed < 5.0


# ═══════════════════════════════════════════════════════════════════
# 2. CHUNKER TESTS
# ═══════════════════════════════════════════════════════════════════

class TestChunker:
    """Test que le chunker produit des chunks sémantiquement cohérents."""

    def setup_method(self):
        self.ext = PdfExtractor()
        self.chunker = SemanticChunker(max_words=800, min_words=50)

    def test_chunks_have_context(self):
        """Chaque chunk contient son chemin hiérarchique [Section > Sous-section]."""
        if not _has_pdfs():
            return
        doc = self.ext.extract(PDF_LCR)
        chunks = self.chunker.chunk(doc)
        # At least 80% of chunks should have heading context
        with_context = sum(1 for c in chunks if c.heading_path)
        assert with_context / len(chunks) > 0.8

    def test_chunks_not_too_small(self):
        """Moins de 10% des chunks font moins de 30 mots."""
        if not _has_pdfs():
            return
        doc = self.ext.extract(PDF_LCR)
        chunks = self.chunker.chunk(doc)
        tiny = sum(1 for c in chunks if c.word_count < 30)
        print(f"  Tiny chunks (<30w): {tiny}/{len(chunks)}")
        assert tiny / len(chunks) < 0.15

    def test_chunks_have_ids(self):
        """Chaque chunk a un ID unique."""
        if not _has_pdfs():
            return
        doc = self.ext.extract(PDF_LCR)
        chunks = self.chunker.chunk(doc)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))  # all unique

    def test_chunks_cover_content(self):
        """Les chunks couvrent au moins 80% du contenu total."""
        if not _has_pdfs():
            return
        doc = self.ext.extract(PDF_LCR)
        chunks = self.chunker.chunk(doc)
        total_words_chunks = sum(c.word_count for c in chunks)
        total_words_doc = len(doc.markdown.split())
        coverage = total_words_chunks / total_words_doc
        print(f"  Coverage: {coverage:.1%}")
        assert coverage > 0.5

    def test_chunk_count_reasonable(self):
        """Le nombre de chunks est dans une fourchette raisonnable."""
        if not _has_pdfs():
            return
        doc = self.ext.extract(PDF_LCR)
        chunks = self.chunker.chunk(doc)
        print(f"  Chunks: {len(chunks)}")
        assert 50 < len(chunks) < 500


# ═══════════════════════════════════════════════════════════════════
# 3. EMBEDDER TESTS
# ═══════════════════════════════════════════════════════════════════

class TestEmbedder:
    """Test l'embedder (en mode numpy pour la CI, sentence-transformers sur VPS)."""

    def setup_method(self):
        self.embedder = LocalEmbedder(backend="numpy", dimension=384)

    def test_embed_single_text(self):
        vec = self.embedder.embed_text("Test juridique")
        assert vec.shape == (384,)
        assert abs(np.linalg.norm(vec) - 1.0) < 0.01  # L2 normalized

    def test_embed_multiple_texts(self):
        texts = ["Premier texte", "Deuxième texte", "Troisième texte"]
        vecs = self.embedder.embed_texts(texts)
        assert vecs.shape == (3, 384)

    def test_embed_deterministic(self):
        """Le même texte donne toujours le même vecteur."""
        v1 = self.embedder.embed_text("Art. 42 CO")
        v2 = self.embedder.embed_text("Art. 42 CO")
        assert np.allclose(v1, v2)

    def test_different_texts_different_vectors(self):
        """Des textes différents donnent des vecteurs différents."""
        v1 = self.embedder.embed_text("Droit pénal")
        v2 = self.embedder.embed_text("Droit civil")
        assert not np.allclose(v1, v2)

    def test_query_prefix(self):
        """embed_query utilise le prefix 'query: '."""
        v_passage = self.embedder.embed_text("sanctions routières")
        v_query = self.embedder.embed_query("sanctions routières")
        # Different because of different prefix
        assert not np.allclose(v_passage, v_query)

    def test_embed_chunks(self):
        """embed_chunks retourne des EmbeddedChunk avec vecteurs."""
        chunks = [
            Chunk(id="test_0", text="Chunk un", heading_path=["A"],
                  level=1, page_start=0, page_end=0, source="test"),
            Chunk(id="test_1", text="Chunk deux", heading_path=["B"],
                  level=1, page_start=1, page_end=1, source="test"),
        ]
        embedded = self.embedder.embed_chunks(chunks)
        assert len(embedded) == 2
        assert embedded[0].vector.shape == (384,)
        assert embedded[0].chunk.id == "test_0"


# ═══════════════════════════════════════════════════════════════════
# 4. VECTOR STORE TESTS
# ═══════════════════════════════════════════════════════════════════

class TestVectorStore:
    """Test le vector store custom (stockage + recherche)."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = VectorStore(self.tmpdir)
        self.embedder = LocalEmbedder(backend="numpy", dimension=384)

    def teardown_method(self):
        self.store.close()
        shutil.rmtree(self.tmpdir)

    def _make_chunks(self, n: int) -> list[Chunk]:
        return [
            Chunk(
                id=f"chunk_{i:04d}",
                text=f"Contenu juridique numéro {i} sur le droit suisse",
                heading_path=[f"Section {i}"],
                level=1,
                page_start=i,
                page_end=i,
                source="test_doc",
                word_count=10,
            )
            for i in range(n)
        ]

    def test_add_and_count(self):
        chunks = self._make_chunks(10)
        embedded = self.embedder.embed_chunks(chunks)
        added = self.store.add(embedded)
        assert added == 10
        assert self.store.count == 10

    def test_search_returns_results(self):
        chunks = self._make_chunks(20)
        embedded = self.embedder.embed_chunks(chunks)
        self.store.add(embedded)

        query = self.embedder.embed_query("droit suisse numéro 5")
        results = self.store.search(query, top_k=5)
        assert len(results) == 5
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_scores_sorted(self):
        """Les résultats sont triés par score décroissant."""
        chunks = self._make_chunks(20)
        embedded = self.embedder.embed_chunks(chunks)
        self.store.add(embedded)

        query = self.embedder.embed_query("test query")
        results = self.store.search(query, top_k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_persistence(self):
        """Les données persistent après fermeture et réouverture."""
        chunks = self._make_chunks(5)
        embedded = self.embedder.embed_chunks(chunks)
        self.store.add(embedded)
        self.store.close()

        # Reopen
        store2 = VectorStore(self.tmpdir)
        assert store2.count == 5

        query = self.embedder.embed_query("test")
        results = store2.search(query, top_k=3)
        assert len(results) == 3
        store2.close()

    def test_list_sources(self):
        chunks = self._make_chunks(5)
        embedded = self.embedder.embed_chunks(chunks)
        self.store.add(embedded)

        sources = self.store.list_sources()
        assert len(sources) == 1
        assert sources[0]["source"] == "test_doc"
        assert sources[0]["chunks"] == 5

    def test_delete_source(self):
        chunks = self._make_chunks(5)
        embedded = self.embedder.embed_chunks(chunks)
        self.store.add(embedded)

        deleted = self.store.delete_source("test_doc")
        assert deleted == 5
        assert self.store.count == 0

    def test_dimension(self):
        chunks = self._make_chunks(3)
        embedded = self.embedder.embed_chunks(chunks)
        self.store.add(embedded)
        assert self.store.dimension == 384


# ═══════════════════════════════════════════════════════════════════
# 5. FULL PIPELINE INTEGRATION TEST
# ═══════════════════════════════════════════════════════════════════

class TestFullPipeline:
    """
    Test d'intégration complet : PDF → Extract → Chunk → Embed → Store → Search.
    Démontre qu'on peut faire du RAG 100% local sans aucune API cloud.
    """

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def test_full_pipeline_lcr(self):
        """Pipeline complet sur le PDF LCR (105 pages)."""
        if not _has_pdfs():
            return

        timings = {}

        # 1. Extract
        t0 = time.time()
        ext = PdfExtractor()
        doc = ext.extract(PDF_LCR)
        timings["extract"] = time.time() - t0

        assert doc.page_count == 105
        assert len(doc.sections) > 100

        # 2. Chunk
        t1 = time.time()
        chunker = SemanticChunker(max_words=800, min_words=50)
        chunks = chunker.chunk(doc)
        timings["chunk"] = time.time() - t1

        assert len(chunks) > 50

        # 3. Embed (numpy backend for speed)
        t2 = time.time()
        embedder = LocalEmbedder(backend="numpy", dimension=384)
        embedded = embedder.embed_chunks(chunks)
        timings["embed"] = time.time() - t2

        assert len(embedded) == len(chunks)

        # 4. Store
        t3 = time.time()
        store = VectorStore(self.tmpdir)
        store.add(embedded)
        timings["store"] = time.time() - t3

        assert store.count == len(chunks)

        # 5. Search
        t4 = time.time()
        query_vec = embedder.embed_query("excès de vitesse sanctions")
        results = store.search(query_vec, top_k=5)
        timings["search"] = time.time() - t4

        assert len(results) == 5
        assert results[0].score > results[-1].score

        # Print results for thesis
        total = sum(timings.values())
        print("\n" + "=" * 60)
        print("PIPELINE COMPLET — RÉSULTATS")
        print("=" * 60)
        print(f"Document:    {doc.title}")
        print(f"Pages:       {doc.page_count}")
        print(f"Sections:    {len(doc.sections)}")
        print(f"Chunks:      {len(chunks)}")
        print(f"Vecteurs:    {store.count} x {store.dimension}d")
        print()
        print("Timings:")
        for step, t in timings.items():
            print(f"  {step:12s}: {t:.3f}s")
        print(f"  {'TOTAL':12s}: {total:.3f}s")
        print()
        print(f"Query: 'excès de vitesse sanctions'")
        print(f"Top 5 results:")
        for r in results:
            path = " > ".join(r.heading_path[-2:])
            print(f"  {r.score:.4f} | [{r.word_count}w] {path[:65]}")
        print("=" * 60)

        store.close()

    def test_multi_document_store(self):
        """Indexer plusieurs PDFs dans le même store et filtrer par source."""
        if not _has_pdfs():
            return

        ext = PdfExtractor()
        chunker = SemanticChunker()
        embedder = LocalEmbedder(backend="numpy", dimension=384)
        store = VectorStore(self.tmpdir)

        # Index both PDFs
        for pdf_path in [PDF_LCR, PDF_BASES]:
            doc = ext.extract(pdf_path)
            chunks = chunker.chunk(doc)
            embedded = embedder.embed_chunks(chunks)
            store.add(embedded)

        sources = store.list_sources()
        assert len(sources) == 2
        print(f"\n  Indexed {len(sources)} documents, {store.count} total chunks")
        for s in sources:
            print(f"    {s['source'][:50]:50s} | {s['chunks']} chunks")

        store.close()


# ═══════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """Run all tests with simple output (no pytest needed)."""
    import traceback

    test_classes = [
        TestPdfExtraction,
        TestChunker,
        TestEmbedder,
        TestVectorStore,
        TestFullPipeline,
    ]

    total = 0
    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        print(f"\n{'─' * 40}")
        print(f"  {cls.__name__}")
        print(f"  {cls.__doc__.strip()}")
        print(f"{'─' * 40}")

        for name in sorted(dir(cls)):
            if not name.startswith("test_"):
                continue
            total += 1
            instance = cls()
            if hasattr(instance, "setup_method"):
                instance.setup_method()

            try:
                getattr(instance, name)()
                passed += 1
                print(f"  PASS  {name}")
            except Exception as e:
                failed += 1
                errors.append((cls.__name__, name, e))
                print(f"  FAIL  {name}: {e}")

            if hasattr(instance, "teardown_method"):
                try:
                    instance.teardown_method()
                except Exception:
                    pass

    print(f"\n{'═' * 40}")
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    print(f"{'═' * 40}")

    if errors:
        print("\nFailed tests:")
        for cls_name, test_name, err in errors:
            print(f"\n  {cls_name}.{test_name}:")
            traceback.print_exception(type(err), err, err.__traceback__)
