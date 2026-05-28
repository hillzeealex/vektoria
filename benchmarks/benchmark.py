"""
Benchmark suite for SwissVectorStore.

Measures extraction speed, chunking quality, embedding throughput,
search latency, search precision, storage efficiency, and memory usage.

Designed for thesis: proving a self-hosted VPS can compete with cloud solutions.
"""

import os
import sys
import time
import json
import shutil
import tempfile
import resource
import statistics
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdf_extractor import PdfExtractor
from chunker import SemanticChunker, Chunk
from embedder import LocalEmbedder
from vector_store import VectorStore
from benchmarks.test_queries import TEST_QUERIES


@dataclass
class ExtractionMetrics:
    total_time_s: float = 0.0
    page_count: int = 0
    section_count: int = 0
    time_per_page_s: float = 0.0
    heading_strategy: str = ""
    toc_pages_detected: int = 0
    markdown_length_chars: int = 0
    peak_rss_mb: float = 0.0


@dataclass
class ChunkingMetrics:
    total_time_s: float = 0.0
    chunk_count: int = 0
    avg_words_per_chunk: float = 0.0
    min_words: int = 0
    max_words: int = 0
    median_words: int = 0
    total_words_chunks: int = 0
    total_words_doc: int = 0
    coverage_pct: float = 0.0
    chunks_with_context_pct: float = 0.0
    tiny_chunks_pct: float = 0.0
    peak_rss_mb: float = 0.0


@dataclass
class EmbeddingMetrics:
    total_time_s: float = 0.0
    vector_count: int = 0
    dimension: int = 0
    vectors_per_second: float = 0.0
    backend: str = ""
    peak_rss_mb: float = 0.0


@dataclass
class SearchLatencyMetrics:
    num_queries: int = 0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    mean_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    peak_rss_mb: float = 0.0


@dataclass
class SearchPrecisionMetrics:
    num_queries: int = 0
    k: int = 5
    recall_at_k: float = 0.0
    per_query: list = field(default_factory=list)


@dataclass
class StorageMetrics:
    vectors_file_bytes: int = 0
    metadata_db_bytes: int = 0
    total_bytes: int = 0
    bytes_per_chunk: float = 0.0
    chunk_count: int = 0


@dataclass
class BenchmarkResults:
    pdf_path: str = ""
    pdf_name: str = ""
    timestamp: str = ""
    extraction: ExtractionMetrics = field(default_factory=ExtractionMetrics)
    chunking: ChunkingMetrics = field(default_factory=ChunkingMetrics)
    embedding: EmbeddingMetrics = field(default_factory=EmbeddingMetrics)
    search_latency: SearchLatencyMetrics = field(default_factory=SearchLatencyMetrics)
    search_precision: SearchPrecisionMetrics = field(default_factory=SearchPrecisionMetrics)
    storage: StorageMetrics = field(default_factory=StorageMetrics)

    def to_dict(self) -> dict:
        return asdict(self)


def _get_rss_mb() -> float:
    """Get current peak RSS in MB using the resource module."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # On macOS, ru_maxrss is in bytes; on Linux it's in KB
    if sys.platform == "darwin":
        return usage.ru_maxrss / (1024 * 1024)
    else:
        return usage.ru_maxrss / 1024


class Benchmark:
    """
    Runs all benchmarks on a given PDF file.

    Usage:
        bench = Benchmark("path/to/document.pdf")
        results = bench.run()
    """

    def __init__(
        self,
        pdf_path: str,
        *,
        embed_backend: str = "numpy",
        embed_dim: int = 384,
        top_k: int = 5,
        search_iterations: int = 100,
    ):
        self.pdf_path = os.path.abspath(pdf_path)
        self.pdf_name = os.path.basename(pdf_path)
        self.embed_backend = embed_backend
        self.embed_dim = embed_dim
        self.top_k = top_k
        self.search_iterations = search_iterations

        self._extractor = PdfExtractor()
        self._chunker = SemanticChunker(max_words=800, min_words=50)
        self._embedder = LocalEmbedder(
            backend=embed_backend, dimension=embed_dim
        )

        # Populated during run
        self._doc = None
        self._chunks = None
        self._embedded = None
        self._store = None
        self._store_dir = None

    def run(self) -> BenchmarkResults:
        """Run all benchmarks and return results."""
        from datetime import datetime

        results = BenchmarkResults(
            pdf_path=self.pdf_path,
            pdf_name=self.pdf_name,
            timestamp=datetime.now().isoformat(),
        )

        print(f"\n{'=' * 62}")
        print(f"  BENCHMARK: {self.pdf_name}")
        print(f"{'=' * 62}")

        results.extraction = self._bench_extraction()
        results.chunking = self._bench_chunking()
        results.embedding = self._bench_embedding()
        results.search_latency = self._bench_search_latency()
        results.search_precision = self._bench_search_precision()
        results.storage = self._bench_storage()

        # Cleanup
        if self._store:
            self._store.close()
        if self._store_dir:
            shutil.rmtree(self._store_dir, ignore_errors=True)

        return results

    def _bench_extraction(self) -> ExtractionMetrics:
        print("\n  [1/6] Extraction (PDF -> Markdown)...")
        metrics = ExtractionMetrics()

        rss_before = _get_rss_mb()
        t0 = time.perf_counter()
        self._doc = self._extractor.extract(self.pdf_path)
        elapsed = time.perf_counter() - t0

        metrics.total_time_s = round(elapsed, 4)
        metrics.page_count = self._doc.page_count
        metrics.section_count = len(self._doc.sections)
        metrics.time_per_page_s = round(elapsed / max(self._doc.page_count, 1), 4)
        metrics.heading_strategy = self._doc.metadata.get("heading_strategy", "unknown")
        metrics.toc_pages_detected = len(self._doc.metadata.get("toc_pages", []))
        metrics.markdown_length_chars = len(self._doc.markdown)
        metrics.peak_rss_mb = round(_get_rss_mb(), 1)

        print(f"        {metrics.page_count} pages, {metrics.section_count} sections")
        print(f"        {metrics.total_time_s}s total ({metrics.time_per_page_s}s/page)")
        return metrics

    def _bench_chunking(self) -> ChunkingMetrics:
        print("  [2/6] Chunking (Markdown -> semantic chunks)...")
        metrics = ChunkingMetrics()

        t0 = time.perf_counter()
        self._chunks = self._chunker.chunk(self._doc)
        elapsed = time.perf_counter() - t0

        word_counts = [c.word_count for c in self._chunks]
        total_words_doc = len(self._doc.markdown.split())
        total_words_chunks = sum(word_counts)
        with_context = sum(1 for c in self._chunks if c.heading_path)
        tiny = sum(1 for c in self._chunks if c.word_count < 30)

        metrics.total_time_s = round(elapsed, 4)
        metrics.chunk_count = len(self._chunks)
        metrics.avg_words_per_chunk = round(statistics.mean(word_counts), 1) if word_counts else 0
        metrics.min_words = min(word_counts) if word_counts else 0
        metrics.max_words = max(word_counts) if word_counts else 0
        metrics.median_words = int(statistics.median(word_counts)) if word_counts else 0
        metrics.total_words_chunks = total_words_chunks
        metrics.total_words_doc = total_words_doc
        metrics.coverage_pct = round(total_words_chunks / max(total_words_doc, 1) * 100, 1)
        metrics.chunks_with_context_pct = round(with_context / max(len(self._chunks), 1) * 100, 1)
        metrics.tiny_chunks_pct = round(tiny / max(len(self._chunks), 1) * 100, 1)
        metrics.peak_rss_mb = round(_get_rss_mb(), 1)

        print(f"        {metrics.chunk_count} chunks, avg {metrics.avg_words_per_chunk} words")
        print(f"        coverage {metrics.coverage_pct}%, context {metrics.chunks_with_context_pct}%")
        return metrics

    def _bench_embedding(self) -> EmbeddingMetrics:
        print(f"  [3/6] Embedding ({self.embed_backend} backend)...")
        metrics = EmbeddingMetrics()

        t0 = time.perf_counter()
        self._embedded = self._embedder.embed_chunks(self._chunks)
        elapsed = time.perf_counter() - t0

        metrics.total_time_s = round(elapsed, 4)
        metrics.vector_count = len(self._embedded)
        metrics.dimension = self.embed_dim
        metrics.vectors_per_second = round(len(self._embedded) / max(elapsed, 0.0001), 1)
        metrics.backend = self.embed_backend
        metrics.peak_rss_mb = round(_get_rss_mb(), 1)

        print(f"        {metrics.vector_count} vectors, {metrics.vectors_per_second} vec/s")
        return metrics

    def _bench_search_latency(self) -> SearchLatencyMetrics:
        print(f"  [4/6] Search latency ({self.search_iterations} queries)...")
        metrics = SearchLatencyMetrics()

        # Build the store
        self._store_dir = tempfile.mkdtemp(prefix="svs_bench_")
        self._store = VectorStore(self._store_dir)
        self._store.add(self._embedded)

        # Run queries repeatedly
        queries = [q["question"] for q in TEST_QUERIES]
        latencies_ms = []

        for i in range(self.search_iterations):
            query_text = queries[i % len(queries)]
            query_vec = self._embedder.embed_query(query_text)

            t0 = time.perf_counter()
            self._store.search(query_vec, top_k=self.top_k)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies_ms.append(elapsed_ms)

        latencies_ms.sort()
        metrics.num_queries = self.search_iterations
        metrics.p50_ms = round(latencies_ms[int(len(latencies_ms) * 0.50)], 3)
        metrics.p95_ms = round(latencies_ms[int(len(latencies_ms) * 0.95)], 3)
        metrics.p99_ms = round(latencies_ms[min(int(len(latencies_ms) * 0.99), len(latencies_ms) - 1)], 3)
        metrics.mean_ms = round(statistics.mean(latencies_ms), 3)
        metrics.min_ms = round(min(latencies_ms), 3)
        metrics.max_ms = round(max(latencies_ms), 3)
        metrics.peak_rss_mb = round(_get_rss_mb(), 1)

        print(f"        p50={metrics.p50_ms}ms  p95={metrics.p95_ms}ms  p99={metrics.p99_ms}ms")
        return metrics

    def _bench_search_precision(self) -> SearchPrecisionMetrics:
        print(f"  [5/6] Search precision (recall@{self.top_k})...")
        metrics = SearchPrecisionMetrics(
            num_queries=len(TEST_QUERIES),
            k=self.top_k,
        )

        hits = 0
        total = 0

        for tq in TEST_QUERIES:
            query_vec = self._embedder.embed_query(tq["question"])
            results = self._store.search(query_vec, top_k=self.top_k)

            # Check if any expected keyword appears in any result text
            result_text = " ".join(r.text.lower() for r in results)
            keywords_found = []
            keywords_missing = []

            for kw in tq["expected_keywords"]:
                if kw.lower() in result_text:
                    keywords_found.append(kw)
                else:
                    keywords_missing.append(kw)

            query_recall = len(keywords_found) / max(len(tq["expected_keywords"]), 1)
            hits += len(keywords_found)
            total += len(tq["expected_keywords"])

            metrics.per_query.append({
                "question": tq["question"],
                "recall": round(query_recall, 3),
                "found": keywords_found,
                "missing": keywords_missing,
            })

        metrics.recall_at_k = round(hits / max(total, 1), 3)
        print(f"        recall@{self.top_k} = {metrics.recall_at_k:.1%} ({hits}/{total} keywords)")
        return metrics

    def _bench_storage(self) -> StorageMetrics:
        print("  [6/6] Storage efficiency...")
        metrics = StorageMetrics()

        store_path = Path(self._store_dir)
        vectors_file = store_path / "vectors.npy"
        db_file = store_path / "metadata.db"

        metrics.vectors_file_bytes = vectors_file.stat().st_size if vectors_file.exists() else 0
        metrics.metadata_db_bytes = db_file.stat().st_size if db_file.exists() else 0
        metrics.total_bytes = metrics.vectors_file_bytes + metrics.metadata_db_bytes
        metrics.chunk_count = self._store.count
        metrics.bytes_per_chunk = round(
            metrics.total_bytes / max(metrics.chunk_count, 1), 1
        )

        total_kb = metrics.total_bytes / 1024
        print(f"        {total_kb:.1f} KB total, {metrics.bytes_per_chunk:.0f} bytes/chunk")
        return metrics
