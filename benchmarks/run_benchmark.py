#!/usr/bin/env python3
"""
CLI benchmark runner for SwissVectorStore.

Usage:
    python benchmarks/run_benchmark.py path/to/document.pdf
    python benchmarks/run_benchmark.py ~/Downloads/Cours\ de\ droit/Droit\ pénal\ LCR\ -\ OFFICIEL.pdf

Runs all benchmarks and outputs a formatted comparison report.
Results are saved to benchmarks/results/ as JSON.
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.benchmark import Benchmark, BenchmarkResults


# ── Cloud reference values ──────────────────────────────────────────
# Hardcoded typical values from public documentation and benchmarks.
# Sources: LlamaParse pricing, OpenAI embeddings API, Pinecone docs.

CLOUD_REFERENCE = {
    "extract_100_pages": "~30-90s (LlamaParse)",
    "extract_cost_per_page": "$0.003/page (LlamaParse)",
    "embed_cost_per_1k_tokens": "$0.0001/1K tok (OpenAI)",
    "search_latency_p50": "~50-200ms (Pinecone)",
    "search_latency_p99": "~100-500ms (Pinecone)",
    "storage_cost_per_gb_month": "$0.33/GB/mo (Pinecone)",
    "cost_per_document_100p": "~$0.30-1.00",
    "data_sovereignty": "0% (US servers)",
    "vendor_lock_in": "High",
}


def _format_report(results: BenchmarkResults) -> str:
    """Build the formatted comparison report."""
    e = results.extraction
    c = results.chunking
    em = results.embedding
    sl = results.search_latency
    sp = results.search_precision
    st = results.storage

    # Extrapolate extraction time for 100 pages
    extract_100p = e.time_per_page_s * 100

    lines = []
    lines.append("")
    lines.append("=" * 66)
    lines.append("  SwissVectorStore -- Benchmark Report")
    lines.append(f"  {results.pdf_name}")
    lines.append(f"  {results.timestamp}")
    lines.append("=" * 66)

    # ── Comparison table ──
    lines.append("")
    lines.append(
        f"{'':2s}"
        f"+{'=' * 25}+{'=' * 17}+{'=' * 21}+"
    )
    lines.append(
        f"{'':2s}"
        f"|{'Metric':^25s}|{'SwissVS (VPS)':^17s}|{'Cloud (estimated)':^21s}|"
    )
    lines.append(
        f"{'':2s}"
        f"+{'-' * 25}+{'-' * 17}+{'-' * 21}+"
    )

    rows = [
        (
            f"Extract {e.page_count} pages",
            f"{e.total_time_s:.2f}s",
            CLOUD_REFERENCE["extract_100_pages"],
        ),
        (
            "Extract per page",
            f"{e.time_per_page_s * 1000:.1f}ms",
            f"{CLOUD_REFERENCE['extract_cost_per_page']}",
        ),
        (
            "Search latency p50",
            f"{sl.p50_ms:.2f}ms",
            CLOUD_REFERENCE["search_latency_p50"],
        ),
        (
            "Search latency p99",
            f"{sl.p99_ms:.2f}ms",
            CLOUD_REFERENCE["search_latency_p99"],
        ),
        (
            f"Recall@{sp.k}",
            f"{sp.recall_at_k:.1%}",
            "N/A (depends on model)",
        ),
        (
            "Embed throughput",
            f"{em.vectors_per_second:.0f} vec/s",
            "~100-500 vec/s (API)",
        ),
        (
            "Storage per chunk",
            f"{st.bytes_per_chunk:.0f} bytes",
            "~2-5 KB (Pinecone)",
        ),
        (
            "Cost per document",
            "0 CHF",
            CLOUD_REFERENCE["cost_per_document_100p"],
        ),
        (
            "Data sovereignty",
            "100% (Swiss VPS)",
            CLOUD_REFERENCE["data_sovereignty"],
        ),
        (
            "Vendor lock-in",
            "None (OSS)",
            CLOUD_REFERENCE["vendor_lock_in"],
        ),
    ]

    for label, svs_val, cloud_val in rows:
        lines.append(
            f"{'':2s}"
            f"|{label:<25s}|{svs_val:>17s}|{cloud_val:>21s}|"
        )

    lines.append(
        f"{'':2s}"
        f"+{'=' * 25}+{'=' * 17}+{'=' * 21}+"
    )

    # ── Detailed metrics ──
    lines.append("")
    lines.append("-" * 66)
    lines.append("  DETAILED METRICS")
    lines.append("-" * 66)

    lines.append("")
    lines.append("  EXTRACTION")
    lines.append(f"    Pages:              {e.page_count}")
    lines.append(f"    Sections:           {e.section_count}")
    lines.append(f"    Strategy:           {e.heading_strategy}")
    lines.append(f"    TOC pages:          {e.toc_pages_detected}")
    lines.append(f"    Markdown length:    {e.markdown_length_chars:,} chars")
    lines.append(f"    Time:               {e.total_time_s:.4f}s")
    lines.append(f"    Time/page:          {e.time_per_page_s * 1000:.2f}ms")
    lines.append(f"    Peak RSS:           {e.peak_rss_mb:.1f} MB")

    lines.append("")
    lines.append("  CHUNKING")
    lines.append(f"    Chunks:             {c.chunk_count}")
    lines.append(f"    Avg words/chunk:    {c.avg_words_per_chunk}")
    lines.append(f"    Min/Med/Max words:  {c.min_words} / {c.median_words} / {c.max_words}")
    lines.append(f"    Coverage:           {c.coverage_pct}%")
    lines.append(f"    With context:       {c.chunks_with_context_pct}%")
    lines.append(f"    Tiny (<30w):        {c.tiny_chunks_pct}%")
    lines.append(f"    Time:               {c.total_time_s:.4f}s")
    lines.append(f"    Peak RSS:           {c.peak_rss_mb:.1f} MB")

    lines.append("")
    lines.append("  EMBEDDING")
    lines.append(f"    Backend:            {em.backend}")
    lines.append(f"    Vectors:            {em.vector_count} x {em.dimension}d")
    lines.append(f"    Throughput:         {em.vectors_per_second} vec/s")
    lines.append(f"    Time:               {em.total_time_s:.4f}s")
    lines.append(f"    Peak RSS:           {em.peak_rss_mb:.1f} MB")

    lines.append("")
    lines.append("  SEARCH LATENCY")
    lines.append(f"    Queries:            {sl.num_queries}")
    lines.append(f"    p50:                {sl.p50_ms:.3f}ms")
    lines.append(f"    p95:                {sl.p95_ms:.3f}ms")
    lines.append(f"    p99:                {sl.p99_ms:.3f}ms")
    lines.append(f"    Mean:               {sl.mean_ms:.3f}ms")
    lines.append(f"    Min/Max:            {sl.min_ms:.3f}ms / {sl.max_ms:.3f}ms")
    lines.append(f"    Peak RSS:           {sl.peak_rss_mb:.1f} MB")

    lines.append("")
    lines.append("  SEARCH PRECISION")
    lines.append(f"    Queries:            {sp.num_queries}")
    lines.append(f"    Recall@{sp.k}:          {sp.recall_at_k:.1%}")

    # Show per-query breakdown
    for pq in sp.per_query:
        status = "OK" if pq["recall"] >= 0.5 else "LOW"
        lines.append(
            f"      [{status:3s}] {pq['recall']:.0%} | {pq['question'][:50]}"
        )
        if pq["missing"]:
            lines.append(f"             missing: {', '.join(pq['missing'])}")

    lines.append("")
    lines.append("  STORAGE")
    lines.append(f"    Vectors file:       {st.vectors_file_bytes:,} bytes ({st.vectors_file_bytes / 1024:.1f} KB)")
    lines.append(f"    Metadata DB:        {st.metadata_db_bytes:,} bytes ({st.metadata_db_bytes / 1024:.1f} KB)")
    lines.append(f"    Total:              {st.total_bytes:,} bytes ({st.total_bytes / 1024:.1f} KB)")
    lines.append(f"    Per chunk:          {st.bytes_per_chunk:.0f} bytes")

    lines.append("")
    lines.append("=" * 66)
    lines.append("")
    lines.append("  NOTE: With numpy (test) backend, search precision reflects")
    lines.append("  hash-based pseudo-embeddings, not real semantic similarity.")
    lines.append("  For real precision, use --backend sentence-transformers.")
    lines.append("")
    lines.append("=" * 66)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Run SwissVectorStore benchmarks on a PDF",
    )
    parser.add_argument("pdf", help="Path to a PDF file to benchmark")
    parser.add_argument(
        "--backend",
        default="numpy",
        choices=["numpy", "sentence-transformers", "ollama"],
        help="Embedding backend (default: numpy)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=384,
        help="Embedding dimension for numpy backend (default: 384)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of search results (default: 5)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of search queries for latency benchmark (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save JSON results (default: benchmarks/results/)",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.pdf):
        print(f"Error: file not found: {args.pdf}")
        sys.exit(1)

    # Run benchmarks
    bench = Benchmark(
        args.pdf,
        embed_backend=args.backend,
        embed_dim=args.dim,
        top_k=args.top_k,
        search_iterations=args.iterations,
    )
    results = bench.run()

    # Print formatted report
    report = _format_report(results)
    print(report)

    # Save JSON results
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results"
        )
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = os.path.splitext(os.path.basename(args.pdf))[0]
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in safe_name)
    json_path = os.path.join(output_dir, f"bench_{safe_name}_{timestamp}.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to: {json_path}")


if __name__ == "__main__":
    main()
