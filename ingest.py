#!/usr/bin/env python3
"""
CLI tool to ingest PDFs into the vector store.

Usage:
  python ingest.py /path/to/document.pdf
  python ingest.py /path/to/folder/      # all PDFs in folder
  python ingest.py --list                 # list indexed documents
  python ingest.py --delete "Document n°1"

Environment variables:
  SVS_DATA_DIR       — storage directory (default: ./data)
  SVS_EMBED_BACKEND  — "sentence-transformers", "ollama", or "numpy" (default: numpy)
  SVS_EMBED_MODEL    — model name (default: intfloat/multilingual-e5-large)
  SVS_EMBED_DIM      — dimension for numpy backend (default: 384)
"""

import os
import sys
import time
import glob
import argparse

from pdf_extractor import PdfExtractor
from chunker import SemanticChunker
from embedder import LocalEmbedder
from vector_store import VectorStore


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into SwissVectorStore")
    parser.add_argument("path", nargs="?", help="PDF file or directory to ingest")
    parser.add_argument("--list", action="store_true", help="List indexed documents")
    parser.add_argument("--delete", type=str, help="Delete a document by source name")
    parser.add_argument("--search", type=str, help="Search the store with a query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--data-dir", default=os.environ.get("SVS_DATA_DIR", "./data"))
    parser.add_argument("--backend", default=os.environ.get("SVS_EMBED_BACKEND", "numpy"))
    parser.add_argument("--model", default=os.environ.get("SVS_EMBED_MODEL", "intfloat/multilingual-e5-large"))
    parser.add_argument("--dim", type=int, default=int(os.environ.get("SVS_EMBED_DIM", "384")))

    args = parser.parse_args()
    store = VectorStore(args.data_dir)

    if args.list:
        sources = store.list_sources()
        if not sources:
            print("No documents indexed.")
        else:
            print(f"{'Source':<55} {'Chunks':>7} {'Words':>8}")
            print("-" * 72)
            for s in sources:
                print(f"{s['source'][:54]:<55} {s['chunks']:>7} {s['words']:>8}")
            print(f"\nTotal: {store.count} chunks")
        store.close()
        return

    if args.delete:
        deleted = store.delete_source(args.delete)
        print(f"Deleted {deleted} chunks from '{args.delete}'")
        store.close()
        return

    embedder = LocalEmbedder(
        backend=args.backend,
        model_name=args.model,
        dimension=args.dim,
    )

    if args.search:
        if store.count == 0:
            print("No documents indexed. Ingest a PDF first.")
            store.close()
            return

        query_vec = embedder.embed_query(args.search)
        results = store.search(query_vec, top_k=args.top_k)

        print(f"\nQuery: '{args.search}'")
        print(f"Results ({len(results)}):\n")
        for i, r in enumerate(results, 1):
            path = " > ".join(r.heading_path[-2:])
            print(f"  {i}. [{r.score:.4f}] {path[:70]}")
            print(f"     Source: {r.source} | Pages: {r.page_start+1}-{r.page_end+1} | {r.word_count}w")
            preview = r.text.replace("\n", " ")[:150]
            print(f"     {preview}...")
            print()

        store.close()
        return

    if not args.path:
        parser.print_help()
        store.close()
        return

    # Collect PDF files
    if os.path.isdir(args.path):
        pdf_files = sorted(glob.glob(os.path.join(args.path, "*.pdf")))
        if not pdf_files:
            print(f"No PDF files found in {args.path}")
            store.close()
            return
    elif os.path.isfile(args.path):
        pdf_files = [args.path]
    else:
        print(f"Path not found: {args.path}")
        store.close()
        return

    extractor = PdfExtractor()
    chunker = SemanticChunker(max_words=800, min_words=50)

    for pdf_path in pdf_files:
        print(f"\n{'=' * 60}")
        print(f"Ingesting: {os.path.basename(pdf_path)}")
        print(f"{'=' * 60}")

        t0 = time.time()

        # Extract
        t = time.time()
        doc = extractor.extract(pdf_path)
        t_extract = time.time() - t
        print(f"  Extract:  {t_extract:.2f}s → {doc.page_count} pages, {len(doc.sections)} sections")

        # Chunk
        t = time.time()
        chunks = chunker.chunk(doc)
        t_chunk = time.time() - t
        print(f"  Chunk:    {t_chunk:.2f}s → {len(chunks)} chunks")

        if not chunks:
            print("  SKIP: no content extracted")
            continue

        # Embed
        t = time.time()
        embedded = embedder.embed_chunks(chunks)
        t_embed = time.time() - t
        print(f"  Embed:    {t_embed:.2f}s → {len(embedded)} vectors")

        # Store
        t = time.time()
        store.add(embedded)
        t_store = time.time() - t
        print(f"  Store:    {t_store:.2f}s")

        total = time.time() - t0
        print(f"  TOTAL:    {total:.2f}s")
        print(f"  Title:    {doc.title}")

    print(f"\n{'=' * 60}")
    print(f"Store: {store.count} total chunks in {args.data_dir}")
    sources = store.list_sources()
    for s in sources:
        print(f"  {s['source'][:50]:50s} | {s['chunks']} chunks")
    print(f"{'=' * 60}")

    store.close()


if __name__ == "__main__":
    main()
