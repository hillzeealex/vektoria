"""
Real semantic search on a PDF, end-to-end through Vektoria.

Ingests a PDF (extract → chunk → embed with a real multilingual model) and runs
French queries whose wording differs from the text — so a hit proves *semantic*
retrieval, not keyword matching.

Run:  python benchmarks/demo_pdf.py /path/to/document.pdf
Needs: pip install 'vektoria[ingest]' fastembed
"""

import sys
import tempfile

from vektoria import IndexManager
from vektoria.embedding import FastEmbedEmbedder
from vektoria.ingest import Ingestor

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 0.22 GB, FR-capable
QUERIES = [
    "Comment devient-on avocat en France ?",
    "Quel magistrat représente l'accusation au procès ?",
    "Quelles qualités faut-il pour réussir ses études de droit ?",
    "Qui rédige les actes authentiques pour une succession ?",
    "Qu'est-ce que la hiérarchie des normes ?",
]


def main(pdf_path: str):
    print(f"Loading embedder ({MODEL}) — first run downloads ~220 MB…")
    emb = FastEmbedEmbedder(MODEL)

    mgr = IndexManager(tempfile.mkdtemp())
    mgr.create_index("droit", dimension=emb.dimension)
    index = mgr.get("droit")

    with open(pdf_path, "rb") as f:
        data = f.read()
    result = Ingestor(emb).ingest(data, "droit_civil.pdf", index, max_words=120, overlap=20)
    print(f"Ingested: {result}\n")

    for q in QUERIES:
        hits = index.query(emb.embed_query(q).tolist(), top_k=2)
        print(f"❓ {q}")
        for h in hits:
            snippet = " ".join(h.metadata["text"].split())[:170]
            print(f"   [{h.score:.3f}] chunk {h.metadata['chunk']}: {snippet}…")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python benchmarks/demo_pdf.py /path/to/document.pdf")
    main(sys.argv[1])
