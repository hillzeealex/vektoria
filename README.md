<div align="center">

# ⬢ Vektoria

### The European vector database. Self-hosted. Sovereign. Yours.

**A Pinecone alternative you run on your own server — in Europe.**
Your data never touches a US cloud. Ever.

[![License: MIT](https://img.shields.io/badge/License-MIT-00e5ff.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-hillzeealex%2Fvektoria-1f6feb.svg)](https://github.com/hillzeealex/vektoria)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-1f6feb.svg)](https://www.python.org/)
[![Status: alpha](https://img.shields.io/badge/Status-alpha-a371f7.svg)](#roadmap)
[![Made in 🇨🇭](https://img.shields.io/badge/Hosted_in-Switzerland-ff2e63.svg)](#why-sovereignty)

</div>

---

## Why Vektoria

Most RAG stacks send your documents to US-based services — Pinecone, Weaviate Cloud, OpenAI. Every vector you store transits servers subject to the [CLOUD Act](https://en.wikipedia.org/wiki/CLOUD_Act), which lets US authorities access data held by American companies **even when the servers sit in Europe**.

For a European or Swiss company handling legal, financial, medical, or strategic documents, that's a compliance problem (GDPR / nLPD) and a confidentiality problem.

**Vektoria is the answer: a vector database you host yourself.** Deploy it on your own VPS — Swiss, French, German, wherever — and your data stays under your control. It's the kind of engine Pinecone runs internally, packaged as something *you* own.

> **Vektoria = Pinecone, but you run it yourself, in Europe.**

## Vektoria vs. Pinecone

| | Pinecone | **Vektoria** |
|---|---|---|
| Hosting | US cloud (managed) | **Your own server** (self-hosted) |
| Data sovereignty | ❌ Subject to CLOUD Act | ✅ **Stays on your infra** |
| Vector search | ✅ | ✅ cosine (brute-force) |
| Keyword + **hybrid** search | ❌ not native | ✅ **built-in BM25 + hybrid** |
| Document ingestion (PDF→vectors) | ❌ bring your own | ✅ **server-side, optional** |
| Metadata filters | ✅ | ✅ |
| Right to erasure / export (GDPR) | partial | ✅ **real delete + export** |
| License | Proprietary | **MIT, open source** |
| Cost | Per-vector pricing | **Free — your hardware** |

## Features

- 🧠 **Vector search** — exact cosine similarity, no approximation surprises.
- 🔎 **Hybrid search** — combine semantic vectors with BM25 keyword scoring (a differentiator Pinecone lacks natively).
- 🏷️ **Metadata filters** — exact match, list membership, and more.
- 📚 **Multi-index** — many independent indexes in one instance.
- 📄 **Document ingestion** *(roadmap)* — send a PDF/DOCX, get it extracted, chunked, embedded, and stored, all server-side.
- 🗑️ **GDPR-grade** — real deletion (right to erasure) and full export (portability).
- 🔒 **Self-hosted** — one Docker command, your server, your rules.

## Quickstart (Python core)

> Vektoria is in **active development**. The core engine below works today; the REST API and Docker image are on the [roadmap](#roadmap).

```python
from vektoria import IndexManager

# One instance can hold many indexes
mgr = IndexManager("./data")
mgr.create_index("contracts", dimension=384)

index = mgr.get("contracts")

# Bring your own vectors (id, values, metadata — text lives in metadata)
index.upsert([
    {"id": "c1", "values": embed("clause de non-concurrence"),
     "metadata": {"text": "clause de non-concurrence", "source": "contrat.pdf"}},
    {"id": "c2", "values": embed("clause de confidentialité"),
     "metadata": {"text": "clause de confidentialité", "source": "contrat.pdf"}},
])

# Pure vector search
hits = index.query(embed("non-concurrence"), top_k=5)

# Hybrid search (vector + BM25 keywords)
hits = index.query(
    embed("non-concurrence"), top_k=5,
    hybrid=True, alpha=0.5, text="non-concurrence",
)

# Metadata filter
hits = index.query(embed("..."), top_k=5, filter={"source": "contrat.pdf"})

for h in hits:
    print(f"{h.score:.3f}  {h.metadata['text']}")

# GDPR: real erasure + export
index.delete(filter={"source": "contrat.pdf"})
dump = index.export()
```

## Architecture

```
        ┌──────────────────────────────────────────────┐
        │  CLIENT                                       │
        │   raw vectors  ──or──  documents (PDF/DOCX…)  │
        └───────────────────────┬──────────────────────┘
                                │  HTTP(S)
        ┌───────────────────────▼──────────────────────┐
        │  YOUR SERVER (Switzerland / EU)               │
        │   REST API (roadmap) · optional API key       │
        │   Ingestion: extract → chunk → embed          │
        │   IndexManager (LRU cache of open indexes)    │
        │   Per-index storage: SQLite (vectors as blobs)│
        │   Search: cosine + BM25 + hybrid              │
        └───────────────────────────────────────────────┘
```

One index = one directory with a single SQLite database as the source of truth. Vectors are L2-normalized on write and stored as float32 blobs alongside their metadata. No data ever leaves the box.

## Self-host on your VPS

> **Current state:** Vektoria ships today as a Python package + engine. The
> one-command Docker image and REST API are on the [roadmap](#roadmap) — until
> then you run it as a library on your own server.

**1. Get a VPS in Europe / Switzerland.** Any provider works — Infomaniak or
Exoscale (🇨🇭), OVHcloud or Scaleway (🇪🇺). Minimum ~4 GB RAM; 8–16 GB if you use
server-side document embedding.

**2. Install.**
```bash
ssh user@your-vps
sudo apt update && sudo apt install -y python3.11 python3.11-venv git
git clone https://github.com/hillzeealex/vektoria.git
cd vektoria
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m pytest tests/vektoria/ -q   # sanity check: 22 passing
```

**3. Use it — your data stays on the box.**
```python
from vektoria import IndexManager

mgr = IndexManager("/var/lib/vektoria")   # data dir on the VPS disk
mgr.create_index("docs", dimension=384)
mgr.get("docs").upsert([{"id": "1", "values": embed("…"), "metadata": {"text": "…"}}])
```

**4. Harden it (GDPR / art. 32).**
- Enable **full-disk encryption (LUKS)** on the VPS for data at rest.
- Keep the data directory on the VPS only — nothing leaves the machine.

**Coming soon — one-command deploy:**
```bash
docker compose up -d                      # REST API + optional API key (roadmap)
ssh -L 8000:localhost:8000 user@your-vps  # reach the dashboard locally, no public port
```

## Roadmap

- [x] **Core engine** — multi-index storage, cosine + BM25 + hybrid search, metadata filters, real delete, export, LRU cache
- [ ] **REST API** — Pinecone-shaped `/v1/indexes` endpoints, optional API-key auth
- [ ] **Document ingestion** — `POST /ingest` (PDF/DOCX → vectors, server-side)
- [ ] **Dashboard** — read-only console + search playground (via SSH tunnel)
- [ ] **Docker** — one-command self-host deployment
- [ ] **Scale backend** — optional [TurboVec](https://github.com/RyanCodrai/turbovec) ANN engine for large indexes

## Why sovereignty

Switzerland has its own data-protection law (nLPD/revFADP), benefits from an EU adequacy decision, and is **not subject to the US CLOUD Act**. Hosting in Switzerland is, for many European customers, an even stronger guarantee than hosting inside the EU. Self-hosting Vektoria takes it further: there's no processor in the loop at all — you are the sole controller of your data.

## License

[MIT](LICENSE) — free to use, modify, and self-host.

<div align="center">
<sub>Built for teams who refuse to send their data across the Atlantic.</sub>
</div>
