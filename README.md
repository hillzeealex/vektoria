# EuropeanVectorStore

RAG pipeline 100% souverain -zéro API cloud, zéro fuite de données.

Conçu pour tourner sur un VPS européen. Toutes les données restent sur votre infrastructure.

## Pourquoi la souveraineté des données ?

Aujourd'hui, la majorité des solutions RAG (Retrieval-Augmented Generation) reposent sur des services cloud américains : LlamaParse pour l'extraction, OpenAI pour les embeddings, Pinecone ou Weaviate pour le stockage vectoriel. **Chaque document que vous envoyez à ces services transite par des serveurs soumis au [CLOUD Act](https://en.wikipedia.org/wiki/CLOUD_Act)**, une loi fédérale américaine qui permet aux autorités US d'accéder aux données stockées par des entreprises américaines -même si les serveurs sont physiquement en Europe.

Pour une entreprise suisse ou européenne, cela pose un problème fondamental :

- **Conformité légale** -La LPD (Loi fédérale sur la protection des données, Suisse) et le RGPD (UE) exigent un contrôle strict sur le transfert de données personnelles hors de l'espace européen. Utiliser OpenAI ou Pinecone pour traiter des documents internes peut constituer une violation.
- **Confidentialité** -Des documents juridiques, financiers, médicaux ou stratégiques envoyés à un service cloud US sont potentiellement accessibles à des tiers. Même avec du chiffrement en transit, le fournisseur a accès aux données en clair lors du traitement.
- **Dépendance (vendor lock-in)** -Les APIs propriétaires changent, augmentent leurs prix, ou ferment. Votre pipeline RAG ne devrait pas dépendre d'une entreprise dont vous ne contrôlez ni la roadmap, ni les conditions d'utilisation.

### La solution : tout en local

EuropeanVectorStore est une alternative open-source qui remplace **l'intégralité de la chaîne RAG** par des composants locaux :

| Étape | Solution cloud (US) | EuropeanVectorStore (local) |
|-------|-------------------|--------------------------|
| Extraction PDF | LlamaParse (API) | PyMuPDF (local) |
| Embeddings | OpenAI ada-002 (API) | sentence-transformers / Ollama (local) |
| Stockage vectoriel | Pinecone / Weaviate (cloud) | numpy + SQLite (local) |
| Recherche par mots-clés | Elastic Cloud (cloud) | BM25 from scratch (local) |
| Coût par document | ~$0.30–1.00 | 0 CHF |
| Données sortent du serveur | Oui | **Non** |

**Aucune donnée ne quitte votre serveur. Jamais.**

### Benchmark (PDF juridique de 105 pages)

```
╔═══════════════════════╦═══════════════╦═════════════════════╗
║ Metric                ║ EVS (VPS) ║ Cloud (estimated)   ║
╠═══════════════════════╬═══════════════╬═════════════════════╣
║ Extract 105 pages     ║         0.90s ║ ~30-90s (LlamaParse)║
║ Search latency p50    ║        0.09ms ║ ~50-200ms (Pinecone)║
║ Embed throughput      ║   5287 vec/s  ║ ~100-500 vec/s (API)║
║ Cost per document     ║         0 CHF ║       ~$0.30-1.00   ║
║ Data sovereignty      ║          100% ║                  0% ║
║ Vendor lock-in        ║    None (OSS) ║                High ║
╚═══════════════════════╩═══════════════╩═════════════════════╝
```

## Architecture

```
Document (PDF, DOCX, TXT, MD, HTML, CSV)
  │
  ├─ ingestion/         → Détection du format, routage vers l'extracteur
  │
  ├─ pdf_extractor/     → Markdown structuré (analyse de police + numérotation)
  │
  ├─ chunker/           → 3 stratégies de découpage
  │   ├── semantic      → Par sections (documents structurés)
  │   ├── recursive     → Par séparateurs (texte brut)
  │   └── similarity    → Par similarité sémantique
  │
  ├─ embedder/          → Embedding local (sentence-transformers ou Ollama)
  │
  ├─ vector_store/      → Stockage + recherche vectorielle
  │   ├── Cosine similarity (numpy)
  │   ├── BM25 keyword search (from scratch, French-aware)
  │   ├── Hybrid search (vector + BM25, alpha configurable)
  │   └── Metadata filters (source, tags, pages, custom fields)
  │
  └─ api/               → FastAPI REST API
```

## Installation

```bash
git clone https://github.com/hillzeealex/european-vector-store.git
cd european-vector-store
pip install -r requirements.txt
```

## Utilisation

### CLI

```bash
# Ingérer un PDF
python ingest.py /path/to/document.pdf

# Ingérer un dossier complet
python ingest.py /path/to/folder/

# Lister les documents indexés
python ingest.py --list

# Rechercher
python ingest.py --search "votre question ici"

# Supprimer un document
python ingest.py --delete "Nom du document"
```

### API REST

```bash
# Lancer le serveur
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

**Endpoints :**

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `POST` | `/ingest` | Upload et indexation d'un document |
| `POST` | `/query` | Recherche sémantique |
| `GET` | `/documents` | Lister les documents indexés |
| `DELETE` | `/documents/{source}` | Supprimer un document |
| `GET` | `/health` | Status du service |

**Exemple de requête :**

```bash
# Ingérer un PDF
curl -X POST http://localhost:8000/ingest \
  -F "file=@document.pdf"

# Rechercher
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Quelles sont les sanctions ?", "top_k": 5}'
```

### Docker

```bash
docker compose up -d

# Télécharger le modèle d'embedding
docker exec european-vector-store-ollama-1 ollama pull nomic-embed-text
```

## Modules

### pdf_extractor

Extraction PDF vers Markdown structuré via PyMuPDF. Deux stratégies de détection de hiérarchie :

1. **Font analysis** -taille de police + gras pour détecter les niveaux de titres
2. **Numbering analysis** -`1.` → h1, `1.1.` → h2, `1.1.1.` → h3 (quand la police est uniforme)

Détection automatique des pages de table des matières (TOC).

### chunker

Trois stratégies de découpage :

- **SemanticChunker** -découpe par sections du document (respecte la hiérarchie)
- **RecursiveChunker** -découpe récursive par séparateurs (`\n\n`, `\n`, `. `, ` `)
- **SimilarityChunker** -regroupe les paragraphes par similarité sémantique

```python
from chunker import create_chunker

chunker = create_chunker("semantic", max_words=800)
# ou
chunker = create_chunker("recursive", max_words=500, overlap_words=50)
```

### embedder

Embedding local avec trois backends :

- **sentence-transformers** -modèles HuggingFace (recommandé : `multilingual-e5-large`)
- **ollama** -via l'API REST Ollama (modèle `nomic-embed-text`)
- **numpy** -pseudo-embeddings pour les tests (aucune dépendance ML)

### vector_store

Vector store custom :

- **Stockage** : vecteurs en fichiers numpy, métadonnées en SQLite
- **Recherche vectorielle** : cosine similarity (brute-force, < 1ms pour 30k vecteurs)
- **Recherche par mots-clés** : BM25 from scratch, tokenisation française avec stopwords
- **Recherche hybride** : combinaison vector + BM25 avec alpha configurable
- **Filtres** : source, tags, pages, métadonnées custom

```python
from vector_store import VectorStore

store = VectorStore("./data")

# Recherche vectorielle
results = store.search(query_vector, top_k=5)

# Recherche hybride (vector + keywords)
results = store.hybrid_search(
    query_text="excès de vitesse",
    query_vector=query_vec,
    alpha=0.7  # 70% vector, 30% keywords
)

# Avec filtres
results = store.search(query_vector, filters={
    "source": "Droit pénal LCR",
    "tags": ["important"],
})
```

## Benchmarks

```bash
python benchmarks/run_benchmark.py /path/to/document.pdf
```

Mesure 7 métriques : extraction speed, chunking quality, embedding throughput, search latency (p50/p95/p99), search precision (recall@k), storage efficiency, memory usage.

## Configuration

Variables d'environnement :

| Variable | Default | Description |
|----------|---------|-------------|
| `SVS_DATA_DIR` | `./data` | Répertoire de stockage |
| `SVS_EMBED_BACKEND` | `numpy` | Backend d'embedding |
| `SVS_EMBED_MODEL` | `intfloat/multilingual-e5-large` | Modèle d'embedding |
| `SVS_EMBED_DIM` | `384` | Dimension des vecteurs |
| `SVS_OLLAMA_URL` | `http://localhost:11434` | URL Ollama |

## Déploiement sur VPS

### Prérequis

| Spec | Minimum | Recommandé |
|------|---------|------------|
| RAM | 8 Go | 16 Go |
| CPU | 4 cores | 10 cores |
| Stockage | 50 Go | 150 Go |
| OS | Ubuntu 22.04+ | Ubuntu 24.04 |

Avec 16 Go RAM : modèle d'embedding + Mistral 7B (LLM) + vector store pour ~100k documents.

### Installation sur VPS (Ubuntu)

```bash
# 1. Se connecter au VPS
ssh user@votre-vps.ch

# 2. Installer les dépendances système
sudo apt update && sudo apt install -y python3.11 python3.11-venv git

# 3. Cloner le projet
git clone https://github.com/hillzeealex/european-vector-store.git
cd european-vector-store

# 4. Créer un environnement virtuel
python3.11 -m venv .venv
source .venv/bin/activate

# 5. Installer les dépendances Python
pip install -r requirements.txt

# 6. (Optionnel) Installer Ollama pour les embeddings
curl -fsSL https://ollama.com/install.sh | sh
ollama pull nomic-embed-text
```

### Lancer l'API en production

```bash
# Configurer les variables d'environnement
export SVS_DATA_DIR=/var/lib/european-vector-store/data
export SVS_EMBED_BACKEND=ollama        # ou "sentence-transformers"
export SVS_EMBED_MODEL=nomic-embed-text
export SVS_EMBED_DIM=768

# Lancer avec uvicorn (production)
uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 4
```

### Systemd (démarrage automatique)

```bash
sudo tee /etc/systemd/system/european-vector-store.service << 'EOF'
[Unit]
Description=EuropeanVectorStore API
After=network.target ollama.service

[Service]
User=www-data
WorkingDirectory=/opt/european-vector-store
Environment=SVS_DATA_DIR=/var/lib/european-vector-store/data
Environment=SVS_EMBED_BACKEND=ollama
Environment=SVS_EMBED_MODEL=nomic-embed-text
Environment=SVS_EMBED_DIM=768
ExecStart=/opt/european-vector-store/.venv/bin/uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable --now european-vector-store
```

### Avec Docker

```bash
docker compose up -d
docker exec european-vector-store-ollama-1 ollama pull nomic-embed-text
```

L'API est accessible sur `http://votre-vps:8000`.

---

## Intégrer l'API dans votre projet

EuropeanVectorStore expose une API REST que n'importe quel projet peut consommer. Voici un exemple complet d'intégration dans une app Next.js / Node.js (ou tout autre backend).

### Exemple : Route API Next.js qui utilise EuropeanVectorStore

```typescript
// app/api/search/route.ts (Next.js App Router)

const SVS_URL = process.env.SVS_URL || "http://votre-vps:8000";

export async function POST(request: Request) {
  const { question } = await request.json();

  // 1. Recherche dans EuropeanVectorStore
  const response = await fetch(`${SVS_URL}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k: 5 }),
  });

  const { results } = await response.json();

  // 2. Utiliser les résultats comme contexte pour un LLM
  const context = results.map((r: any) => r.text).join("\n\n---\n\n");

  // 3. Appeler votre LLM local (Ollama) ou autre
  const llmResponse = await fetch("http://votre-vps:11434/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "mistral",
      prompt: `Contexte:\n${context}\n\nQuestion: ${question}\n\nRéponse:`,
      stream: false,
    }),
  });

  const { response: answer } = await llmResponse.json();
  return Response.json({ answer, sources: results });
}
```

### Exemple : Script Python

```python
import requests

SVS_URL = "http://votre-vps:8000"

# Ingérer un document
with open("mon_document.pdf", "rb") as f:
    resp = requests.post(f"{SVS_URL}/ingest", files={"file": f})
    print(resp.json())  # {"status": "ok", "chunks": 42, "source": "mon_document.pdf"}

# Rechercher
resp = requests.post(f"{SVS_URL}/query", json={
    "question": "Quelles sont les obligations contractuelles ?",
    "top_k": 5,
})
for result in resp.json()["results"]:
    print(f"[{result['score']:.2f}] {result['text'][:100]}...")
```

### Exemple : cURL (tout langage)

```bash
# Ingérer
curl -X POST http://votre-vps:8000/ingest -F "file=@contrat.pdf"

# Rechercher
curl -X POST http://votre-vps:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Clause de non-concurrence", "top_k": 3}'

# Lister les documents
curl http://votre-vps:8000/documents

# Supprimer un document
curl -X DELETE http://votre-vps:8000/documents/contrat.pdf

# Health check
curl http://votre-vps:8000/health
```

---

## Tests

```bash
python tests/test_full_pipeline.py
```

30 tests couvrant l'ensemble du pipeline : extraction, chunking, embedding, vector store, intégration.

## Licence

MIT
