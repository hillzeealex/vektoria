"""
Honest head-to-head: Vektoria vs FAISS, hnswlib, Chroma, and Pinecone.

Every engine indexes the SAME vectors and answers the SAME queries on this
machine, so we compare algorithms — not network or hardware. We report:

  - build      : time to index all vectors (s)
  - p50 / p95  : query latency (ms)
  - recall@10  : fraction of the true top-10 returned (vs exact ground truth)

Exact engines (Vektoria, FAISS-Flat) have recall = 1.00 by construction; the
interesting question is their latency relative to the approximate engines.

Engines that aren't installed are skipped. Pinecone runs only if PINECONE_API_KEY
is set, and is flagged because its latency includes the network round-trip.

Run:  python benchmarks/bench_vs.py            # default N=10000
      N=50000 python benchmarks/bench_vs.py
"""

import os
import statistics
import tempfile
import time

import numpy as np

N = int(os.environ.get("N", "10000"))
DIM = 384
N_QUERIES = 200
K = 10


def unit_rows(n, dim, rng):
    m = rng.standard_normal((n, dim)).astype(np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def recall_at_k(got_ids, truth_ids):
    hits = sum(len(set(g) & set(t)) for g, t in zip(got_ids, truth_ids))
    return hits / (len(truth_ids) * K)


def summarize(name, build_s, latencies_ms, got_ids, truth_ids, note=""):
    latencies_ms = sorted(latencies_ms)
    return {
        "name": name,
        "build_s": build_s,
        "p50": statistics.median(latencies_ms),
        "p95": latencies_ms[int(len(latencies_ms) * 0.95)],
        "recall": recall_at_k(got_ids, truth_ids),
        "note": note,
    }


# ── adapters: each returns (build_s, latencies_ms, got_ids) ───────────
def run_vektoria(base, queries):
    from vektoria import Index

    tmp = tempfile.mkdtemp()
    idx = Index.create(f"{tmp}/idx", dimension=DIM)
    t = time.perf_counter()
    for s in range(0, len(base), 5000):
        idx.upsert([{"id": str(s + i), "values": base[s + i].tolist(), "metadata": {}}
                    for i in range(len(base[s:s + 5000]))])
    build = time.perf_counter() - t

    lat, got = [], []
    for q in queries:
        t = time.perf_counter()
        res = idx.query(q.tolist(), top_k=K)
        lat.append((time.perf_counter() - t) * 1000)
        got.append([int(m.id) for m in res])
    idx.close()
    return build, lat, got


def run_faiss_flat(base, queries):
    import faiss

    index = faiss.IndexFlatIP(DIM)
    t = time.perf_counter()
    index.add(base)
    build = time.perf_counter() - t
    lat, got = [], []
    for q in queries:
        t = time.perf_counter()
        _, ids = index.search(q.reshape(1, -1), K)
        lat.append((time.perf_counter() - t) * 1000)
        got.append(ids[0].tolist())
    return build, lat, got


def run_faiss_hnsw(base, queries):
    import faiss

    index = faiss.IndexHNSWFlat(DIM, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    t = time.perf_counter()
    index.add(base)
    build = time.perf_counter() - t
    index.hnsw.efSearch = 64
    lat, got = [], []
    for q in queries:
        t = time.perf_counter()
        _, ids = index.search(q.reshape(1, -1), K)
        lat.append((time.perf_counter() - t) * 1000)
        got.append(ids[0].tolist())
    return build, lat, got


def run_hnswlib(base, queries):
    import hnswlib

    index = hnswlib.Index(space="cosine", dim=DIM)
    index.init_index(max_elements=len(base), ef_construction=200, M=32)
    t = time.perf_counter()
    index.add_items(base, np.arange(len(base)))
    build = time.perf_counter() - t
    index.set_ef(64)
    lat, got = [], []
    for q in queries:
        t = time.perf_counter()
        labels, _ = index.knn_query(q.reshape(1, -1), k=K)
        lat.append((time.perf_counter() - t) * 1000)
        got.append(labels[0].tolist())
    return build, lat, got


def run_chroma(base, queries):
    import chromadb

    client = chromadb.Client()
    col = client.create_collection("bench", metadata={"hnsw:space": "cosine"})
    t = time.perf_counter()
    ids = [str(i) for i in range(len(base))]
    for s in range(0, len(base), 5000):
        col.add(ids=ids[s:s + 5000], embeddings=base[s:s + 5000].tolist())
    build = time.perf_counter() - t
    lat, got = [], []
    for q in queries:
        t = time.perf_counter()
        res = col.query(query_embeddings=[q.tolist()], n_results=K)
        lat.append((time.perf_counter() - t) * 1000)
        got.append([int(x) for x in res["ids"][0]])
    client.delete_collection("bench")
    return build, lat, got


def run_pinecone(base, queries):
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    name = "vektoria-bench"
    if pc.has_index(name):
        pc.delete_index(name)
    pc.create_index(name, dimension=DIM, metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"))
    while not pc.describe_index(name).status["ready"]:
        time.sleep(1)
    index = pc.Index(name)

    t = time.perf_counter()
    for s in range(0, len(base), 200):
        index.upsert(vectors=[(str(s + i), base[s + i].tolist())
                              for i in range(len(base[s:s + 200]))])
    # serverless is eventually consistent — wait until all vectors are queryable
    for _ in range(60):
        if index.describe_index_stats().get("total_vector_count", 0) >= len(base):
            break
        time.sleep(2)
    build = time.perf_counter() - t

    lat, got = [], []
    for q in queries:
        t = time.perf_counter()
        res = index.query(vector=q.tolist(), top_k=K)
        lat.append((time.perf_counter() - t) * 1000)
        got.append([int(m["id"]) for m in res["matches"]])
    pc.delete_index(name)
    return build, lat, got


ADAPTERS = [
    ("Vektoria (exact)", run_vektoria, ""),
    ("FAISS-Flat (exact)", run_faiss_flat, ""),
    ("FAISS-HNSW (approx)", run_faiss_hnsw, ""),
    ("hnswlib (approx)", run_hnswlib, ""),
    ("Chroma (approx)", run_chroma, ""),
    ("Pinecone (cloud)", run_pinecone, "incl. network"),
]


def main():
    rng = np.random.default_rng(7)
    base = unit_rows(N, DIM, rng)
    queries = unit_rows(N_QUERIES, DIM, rng)

    # Exact ground truth for recall.
    sims = queries @ base.T
    truth = [np.argpartition(row, -K)[-K:].tolist() for row in sims]

    print(f"\nDataset: N={N:,} vectors, dim={DIM}, {N_QUERIES} queries, recall@{K}\n")
    print(f"{'engine':<22} | {'build':>8} | {'p50':>8} | {'p95':>8} | {'recall@10':>9}")
    print("-" * 70)

    results = []
    for name, fn, note in ADAPTERS:
        if "Pinecone" in name and not os.environ.get("PINECONE_API_KEY"):
            print(f"{name:<22} | (skipped — set PINECONE_API_KEY)")
            continue
        try:
            build, lat, got = fn(base, queries)
        except Exception as e:  # noqa: BLE001
            print(f"{name:<22} | (skipped — {type(e).__name__}: {str(e)[:40]})")
            continue
        r = summarize(name, build, lat, got, truth, note)
        results.append(r)
        suffix = f"  {note}" if note else ""
        print(f"{name:<22} | {r['build_s']:>6.2f}s | {r['p50']:>6.2f}ms | "
              f"{r['p95']:>6.2f}ms | {r['recall']:>8.2%}{suffix}")

    print("\nExact engines (Vektoria, FAISS-Flat) → recall 1.00; compare their latency.")
    print("Approximate engines trade recall for speed at scale. Pinecone's latency")
    print("includes the network round-trip, so it measures the service, not the algorithm.")
    return results


if __name__ == "__main__":
    main()
