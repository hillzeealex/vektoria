"""
Honest benchmark of the Vektoria brute-force engine.

Measures upsert throughput and query latency (p50/p95) as the index grows.
Vektoria's search is *exact* (brute-force cosine), so recall is 100% by
construction — the interesting axis is how latency scales with size, which is
what tells you when an approximate (ANN) backend like TurboVec starts to matter.

Run:  python benchmarks/bench_engine.py
"""

import shutil
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from vektoria import Index

DIM = 384
SIZES = [1_000, 10_000, 50_000, 100_000]
N_QUERIES = 200
UPSERT_BATCH = 5_000


def _unit_rows(n, dim, rng):
    m = rng.standard_normal((n, dim)).astype(np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def bench(size, rng):
    tmp = Path(tempfile.mkdtemp())
    try:
        idx = Index.create(tmp / "bench", dimension=DIM)

        vectors = _unit_rows(size, DIM, rng)
        t0 = time.perf_counter()
        for start in range(0, size, UPSERT_BATCH):
            batch = vectors[start:start + UPSERT_BATCH]
            idx.upsert([
                {"id": f"v{start + i}", "values": batch[i].tolist(), "metadata": {}}
                for i in range(len(batch))
            ])
        upsert_s = time.perf_counter() - t0

        queries = _unit_rows(N_QUERIES, DIM, rng)
        lat = []
        for q in queries:
            t = time.perf_counter()
            idx.query(q.tolist(), top_k=10)
            lat.append((time.perf_counter() - t) * 1000)  # ms
        idx.close()

        lat.sort()
        return {
            "size": size,
            "upsert_per_s": size / upsert_s,
            "p50_ms": statistics.median(lat),
            "p95_ms": lat[int(len(lat) * 0.95)],
            "ram_mb": size * DIM * 4 / 1e6,  # float32 matrix footprint
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    rng = np.random.default_rng(42)
    print(f"Vektoria brute-force engine — dim={DIM}, exact cosine (recall=100%)\n")
    print(f"{'vectors':>9} | {'upsert/s':>9} | {'query p50':>9} | {'query p95':>9} | {'RAM':>8}")
    print("-" * 60)
    for size in SIZES:
        r = bench(size, rng)
        print(f"{r['size']:>9,} | {r['upsert_per_s']:>9,.0f} | "
              f"{r['p50_ms']:>7.2f}ms | {r['p95_ms']:>7.2f}ms | {r['ram_mb']:>6.0f}MB")
    print("\nExact search → 100% recall. Latency grows ~linearly with size;")
    print("that linear slope is what an ANN backend (e.g. TurboVec) would flatten.")


if __name__ == "__main__":
    main()
