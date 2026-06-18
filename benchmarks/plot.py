"""
Render Vektoria's benchmark charts (PNG) from the measured numbers.

The data below is what benchmarks/bench_engine.py and benchmarks/bench_vs.py
produced on a dev machine (dim=384, exact ground truth for recall). Re-run those
scripts to refresh the numbers, then update the constants here.

Run:  python benchmarks/plot.py   ->  writes assets/*.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Vektoria brand palette on dark ───────────────────────────────────
BG = "#05060f"
FG = "#e8ecff"
MUTED = "#8b93b8"
CYAN = "#00e5ff"
PURPLE = "#a371f7"
PINK = "#ff2e63"
GRID = "#1b2030"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": GRID,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.family": "DejaVu Sans", "axes.titleweight": "bold", "axes.titlesize": 13,
})

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(ASSETS, exist_ok=True)

# ── measured data ────────────────────────────────────────────────────
SIZES = [1_000, 10_000, 50_000, 100_000]
P50 = [0.08, 0.27, 1.25, 2.11]   # ms, exact query
P95 = [0.12, 0.44, 1.99, 2.69]

# head-to-head @ 10k vectors: (name, p50_ms, recall%, color, highlight)
ENGINES = [
    ("Vektoria",   0.25, 100.0, CYAN,   True),
    ("FAISS-Flat", 0.27, 100.0, "#7fd1ff", False),
    ("FAISS-HNSW", 0.15, 69.0,  PURPLE, False),
    ("hnswlib",    0.22, 65.0,  "#c4a0ff", False),
    ("Chroma",     0.71, 52.0,  MUTED,  False),
    ("Pinecone",   92.0, 98.0,  PINK,   False),
]


def _style(ax):
    ax.grid(True, alpha=0.4, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def chart_scaling():
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=140)
    ax.plot(SIZES, P50, "-o", color=CYAN, lw=2.4, ms=7, label="p50 latency")
    ax.plot(SIZES, P95, "--o", color=PURPLE, lw=2, ms=6, label="p95 latency")
    for x, y in zip(SIZES, P50):
        ax.annotate(f"{y:g} ms", (x, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", color=FG, fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks(SIZES)
    ax.set_xticklabels([f"{s//1000}k" for s in SIZES])
    ax.set_xlabel("vectors in index")
    ax.set_ylabel("query latency (ms)")
    ax.set_title("Exact query latency scales linearly  ·  100% recall", color=FG)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG)
    _style(ax)
    fig.tight_layout()
    fig.savefig(f"{ASSETS}/latency_scaling.png")
    plt.close(fig)


def chart_tradeoff():
    # (x-factor, y-offset) per label so the two exact engines don't collide
    LABELPOS = {
        "Vektoria":   (0.50, 2.3),
        "FAISS-Flat": (1.08, -3.4),
        "FAISS-HNSW": (1.18, 1.3),
        "hnswlib":    (1.10, -3.4),
        "Chroma":     (1.18, 1.3),
        "Pinecone":   (0.42, 1.8),
    }
    fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=140)
    for name, lat, rec, color, hi in ENGINES:
        ax.scatter(lat, rec, s=230 if hi else 130, color=color,
                   edgecolors=FG if hi else "none", linewidths=1.8, zorder=3)
        fx, dy = LABELPOS[name]
        ax.annotate(name, (lat, rec), xytext=(lat * fx, rec + dy),
                    color=FG, fontsize=10, fontweight="bold" if hi else "normal")
    ax.set_xscale("log")
    ax.set_xlabel("query latency (ms, log)  —  lower is better →")
    ax.set_ylabel("recall@10 (%)  —  higher is better ↑")
    ax.set_ylim(45, 104)
    ax.set_title("Recall vs latency  ·  Vektoria ties FAISS-Flat, beats the cloud hop", color=FG)
    ax.axhspan(99, 104, color=CYAN, alpha=0.05)
    _style(ax)
    fig.tight_layout()
    fig.savefig(f"{ASSETS}/recall_vs_latency.png")
    plt.close(fig)


def chart_local_bars():
    local = [(n, lat, rec, c) for (n, lat, rec, c, _) in ENGINES if n != "Pinecone"]
    names = [n for n, *_ in local]
    lats = [lat for _, lat, *_ in local]
    recs = [rec for _, _, rec, _ in local]
    colors = [c for *_, c in local]

    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=140)
    bars = ax.bar(names, lats, color=colors, width=0.62, zorder=3)
    for bar, rec in zip(bars, recs):
        ax.annotate(f"recall {rec:.0f}%", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    textcoords="offset points", xytext=(0, 6), ha="center",
                    color=FG, fontsize=9)
    ax.set_ylabel("query latency (ms)")
    ax.set_title("Local engines @ 10k vectors  ·  latency & recall", color=FG)
    ax.set_ylim(0, max(lats) * 1.35)
    _style(ax)
    fig.tight_layout()
    fig.savefig(f"{ASSETS}/local_engines.png")
    plt.close(fig)


if __name__ == "__main__":
    chart_scaling()
    chart_tradeoff()
    chart_local_bars()
    print(f"Wrote charts to {os.path.normpath(ASSETS)}/")
