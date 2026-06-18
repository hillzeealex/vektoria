"""
Render Vektoria's benchmark charts (PNG) from the measured numbers.

Design goals (data-analysis hygiene): neutral descriptive titles, colour encodes
the engine *category* (exact / approximate / cloud), Vektoria is highlighted by a
ring rather than a louder colour, direct labels instead of legends where possible,
and a source line so the numbers are traceable. The data is what
benchmarks/bench_engine.py and benchmarks/bench_vs.py produced (dim=384, recall@10
vs exact ground truth). Re-run those, then update the constants here.

Run:  python benchmarks/plot.py   ->  writes assets/*.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

# ── palette ──────────────────────────────────────────────────────────
BG = "#0a0c16"
PANEL = "#0e1120"
FG = "#e8ecff"
MUTED = "#7b84a8"
GRID = "#1c2236"
EXACT = "#22d3ee"     # cyan  — exact engines
APPROX = "#a78bfa"    # violet — approximate (ANN) engines
CLOUD = "#fb7185"     # rose  — hosted cloud service

plt.rcParams.update({
    "figure.facecolor": BG, "savefig.facecolor": BG, "axes.facecolor": PANEL,
    "text.color": FG, "axes.labelcolor": MUTED, "axes.edgecolor": GRID,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlepad": 14,
    "axes.labelsize": 11, "figure.dpi": 160,
})

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(ASSETS, exist_ok=True)
SOURCE = "dim 384  ·  200 queries  ·  recall@10 vs exact ground truth  ·  single machine"

# ── measured data ────────────────────────────────────────────────────
SIZES = [1_000, 10_000, 50_000, 100_000]
P50 = [0.08, 0.27, 1.25, 2.11]   # ms, exact query
P95 = [0.12, 0.44, 1.99, 2.69]

# head-to-head @ 10k vectors: name, p50_ms, recall%, category, is_vektoria
ENGINES = [
    ("Vektoria",   0.25, 100.0, "exact",  True),
    ("FAISS-Flat", 0.27, 100.0, "exact",  False),
    ("FAISS-HNSW", 0.15, 69.0,  "approx", False),
    ("hnswlib",    0.22, 65.0,  "approx", False),
    ("Chroma",     0.71, 52.0,  "approx", False),
    ("Pinecone",   92.0, 98.0,  "cloud",  False),
]
CAT_COLOR = {"exact": EXACT, "approx": APPROX, "cloud": CLOUD}
CAT_LABEL = {"exact": "Exact (100% recall)", "approx": "Approximate (ANN)", "cloud": "Hosted cloud"}


def _finish(fig, ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, alpha=0.35, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0.055, 1, 1))           # reserve a strip for the source
    fig.text(0.5, 0.018, SOURCE, ha="center", color=MUTED, fontsize=8.5)


def chart_tradeoff():
    fig, ax = plt.subplots(figsize=(7.6, 5.0))

    # faint band marking the exact (100% recall) frontier
    ax.axhspan(99, 103, color=EXACT, alpha=0.06, zorder=0)

    labelpos = {  # (x-factor, y-offset) to keep labels off the markers
        "Vektoria":   (0.52, 2.6), "FAISS-Flat": (1.10, -3.6),
        "FAISS-HNSW": (1.16, 1.6), "hnswlib":    (1.05, -3.8),
        "Chroma":     (1.16, 1.6), "Pinecone":   (0.40, 2.2),
    }
    for name, lat, rec, cat, vk in ENGINES:
        color = CAT_COLOR[cat]
        ax.scatter(lat, rec, s=300 if vk else 150, color=color, zorder=3,
                   edgecolors=FG if vk else "none", linewidths=2.2)
        fx, dy = labelpos[name]
        ax.annotate(name, (lat, rec), xytext=(lat * fx, rec + dy),
                    color=FG, fontsize=10.5, fontweight="bold" if vk else "normal", zorder=4)

    ax.set_xscale("log")
    ax.set_xlabel("query latency  (ms, log scale) — lower is better →")
    ax.set_ylabel("recall@10  (%) — higher is better ↑")
    ax.set_ylim(45, 105)
    ax.set_title("Recall vs. query latency — 10,000 vectors", loc="left")

    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=9,
                      markerfacecolor=CAT_COLOR[c], markeredgecolor="none", label=CAT_LABEL[c])
               for c in ("exact", "approx", "cloud")]
    ax.legend(handles=handles, loc="lower right", frameon=True, facecolor=PANEL,
              edgecolor=GRID, labelcolor=FG, fontsize=9.5)
    _finish(fig, ax)
    fig.savefig(f"{ASSETS}/recall_vs_latency.png")
    plt.close(fig)


def chart_local_bars():
    local = sorted([e for e in ENGINES if e[3] != "cloud"], key=lambda e: e[1])
    names = [e[0] for e in local]
    lats = [e[1] for e in local]
    recs = [e[2] for e in local]
    colors = [CAT_COLOR[e[3]] for e in local]
    edges = [FG if e[4] else "none" for e in local]

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    bars = ax.bar(names, lats, color=colors, edgecolor=edges, linewidth=2.0, width=0.6, zorder=3)
    for bar, lat, rec in zip(bars, lats, recs):
        x = bar.get_x() + bar.get_width() / 2
        ax.annotate(f"{lat:.2f} ms", (x, bar.get_height()), textcoords="offset points",
                    xytext=(0, 5), ha="center", color=FG, fontsize=9.5, fontweight="bold")
        ax.annotate(f"recall {rec:.0f}%", (x, 0), textcoords="offset points",
                    xytext=(0, 6), ha="center", color=BG, fontsize=8.5, fontweight="bold")
    ax.set_ylabel("query latency  (ms)")
    ax.set_ylim(0, max(lats) * 1.30)
    ax.set_title("Local engines — latency & recall @ 10k (sorted by speed)", loc="left")

    handles = [Line2D([0], [0], marker="s", linestyle="", markersize=10,
                      markerfacecolor=CAT_COLOR[c], markeredgecolor="none", label=CAT_LABEL[c])
               for c in ("exact", "approx")]
    ax.legend(handles=handles, loc="upper left", frameon=True, facecolor=PANEL,
              edgecolor=GRID, labelcolor=FG, fontsize=9.5)
    _finish(fig, ax)
    fig.savefig(f"{ASSETS}/local_engines.png")
    plt.close(fig)


def chart_scaling():
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.fill_between(SIZES, P50, P95, color=EXACT, alpha=0.10, zorder=1)
    ax.plot(SIZES, P95, "--o", color=APPROX, lw=1.8, ms=5, label="p95", zorder=3)
    ax.plot(SIZES, P50, "-o", color=EXACT, lw=2.6, ms=7, label="p50", zorder=4)
    for x, y in zip(SIZES, P50):
        ax.annotate(f"{y:g} ms", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", color=FG, fontsize=9.5, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks(SIZES)
    ax.set_xticklabels([f"{s // 1000}k" for s in SIZES])
    ax.set_xlabel("vectors in index")
    ax.set_ylabel("query latency  (ms)")
    ax.set_ylim(0, max(P95) * 1.18)
    ax.set_title("Exact query latency vs. index size  ·  100% recall", loc="left")
    ax.legend(loc="upper left", frameon=True, facecolor=PANEL, edgecolor=GRID,
              labelcolor=FG, fontsize=9.5)
    _finish(fig, ax)
    fig.savefig(f"{ASSETS}/latency_scaling.png")
    plt.close(fig)


if __name__ == "__main__":
    chart_tradeoff()
    chart_local_bars()
    chart_scaling()
    print(f"Wrote charts to {os.path.normpath(ASSETS)}/")
