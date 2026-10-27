"""Render docs/img/bench-retrieval.png from reports/llmops/latest.json.

A plain SVG bar chart (hit@1, hit@5, MRR@5 per retrieval strategy) rendered
to PNG with Edge headless, so the vitrine and the video show the measured
numbers and nothing else.

Usage (Windows): python scripts/render_bench_chart.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "llmops" / "latest.json"
OUT = ROOT / "docs" / "img" / "bench-retrieval.png"
WORK = ROOT / "data" / "promo"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

LABELS = {
    "dense": "Sens seul (dense bge-m3)",
    "bm25": "Mots exacts seuls (BM25)",
    "hybrid_rrf": "Hybride (fusion RRF)",
    "hybrid_rrf_rerank": "Hybride + reranking (production)",
}
SERIES = [
    ("hit@1", "Bonne page en 1re position", "#9aa7b8"),
    ("hit@5", "Bonne page dans les 5", "#d9822b"),
    ("mrr@5", "MRR@5", "#3d5a80"),
]


def main() -> int:
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    ret = rep["retrieval"]
    n = ret["hybrid_rrf"]["n"]
    w, h, left, top, row_h = 1400, 640, 380, 120, 110
    bar_w = 900
    parts = [
        f'<text x="40" y="52" font-size="30" font-weight="700" fill="#1c232b">Recherche : 4 stratégies, {n} questions « page connue »</text>',
        '<text x="40" y="86" font-size="18" fill="#5a6573">Mesuré sur 1 734 pages de catalogues SKF (4 498 passages), top 5. '
        "Source : reports/llmops/latest.json</text>",
    ]
    for i, (key, label) in enumerate(LABELS.items()):
        y = top + i * row_h
        parts.append(
            f'<text x="40" y="{y + 42}" font-size="20" font-weight="600" fill="#1c232b">{label}</text>'
        )
        for j, (metric, _, color) in enumerate(SERIES):
            v = ret[key][metric]
            by = y + 8 + j * 28
            parts.append(
                f'<rect x="{left}" y="{by}" width="{bar_w}" height="22" fill="#eef1f4" rx="3"/>'
            )
            parts.append(
                f'<rect x="{left}" y="{by}" width="{bar_w * v:.0f}" height="22" fill="{color}" rx="3"/>'
            )
            shown = f"{v:.2f}".replace(".", ",") if metric == "mrr@5" else f"{v * 100:.0f} %"
            parts.append(
                f'<text x="{left + bar_w * v + 8:.0f}" y="{by + 17}" font-size="16" fill="#1c232b">{shown}</text>'
            )
    ly = top + 4 * row_h + 10
    for j, (_, label, color) in enumerate(SERIES):
        x = 40 + j * 330
        parts.append(f'<rect x="{x}" y="{ly}" width="18" height="18" fill="{color}" rx="3"/>')
        parts.append(
            f'<text x="{x + 26}" y="{ly + 15}" font-size="17" fill="#1c232b">{label}</text>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="Segoe UI, sans-serif">'
        f'<rect width="{w}" height="{h}" fill="#ffffff"/>{"".join(parts)}</svg>'
    )
    WORK.mkdir(parents=True, exist_ok=True)
    src = WORK / "bench-retrieval.html"
    src.write_text(
        f'<!doctype html><meta charset="utf-8"><body style="margin:0">{svg}</body>',
        encoding="utf-8",
    )
    subprocess.run(
        [
            EDGE,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--force-device-scale-factor=1",
            f"--user-data-dir={WORK / 'edge-profile'}",
            f"--window-size={w},{h}",
            f"--screenshot={OUT}",
            src.as_uri(),
        ],
        check=True,
        capture_output=True,
    )
    print(f"chart -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
