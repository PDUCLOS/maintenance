"""Inventory tab — list every PDF in data/raw/pdf with metadata.

Reads the filesystem directly (no API roundtrip). The brand is
inferred from the filename (Schaeffler / SKF / NTN-SNR / unknown).
Uses `pdfinfo` from poppler-utils for page count + title metadata
— the same tool we use in scripts/01_setup_mlx.sh for the audit.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import settings


def _pdf_metadata(pdf_path: Path) -> dict:
    """Extract PDF metadata via the `pdfinfo` shell tool (poppler-utils)."""
    if not pdf_path.is_file():
        return {"pages": "—", "title": "(file not found)", "size_mb": "—"}
    try:
        out = subprocess.check_output(
            ["pdfinfo", str(pdf_path)], stderr=subprocess.DEVNULL, timeout=10
        ).decode("utf-8", errors="ignore")
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return {
            "pages": "?",
            "title": "(pdfinfo unavailable)",
            "size_mb": f"{pdf_path.stat().st_size / 1048576:.1f}",
        }
    info: dict = {
        "pages": "?",
        "title": "(no title)",
        "size_mb": f"{pdf_path.stat().st_size / 1048576:.1f}",
    }
    for line in out.splitlines():
        if line.startswith("Pages:"):
            info["pages"] = line.split(":", 1)[1].strip()
        elif line.startswith("Title:"):
            t = line.split(":", 1)[1].strip()
            if t:
                info["title"] = t
    return info


def _infer_brand(filename: str) -> str:
    """Infer the manufacturer from the PDF filename."""
    name_lc = filename.lower()
    if "schaeffler" in name_lc or "fag" in name_lc or "ina" in name_lc:
        return "Schaeffler"
    if "skf" in name_lc:
        return "SKF"
    if "ntn" in name_lc or "snr" in name_lc:
        return "NTN-SNR"
    return "?"


def render() -> None:
    """Render the Inventory tab content (called inside `with tab_inventory:`)."""
    st.title("📦 Inventaire des données")
    st.caption(
        "Provenance et métadonnées de toutes les sources ingérées. "
        "Voir `data/raw/pdf/INVENTORY.md` pour les détails complets."
    )

    st.subheader("Catalogues industriels (Schaeffler, SKF, NTN-SNR)")
    pdf_dir = settings.pdf_dir
    if not pdf_dir.is_dir():
        st.warning(f"Dossier PDF manquant : {pdf_dir}.")
        return

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        st.info("Aucun PDF. Ajoutes-en dans `data/raw/pdf/` et relance `make ingest`.")
        return

    rows = []
    for f in pdfs:
        meta = _pdf_metadata(f)
        title = meta["title"]
        rows.append(
            {
                "Fichier": f.name,
                "Marque": _infer_brand(f.name),
                "Pages": meta["pages"],
                "Taille": f"{meta['size_mb']} MB",
                "Titre (métadonnée PDF)": title[:80] + ("…" if len(title) > 80 else ""),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    total_mb = sum(f.stat().st_size for f in pdfs) / 1048576
    st.caption(
        f"**Total : {len(pdfs)} PDFs, {total_mb:.1f} Mo.** "
        f"Source : sites officiels des fabricants (Schaeffler, SKF). "
        f"Inventaire détaillé : [`data/raw/pdf/INVENTORY.md`](data/raw/pdf/INVENTORY.md)."
    )


__all__ = ["_infer_brand", "_pdf_metadata", "render"]
