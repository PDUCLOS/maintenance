# PDF Inventory — Schaeffler / FAG / SKF / NTN-SNR / GGB industrial catalogues

> Mis à jour le 2026-07-16. Tous les PDF sont téléchargés depuis les sites
> officiels des fabricants (pas d'auth, pas de paywall). Le pipeline RAG
> ingère **tous** les PDF présents dans `data/raw/pdf/` via
> `make ingest` → chunking + bge-m3 embeddings → upsert dans ChromaDB.

## Summary

| Stat | Value |
|------|-------|
| Total PDFs | **13** |
| Total pages | **5,105** |
| Total size | **178 MB** |
| Languages | 100% English (2 PDFs NTN-SNR ont une table des matières bilingue FR+EN) |
| Brands covered | Schaeffler (incl. FAG + INA) · SKF · NTN-SNR · GGB |
| Use in RAG | `make ingest` → all PDFs are chunked + embedded |

> Note : la version précédente de cet inventaire listait 7 PDF
> (3 343 pages). Le projet a accumulé d'autres catalogues entre
> les sessions (juillet 2026), notamment des PDF NTN-SNR et un
> GGB self-lubricating handbook. Le compte actuel reflète ce qui
> est réellement sur disque.

## Detailed inventory (par fabricant)

### Schaeffler / FAG / INA (6 PDF, 2 224 pages, 91 MB)

| File | Catalogue | Pages | Size | Subject |
|------|-----------|------:|-----:|---------|
| `schaeffler-gl1-large-size-bearings.pdf` | GL 1 | 1,138 | 57 MB | Large size bearings (ball, roller, back-up, spherical plain, housings) |
| `schaeffler-product-reference-guide.pdf` | PRG (US edition) | 732 | 12 MB | Product reference guide — full INA/FAG range, US market |
| `schaeffler-sp1-super-precision-bearings.pdf` | SP 1 | 164 | 7 MB | Super precision bearings for machine-tool spindles |
| `schaeffler_rolling_bearing_damage_wl82102.pdf` | WL 82 102 | 76 | 3 MB | Schaeffler damage detection (failure modes + diagnostics) |
| `fag-equipment-and-services.pdf` | WL 80 250/4 | 116 | 9 MB | FAG mounting and maintenance services for rolling bearings |
| `ggb-dp4-dp4b-self-lubricating-bearings-bushings-handbook-06.26.pdf` | GGB DP4 | 60 | 4 MB | Self-lubricating bearings + bushings (polymers, composites) |

### SKF (5 PDF, 2 753 pages, 76 MB)

| File | Catalogue | Pages | Size | Subject |
|------|-----------|------:|-----:|---------|
| `skf-17000-rolling-bearings.pdf` | 17000/1 (Oct 2018) | 1,152 | 23 MB | Main SKF rolling bearings catalogue |
| `skf-100-700-bearings-mounted-products-2018.pdf` | 100-700 (2018) | 587 | 16 MB | SKF bearings and mounted products — US catalogue |
| `skf-bearing-maintenance-handbook.pdf` | PM CTP CAT M | 454 | 14 MB | SKF bearing maintenance handbook (v1) |
| `skf-bearing-maintenance-handbook-v2.pdf` | PM CTP CAT M | 454 | 14 MB | SKF bearing maintenance handbook (v2 — version récente) |
| `skf_bearing_damage_failure_analysis.pdf` | (PUB BU/P1 17000/1) | 106 | 9 MB | SKF damage + failure analysis guide |

> **Note** : les deux `skf-bearing-maintenance-handbook*.pdf` ont 454
> pages mais des hashes différents (v1 vs v2). Les deux sont ingérés
> — la v2 a une mise en page légèrement plus récente. Pour économiser
> de l'espace disque, on peut garder uniquement la v2.

### NTN-SNR (2 PDF, 66 pages, 10 MB)

| File | Catalogue | Pages | Size | Subject |
|------|-----------|------:|-----:|---------|
| `ntn_snr_catalogue_roulements_billes.pdf` | (catalogue 2024) | 46 | 9 MB | Catalogue roulements à billes NTN-SNR (FR) |
| `ntn_snr_diagnostic_defaillances_2024.pdf` | (guide 2024) | 20 | 1 MB | Diagnostic défaillances roulements NTN-SNR (FR) |

> Ces deux PDF sont en **français** — unique dans le corpus
> (tous les autres sont en anglais). Tester le miroir FR→FR sur
> des questions les citant est un bon test de robustesse.

## Topic coverage

After ingestion, the RAG has technical knowledge on:

- **Bearing types**: ball, roller (cylindrical, spherical, tapered, needle), back-up, plain, self-lubricating (composite)
- **Bearing selection**: load ratings, life calculation, internal clearance, tolerances
- **Lubrication**: grease types, relubrication intervals, lubricant database
- **Mounting & dismounting**: tools, methods, alignment, heating devices
- **Maintenance**: condition monitoring, failure modes, troubleshooting, damage analysis
- **Application sectors**: machine tools (spindles), paper industry, railway, heavy duty
- **Standards**: ISO, DIN, JIS, ABEC, P0/P6/P5/P4/P2 tolerance classes

## Source URLs and licensing

All PDFs were downloaded from official manufacturer sites. They are
**technical publications distributed free of charge** for engineering
reference. Each PDF carries the manufacturer's copyright notice in its
front matter. For a portfolio/RAG POC this is appropriate use; for a
commercial product, formal licensing would be required.

### Schaeffler PDFs

| Local file | Source URL |
|------------|-----------|
| `schaeffler-gl1-large-size-bearings.pdf` | https://www.schaeffler.com/remotemedien/media/_shared_media/08_media_library/01_publications/schaeffler_2/catalogue_1/downloads_6/gl1_de_en.pdf |
| `schaeffler-sp1-super-precision-bearings.pdf` | https://www.schaeffler.com/remotemedien/media/_shared_media/08_media_library/01_publications/schaeffler_2/catalogue_1/downloads_6/sp1_de_en.pdf |
| `fag-equipment-and-services.pdf` | https://www.schaeffler.com/remotemedien/media/_shared_media/08_media_library/01_publications/schaeffler_2/catalogue_1/downloads_6/wl_80250_4_de_en.pdf |
| `schaeffler-product-reference-guide.pdf` | https://www.schaeffler.com/remotemedien/media/_shared_media/08_media_library/01_publications/schaeffler_2/catalogue_1/downloads_6/prg_us_us.pdf |

> All Schaeffler PDFs: © Schaeffler Technologies AG & Co. KG. Issued for
> free engineering reference. "Reproduction in whole or in part without
> our authorisation is prohibited" — for a portfolio RAG demo, this is
> citation, not reproduction.

### SKF PDFs

| Local file | Source URL |
|------------|-----------|
| `skf-17000-rolling-bearings.pdf` | https://www.skf.com/binaries/pub12/Images/0901d196802809de-Rolling-bearings---17000_1-EN_tcm_12-121486.pdf |
| `skf-100-700-bearings-mounted-products-2018.pdf` | https://www.skf.com/binaries/pub12/Images/0901d196807026e8-100-700_SKF_bearings_and_mounted_products_2018_tcm_12-314117.pdf |
| `skf-bearing-maintenance-handbook.pdf` | (CDN SKF, référence PUB BU/P1 17000/1) |
| `skf-bearing-maintenance-handbook-v2.pdf` | (CDN SKF, version 2024) |
| `skf_bearing_damage_failure_analysis.pdf` | (CDN SKF, publication 2018) |

> All SKF PDFs: © SKF Group. Same copyright caveat as above.

### NTN-SNR PDFs (FR)

| Local file | Source URL |
|------------|-----------|
| `ntn_snr_catalogue_roulements_billes.pdf` | https://www.ntn-snr.com/fr/produits/catalogue (téléchargement manuel) |
| `ntn_snr_diagnostic_defaillances_2024.pdf` | https://www.ntn-snr.com/fr/produits/catalogue (téléchargement manuel) |

> All NTN-SNR PDFs: © NTN-SNR. Issued for free engineering reference.

### GGB PDF

| Local file | Source URL |
|------------|-----------|
| `ggb-dp4-dp4b-self-lubricating-bearings-bushings-handbook-06.26.pdf` | https://www.ggb.com/en/products/self-lubricating-bearings (téléchargement manuel) |

> © GGB. Issued for free engineering reference.

## How to re-download

The PDFs are downloaded once and ignored by git (see `.gitignore`:
`data/raw/pdf/*.pdf` is ignored). This avoids bloating the git repo.
To re-download:

```bash
mkdir -p data/raw/pdf
cd data/raw/pdf
# Then use curl for each URL — see Source URLs above.
```

## License summary (for the AI Act data governance section)

| Item | Status |
|------|--------|
| PII in corpus | **None** (technical documentation, no personal data) |
| Provenance documented | ✅ (this file) |
| Source licence | Manufacturer technical publications, freely distributed |
| Right to use for portfolio RAG demo | ✅ Fair use, no PII, citation only |
| Right to republish | ❌ Each PDF has "no reproduction without authorisation" |
| For a commercial deployment | Formal licensing required from each manufacturer |
