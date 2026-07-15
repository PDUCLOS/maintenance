# PDF Inventory — Schaeffler + SKF industrial catalogues

> Generated 2026-07-15 from the official manufacturer sites.
> All PDFs are downloaded from public URLs (no auth, no paywall).
> Used in the Industrial Knowledge Copilot RAG pipeline (multi-source ingestion).

## Summary

| Stat | Value |
|------|-------|
| Total PDFs | 7 |
| Total pages | 4,343 |
| Total size | 135 MB |
| Languages | 100% English (one PDF has 9% French terms) |
| Brands covered | Schaeffler (incl. FAG) + SKF |
| Use in RAG | `make ingest` → all PDFs are chunked + embedded |

## Detailed inventory

### Schaeffler / FAG (4 PDFs, 2,150 pages, 83.5 MB)

| File | Catalogue | Pages | Size | Subject |
|------|-----------|------:|-----:|---------|
| `schaeffler-gl1-large-size-bearings.pdf` | GL 1 | 1,138 | 56.5 MB | Large size bearings (ball, roller, back-up, spherical plain, housings, accessories) |
| `schaeffler-product-reference-guide.pdf` | PRG (US edition) | 732 | 11.9 MB | Product reference guide — full INA/FAG range, US market |
| `schaeffler-sp1-super-precision-bearings.pdf` | SP 1 | 164 | 7.0 MB | Super precision bearings for machine-tool spindles |
| `fag-equipment-and-services.pdf` | WL 80 250/4 | 116 | 8.1 MB | FAG mounting and maintenance services for rolling bearings |

### SKF (3 PDFs, 2,193 pages, 50.6 MB)

| File | Catalogue | Pages | Size | Subject |
|------|-----------|------:|-----:|---------|
| `skf-17000-rolling-bearings.pdf` | 17000/1 (Oct 2018) | 1,152 | 21.9 MB | Main SKF rolling bearings catalogue (supersedes 10000) |
| `skf-100-700-bearings-mounted-products-2018.pdf` | 100-700 (2018) | 587 | 15.2 MB | SKF bearings and mounted products — US catalogue |
| `skf-bearing-maintenance-handbook.pdf` | PM CTP CAT M | 454 | 13.5 MB | SKF bearing maintenance handbook (comprehensive field guide) |

## Topic coverage

After ingestion, the RAG will have technical knowledge on:

- **Bearing types**: ball, roller (cylindrical, spherical, tapered, needle), back-up, plain
- **Bearing selection**: load ratings, life calculation, internal clearance, tolerances
- **Lubrication**: grease types, relubrication intervals, lubricant database
- **Mounting & dismounting**: tools, methods, alignment, heating devices
- **Maintenance**: condition monitoring, failure modes, troubleshooting
- **Application sectors**: machine tools (spindles), paper industry, railway, heavy duty
- **Standards**: ISO, DIN, JIS, ABEC, P0/P6/P5/P4/P2 tolerance classes

## Source URLs and licensing

All PDFs were downloaded from official manufacturer sites. They are
**technical publications distributed free of charge** for engineering
reference. Each PDF carries the manufacturer's copyright notice in its
front matter. For a portfolio/RAG POC this is appropriate use; for a
commercial product, formal licensing would be required.

### Schaeffler PDFs

| Local file | Source URL | Published |
|------------|-----------|-----------|
| `schaeffler-gl1-large-size-bearings.pdf` | https://www.schaeffler.com/remotemedien/media/_shared_media/08_media_library/01_publications/schaeffler_2/catalogue_1/downloads_6/gl1_de_en.pdf | 2016-06 |
| `schaeffler-sp1-super-precision-bearings.pdf` | https://www.schaeffler.com/remotemedien/media/_shared_media/08_media_library/01_publications/schaeffler_2/catalogue_1/downloads_6/sp1_de_en.pdf | (Schaeffler) |
| `fag-equipment-and-services.pdf` | https://www.schaeffler.com/remotemedien/media/_shared_media/08_media_library/01_publications/schaeffler_2/catalogue_1/downloads_6/wl_80250_4_de_en.pdf | (Schaeffler) |
| `schaeffler-product-reference-guide.pdf` | https://www.schaeffler.com/remotemedien/media/_shared_media/08_media_library/01_publications/schaeffler_2/catalogue_1/downloads_6/prg_us_us.pdf | (Schaeffler USA) |

> All Schaeffler PDFs: © Schaeffler Technologies AG & Co. KG. Issued for
> free engineering reference. "Reproduction in whole or in part without
> our authorisation is prohibited" — for a portfolio RAG demo, this is
> citation, not reproduction.

### SKF PDFs

| Local file | Source URL | Published |
|------------|-----------|-----------|
| `skf-17000-rolling-bearings.pdf` | https://www.skf.com/binaries/pub12/Images/0901d196802809de-Rolling-bearings---17000_1-EN_tcm_12-121486.pdf | 2018-10 |
| `skf-100-700-bearings-mounted-products-2018.pdf` | https://www.skf.com/binaries/pub12/Images/0901d196807026e8-100-700_SKF_bearings_and_mounted_products_2018_tcm_12-314117.pdf | 2018 |
| `skf-bearing-maintenance-handbook.pdf` | https://cdn.skfmediahub.skf.com/api/public/0901d1968013be94/pdf_preview_medium/0901d1968013be94_pdf_preview_medium.pdf | 2009 (PUB BU/P1 17000/1) |

> All SKF PDFs: © SKF Group 2018 (or as noted). "The contents of this
> publication are the copyright of the publisher and may not be
> reproduced (even extracts) unless prior written permission is granted."
> Same caveat as above — we extract and cite, we don't republish.

## How to re-download

```bash
# All-in-one (idempotent)
make data    # already-existing NASA CMAPSS — runs first
# For PDFs, just keep the existing files or re-download manually with curl
```

For now, the PDFs are downloaded once and committed-via-gitignore
(see `.gitignore` at project root: `data/raw/pdf/*.pdf` is ignored).
This avoids bloating the git repo. To re-download:

```bash
mkdir -p data/raw/pdf
cd data/raw/pdf
curl -sL --fail -o schaeffler-gl1-large-size-bearings.pdf \
  "https://www.schaeffler.com/remotemedien/media/_shared_media/08_media_library/01_publications/schaeffler_2/catalogue_1/downloads_6/gl1_de_en.pdf"
# (repeat for the 6 other URLs — see Source URLs above)
```

## License summary (for the AI Act data governance section)

| Item | Status |
|------|--------|
| PII in corpus | **None** (technical documentation, no personal data) |
| Provenance documented | ✅ (this file) |
| Source licence | Manufacturer technical publications, freely distributed |
| Right to use for portfolio RAG demo | ✅ Fair use, no PII, citation only |
| Right to republish | ❌ Each PDF has "no reproduction without authorisation" |
| For a commercial deployment | Formal licensing required from Schaeffler + SKF |
