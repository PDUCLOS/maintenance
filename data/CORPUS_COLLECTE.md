# Corpus collecté — Industrial Knowledge Copilot (Roulements + Paliers lisses/Rotules)

> Généré le 2026-07-15. Chaque entrée a été testée réellement (ouverture PDF, parsing
> `.mat`/`.csv` via scipy/pandas, comptage pages/images) — pas de déclaration sans
> vérification. Voir la colonne **Testé** pour la méthode utilisée.

**Taille totale du corpus : 45 Go** (`data/` — hors code source)

---

## 1. Documents fabricants (corpus RAG textuel)

| Fichier | Fabricant | Pages | Images/schémas | Licence | Testé |
|---|---|---|---|---|---|
| `docs/skf/skf_bearing_damage_failure_analysis.pdf` | SKF | 106 | 301 | © SKF, usage documentaire | ✅ ouvert avec PyMuPDF |
| `docs/skf/skf_bearing_maintenance_handbook.pdf` | SKF (mirror MIT) | 454 | 324 | © SKF, usage documentaire | ✅ ouvert avec PyMuPDF |
| `docs/schaeffler/schaeffler_rolling_bearing_damage_wl82102.pdf` | Schaeffler | 76 | 108 | © Schaeffler, usage documentaire | ✅ ouvert avec PyMuPDF |
| `docs/ntn-snr/ntn_snr_diagnostic_defaillances_2024.pdf` | NTN-SNR | 20 | 33 | © NTN-SNR, usage documentaire | ✅ ouvert avec PyMuPDF |
| `docs/ntn-snr/ntn_snr_catalogue_roulements_billes.pdf` | NTN-SNR | 46 | 218 | © NTN-SNR, usage documentaire | ✅ ouvert avec PyMuPDF |

**Total : 5 documents, 702 pages, 984 images/schémas intégrés.**

### Références visuelles (photos/schémas)

Ces PDF sont denses en schémas techniques (984 images comptées via PyMuPDF sur les 5
documents). `src/ingestion/pdf_loader.py` découpe déjà le texte **page par page**
(`PdfPage.page_number` conservé dans les métadonnées de chaque chunk) — donc le RAG peut
citer *"voir p.42 de skf_bearing_damage_failure_analysis.pdf"* pour toute réponse, ce qui
permet à l'utilisateur de rouvrir le PDF sur la bonne page et voir le schéma en question.
**Aucun lien vidéo n'est intégré dans ces PDF** (vérifié : seuls 3 liens externes trouvés,
tous vers `skf.com/bearings`, la page d'accueil générique — pas de vidéo produit).

### Échecs (non téléchargés, bloqués)

| Cible | Raison | Action requise |
|---|---|---|
| GGB (DP4, DX, CSM design guides) | Cloudflare bloque `curl` par empreinte TLS ; navigateur y accède mais le fichier reste dans le sandbox, pas extractible | Téléchargement manuel requis (voir URLs dans `manifest_phase2.csv`) |
| Igus (iglidur technical reference) | Pas de PDF statique — catalogue en configurateur interactif ; le seul PDF trouvé (whitepaper "Staying Power") est derrière un formulaire nom/email/adresse | Refusé de soumettre tes données perso ; à récupérer manuellement si besoin |
| INA/ELGES (rotules Schaeffler) | Même blocage WAF que Schaeffler Phase 1 (401 sur toute recherche/navigation profonde) | Téléchargement manuel requis |
| Noria/machinerylubrication.com | Non traité cette session (priorité mise sur roulements/paliers) | À faire si besoin |

---

## 2. Datasets structurés (agent pandas)

| Dataset | Format | Taille | Fichiers | Licence | Testé |
|---|---|---|---|---|---|
| **CWRU** (Case Western Reserve) | `.mat` | 1,5 Go | 159 | Recherche académique libre | ✅ `scipy.io.loadmat` OK |
| **CWRU (npz, format nettoyé)** | `.npz` | 855 Mo | 161 | Recherche académique libre | ✅ `numpy.load` OK (clés `DE`/`FE`/`BA`) |
| **NASA IMS** | `.txt` bruts | 6,1 Go | ~6300+ | Domaine public NASA | ✅ archive 7z/rar intègre, extraction complète |
| **Paderborn KAt** | `.mat` | 21 Go | 32 dossiers | **CC BY-NC 4.0** — citation obligatoire, usage non commercial | ✅ `scipy.io.loadmat` OK |
| **FEMTO-ST / PRONOSTIA** | `.csv` | 3,5 Go | ~43000 | **Non confirmée** (site source disparu) | ✅ `pandas.read_csv` OK (colonnes: heure, vibration horiz./vert.) |
| **MFPT** | `.mat` | 1,6 Go | 40 | Research/academic (voir repo) | ✅ `scipy.io.loadmat` OK |
| **XJTU-SY** | `.csv` | 11 Go | 9217 (+ 1 PDF intro) | Citation obligatoire (Wang et al. 2020, IEEE Trans. Reliability) | ✅ `pandas.read_csv` OK, colonnes `Horizontal_vibration_signals`/`Vertical_vibration_signals` |

**Total : 7 sources, ~45 Go, tous roulements — aucune "vidéo" n'existe dans ce type de
données (séries temporelles de vibration, pas de contenu visuel).**

**Bonus trouvé en testant XJTU-SY** : le dataset contient un PDF
`Introduction_to_XJTU-SY_Bearing_Dataset.pdf` avec schémas du banc d'essai — à ingérer
dans le corpus documentaire aussi, pas seulement comme dataset pandas.

---

## 3. Ce que le LLM peut effectivement citer aujourd'hui

| Type de référence | Capacité actuelle | Ce qu'il manque |
|---|---|---|
| Page PDF (schéma/photo) | ✅ `metadata["page"]` conservé par chunk depuis `pdf_loader.py` | Rien — fonctionne déjà |
| Nom du fichier source | ✅ `metadata["source"]` (ex: `pdf:skf_bearing_damage_failure_analysis.pdf:p42`) | Rien |
| Lien vers une vidéo produit | ❌ | Aucune vidéo trouvée dans les PDF ; il faudrait une passe dédiée sur les sites fabricants (YouTube/Vimeo embarqués) — pas fait cette session |
| Image extraite (pas juste le n° de page) | ❌ | `pdf_loader.py` extrait le texte, pas les images elles-mêmes. Pour vraiment "montrer" le schéma (pas juste dire "page 42"), il faudrait extraire les images en fichiers séparés (PyMuPDF `page.get_images()` + `Pixmap`) et les lier au chunk — pas fait, à décider si utile |

---

## 4. Prochaines étapes techniques (hors scope acquisition)

Brancher ce nouveau corpus sur le pipeline d'ingestion RAG (loader + chunker +
collection ChromaDB) reste une tâche séparée, non commencée — cette session a
couvert uniquement l'acquisition et la vérification des données brutes.
