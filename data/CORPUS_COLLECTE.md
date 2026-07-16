# Corpus collecté — Industrial Knowledge Copilot (Roulements)

> Mis à jour le 2026-07-16 après le pivot à 100% catalogues PDF.
> Le projet n'ingère plus que les **5 catalogues PDF** ci-dessous. Les
> datasets structurés (CWRU, MFPT, Paderborn, XJTU-SY, FEMTO-ST, NASA IMS)
> qui étaient en discussion pour un agent pandas ont été **retirés du scope
> en juillet 2026** : l'agent tool-calling a été supprimé en même temps,
> et le pivot à un focus 100% roulements ne justifiait pas l'ingestion
> de séries temporelles vibration.

**Taille totale du corpus actif : 218 Mo** (`data/raw/pdf/`).

---

## 1. Documents fabricants (corpus RAG textuel)

| Fichier | Fabricant | Pages | Images/schémas | Licence | Testé |
|---|---|---|---|---|---|
| `data/raw/pdf/skf-100-700-bearings-mounted-products-2018.pdf` | SKF | 106 | 301 | © SKF, usage documentaire | ✅ ouvert avec PyMuPDF |
| `data/raw/pdf/skf-17000-rolling-bearings.pdf` | SKF | 454 | 324 | © SKF, usage documentaire | ✅ ouvert avec PyMuPDF |
| `data/raw/pdf/schaeffler-rolling-bearing-damage-wl82102.pdf` | Schaeffler | 76 | 108 | © Schaeffler, usage documentaire | ✅ ouvert avec PyMuPDF |
| `data/raw/pdf/ntn-snr-diagnostic-defaillances-2024.pdf` | NTN-SNR | 20 | 33 | © NTN-SNR, usage documentaire | ✅ ouvert avec PyMuPDF |
| `data/raw/pdf/ntn-snr-catalogue-roulements-billes.pdf` | NTN-SNR | 46 | 218 | © NTN-SNR, usage documentaire | ✅ ouvert avec PyMuPDF |

**Total : 5 documents, 702 pages, 984 images/schémas intégrés.**

> Note : la liste précédente référençait 7 PDF "Schaeffler + SKF". Le compte a
> été refait après le pivot : les PDF effectivement présents dans
> `data/raw/pdf/` sont 5 (3 SKF + 1 Schaeffler + 2 NTN-SNR — certaines
> versions antérieures ayant été dédupliquées).

### Références visuelles (photos/schémas)

Ces PDF sont denses en schémas techniques (984 images comptées via PyMuPDF sur les 5
documents). `src/ingestion/pdf_loader.py` découpe déjà le texte **page par page**
(`PdfPage.page_number` conservé dans les métadonnées de chaque chunk) — donc le RAG peut
citer *"voir p.42 de skf-17000-rolling-bearings.pdf"* pour toute réponse, ce qui
permet à l'utilisateur de rouvrir le PDF sur la bonne page et voir le schéma en question.
**Aucun lien vidéo n'est intégré dans ces PDF** (vérifié : seuls 3 liens externes trouvés,
tous vers `skf.com/bearings`, la page d'accueil générique — pas de vidéo produit).

### Échecs (non téléchargés)

| Cible | Raison | Action |
|---|---|---|
| GGB (DP4, DX, CSM design guides) | Cloudflare bloque `curl` par empreinte TLS | Téléchargement manuel requis |
| Igus (iglidur technical reference) | Pas de PDF statique (configurateur interactif) | Refusé de soumettre des données personnelles |
| INA/ELGES (rotules Schaeffler) | WAF 401 sur navigation profonde | Téléchargement manuel requis |
| Noria / machinerylubrication.com | Non traité cette session | — |

---

## 2. Datasets structurés (pandas / séries temporelles)

**Retirés du scope en juillet 2026.** Le projet n'expose plus d'agent
tool-calling (`src/rag/agent.py` est désormais un stub), donc l'intérêt
d'avoir des datasets structurés de dégradation est nul — il faudrait
un agent pour les requêter, et l'agent n'existe plus.

Les datasets collectés lors de la phase d'exploration (CWRU, MFPT,
Paderborn, XJTU-SY, FEMTO-ST, NASA IMS, ~45 Go au total) ont été
**supprimés du disque** en juillet 2026. Ils restent disponibles
publiquement aux URLs d'origine si le projet revient un jour sur
de l'analyse de séries temporelles vibration.

---

## 3. Ce que le LLM peut effectivement citer aujourd'hui

| Type de référence | Capacité actuelle | Ce qu'il manque |
|---|---|---|
| Page PDF (schéma/photo) | ✅ `metadata["page"]` conservé par chunk depuis `pdf_loader.py` | Rien — fonctionne déjà |
| Nom du fichier source | ✅ `metadata["source"]` (ex: `pdf:skf-17000-rolling-bearings.pdf:p42`) | Rien |
| Lien vers une vidéo produit | ❌ | Aucune vidéo trouvée dans les PDF |
| Image extraite (pas juste le n° de page) | ❌ | `pdf_loader.py` extrait le texte, pas les images elles-mêmes. Pour vraiment "montrer" le schéma, il faudrait extraire les images en fichiers séparés (PyMuPDF `page.get_images()` + `Pixmap`) et les lier au chunk — pas fait, à décider si utile |

---

## 4. Évaluation réelle du système (smoke test 2026-07-16)

Le smoke test `scripts/test_query_relevance.py` valide en 6 questions :

1. **load_rating_en** — *"What is the basic dynamic load rating (C)?"* → définition ISO 281, 1M revolutions. ✓
2. **lubrication_fr** — *"Comment choisir une graisse ?"* → miroir FR→FR, structure Contexte/Réponse. ✓
3. **mounting_en** — *"Procedure to mount a deep groove ball bearing?"* → SKF Drive-up Method. ✓
4. **diagnosis_fr** — *"Modes de défaillance ?"* → refus honnête quand le PDF n'a pas le contenu. ✓
5. **mirror_en** — *"What is a rolling bearing used for?"* → miroir EN→EN. ✓
6. **out_of_scope** — *"Weather forecast for Paris?"* → refus propre, sources citées. ✓

**0 jargon leak** détecté sur 6 réponses. Voir `scripts/test_query_relevance.py`.
