# Industrial Knowledge Copilot — Plan de projet détaillé

> **Repo GitHub cible :** `github.com/PDUCLOS/industrial-knowledge-copilot` (à confirmer)
> **Auteur :** Patrice Duclos · RNCP 38777 Lead Data / AI Architect
> **Statut :** draft v1 — 2026-07-15
> **Effort estimé :** 3-4 weekends (≈ 30-40 h)
> **Objectif :** projet portfolio qui bouche le gap #1 du CV (LLM/RAG/IA générative en production)

---

## 1. Vision & positionnement CV

### Le pitch (à dire en entretien, 90 sec)

> "J'ai 14 ans de technico-commercial B2B chez Michaud Chailly, 6 ans de pilotage data-driven chez DEXIS BFC, et une plateforme MLOps de production (LyonFlow).
>
> Sur ce projet, j'ai pris ce vécu terrain et j'ai construit un copilote RAG qui répond à des questions techniques sur des produits industriels en croisant fiches techniques PDF et données structurées. Le tout en local — pas d'API payante — avec une évaluation RAGAS pour mesurer la qualité des réponses, et une stack industrialisée (Docker Compose, FastAPI, Streamlit, monitoring).
>
> C'est exactement le scope d'un POC d'IA générative appliqué à l'industrie, tel qu'on le voit dans les JDs Data Scientist/Architecte IA 2026."

### Pourquoi ce projet te fait gagner des points

| Critère d'évaluation recruteur | Ce que le projet prouve |
|-------------------------------|-------------------------|
| Maîtrise LLM / RAG / IA générative | Chaîne RAG complète + prompt engineering + évaluation |
| Industrialisation | Docker Compose, logs structurés, monitoring, tests |
| Sensibilité métier B2B | Cas d'usage industriel cohérent avec ton parcours |
| Évaluation de modèles IA | RAGAS — métriques standardisées (faithfulness, answer relevancy) |
| Veille techno | Stack 2026 (LangChain v0.3, Ollama, Mistral 7B, ChromaDB) |
| Communication technique | README pro, architecture diagram, démo live |

---

## 2. Cas d'usage cible (à confirmer)

### Option A — Maintenance prédictive (RECOMMANDÉE pour toi)
- **Domaine :** prédiction de panne / RUL (Remaining Useful Life) sur machines industrielles
- **Data publique :** NASA CMAPSS (turbofan engine degradation) — 4 sous-datasets, doc technique NASA
- **Pourquoi :** alignement avec ton ADN Carrier HVAC / DEXIS BFC industriel
- **Type de RAG :** hybride — RAG sur docs techniques + tool calling Python pour interroger les données CMAPSS

### Option B — Fiches produits B2B
- **Domaine :** questions techniques sur produits (roulements, transmission, visserie…)
- **Data :** catalogues PDF publics (SKF, Schaeffler, NTN-SNR, Michaud Chailly si tu retrouves des docs)
- **Pourquoi :** très parlant pour des recruteurs B2B/industrie
- **Type de RAG :** pur retrieval sur PDF

### Option C — Documentation interne RH / IT (générique)
- **Domaine :** assistant pour répondre aux questions employés (congés, IT, onboarding)
- **Data :** tu synthétises 30-50 faux documents (FAQ, politiques, procédures)
- **Pourquoi :** très demandé par ESN et scale-ups
- **Type de RAG :** pur retrieval + multi-turn conversation

**→ Mon avis : Option A (CMAPSS)** — combine RAG classique ET agent avec tool calling Python sur DataFrame. Double compétence visible en entretien. Data NASA est libre, propre, et l'angle maintenance prédictive résonne direct avec Carrier HVAC / DEXIS.

---

## 3. Architecture technique

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI (chat)                      │
│                    localhost:8501                            │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                FastAPI  /query  /ingest  /eval              │
│                localhost:8000                                │
└──────┬──────────────────────┬───────────────────────────────┘
       │                      │
       ▼                      ▼
┌──────────────┐      ┌────────────────────────────────┐
│   LangChain  │      │       RAGAS Evaluator          │
│   RAG Chain  │      │  (faithfulness, relevancy…)   │
└──────┬───────┘      └────────────────────────────────┘
       │
       ├─────► Ollama (LLM local)        :11434
       │       └─ Mistral 7B Instruct
       │
       ├─────► Ollama (Embeddings)       :11435
       │       └─ nomic-embed-text
       │
       └─────► ChromaDB (vector store)   :8001
               └─ persistent volume
```

### Stack détaillée (toutes versions pinning dans `requirements.txt`)

| Composant | Technologie | Pourquoi ce choix |
|-----------|-------------|-------------------|
| **LLM** | Ollama + Mistral 7B Instruct (Q4_K_M) | Local, gratuit, FR correct, 4 Go RAM, pas d'API key |
| **Embeddings** | Ollama + nomic-embed-text | Local, multilingue, 137M params, rapide |
| **Orchestration** | LangChain v0.3+ | Standard marché 2026, demandé dans 80% JDs |
| **Vector store** | ChromaDB (persistent) | Local, simple, suffisant pour < 100k chunks |
| **API** | FastAPI + Uvicorn | Déjà maîtrisé (LyonFlow) |
| **UI** | Streamlit | Déjà maîtrisé, parfait pour démo |
| **Évaluation** | RAGAS v0.2+ | Standard de fait pour évaluer un RAG |
| **Container** | Docker Compose (1 service = 1 container) | Reproductible, pas de pollution locale |
| **Tests** | pytest + RAGAS eval suite | Pas de mock — tests sur vraies données NASA |
| **Logs** | loguru (structurés JSON) | Format uniforme, parseable |
| **Data source A** | NASA CMAPSS (4 datasets + readme) | Libre, industriel, 100+ Mo |
| **Data source B** | 5-10 PDF techniques PDF (Schaeffler, NTN…) | Libres, multilingues |

### Estimation ressources (Mac M-series ou Linux)

- RAM : 8 Go suffisent (Mistral 7B Q4 = 4.4 Go, embeddings 0.3 Go, + 2 Go OS)
- Disque : 6 Go pour les modèles + 1 Go data + 2 Go code/env
- GPU : pas obligatoire (CPU OK, ~10-20 sec/query). GPU Metal MPS = 2-3x plus rapide

---

## 4. Arborescence GitHub cible

```
industrial-knowledge-copilot/
├── README.md                    # pitch + archi diagram + quickstart
├── LICENSE                      # MIT
├── .gitignore
├── .env.example                 # variables d'env (sans secrets)
├── docker-compose.yml           # 4 services : ollama, chroma, api, ui
├── Dockerfile                   # image unique API + UI (multi-stage)
├── pyproject.toml               # deps + tool config (ruff, pytest)
├── requirements.txt             # lock des versions
│
├── data/
│   ├── raw/
│   │   ├── cmapss/              # NASA CMAPSS brut (4 sous-datasets)
│   │   └── pdf/                 # 5-10 PDF techniques
│   ├── processed/
│   │   ├── chunks.jsonl         # chunks nettoyés (id, text, source, metadata)
│   │   └── eval_dataset.jsonl   # Q&R pour RAGAS
│   └── README.md                # provenance + licence des data
│
├── src/
│   ├── __init__.py
│   ├── config.py                # settings (pydantic-settings, .env)
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pdf_loader.py        # PyMuPDF → chunks
│   │   ├── cmapss_loader.py     # NASA → DataFrame + métadata
│   │   └── chunker.py           # stratégie de chunking (recursive, overlap)
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embeddings.py        # wrapper Ollama embeddings
│   │   ├── vectorstore.py       # wrapper ChromaDB (persist, search)
│   │   ├── retriever.py         # hybrid search (BM25 + dense)
│   │   ├── chain.py             # chaîne LangChain (retrieval → prompt → LLM)
│   │   ├── agent.py             # agent avec tools (SQL Python sur CMAPSS)
│   │   └── prompts/
│   │       ├── system_fr.txt    # system prompt FR (industrial expert)
│   │       ├── system_en.txt    # system prompt EN
│   │       └── qa_template.py   # template LangChain
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app
│   │   ├── routes/
│   │   │   ├── query.py         # POST /query
│   │   │   ├── ingest.py        # POST /ingest (rebuild index)
│   │   │   └── eval.py          # POST /eval (lance RAGAS)
│   │   └── schemas.py           # Pydantic models (Query, Response, Eval)
│   ├── ui/
│   │   └── streamlit_app.py     # chat interface + sidebar (sources, scores)
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── dataset.py           # génère eval_dataset.jsonl depuis CMAPSS
│   │   └── ragas_runner.py      # lance RAGAS, exporte métriques
│   └── utils/
│       ├── __init__.py
│       ├── logger.py            # loguru config (JSON structurés)
│       └── timing.py            # décorateur pour mesurer latence
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # fixtures (client FastAPI, sample chunks)
│   ├── test_ingestion.py        # PDF loader, chunker, CMAPSS loader
│   ├── test_rag_chain.py        # chaîne RAG (assertion sur réponse type)
│   ├── test_api.py              # endpoints /query /ingest /eval
│   ├── test_eval.py             # RAGAS metrics > seuil minimum
│   └── fixtures/
│       └── sample_doc.pdf       # 1 PDF test pour pytest
│
├── scripts/
│   ├── 01_setup_ollama.sh       # pull Mistral + nomic-embed
│   ├── 02_ingest.sh             # lance ingestion complète
│   ├── 03_run_eval.sh           # lance RAGAS et exporte rapport
│   └── 99_clean.sh              # reset complet (volumes Docker)
│
├── reports/                     # snapshots d'évaluation RAGAS
│   ├── eval_2026-07-20.json     # scores baseline
│   └── eval_2026-07-27.json     # scores après tuning
│
├── docs/
│   ├── architecture.md          # archi détaillée + diagrammes Mermaid
│   ├── evaluation.md            # méthodologie RAGAS + résultats
│   ├── pitch_entrevue.md        # script 90 sec + démo live
│   └── screenshots/             # PNG de l'UI et des résultats
│
└── .github/
    └── workflows/
        ├── ci.yml               # pytest + ruff sur PR
        └── docker-publish.yml   # build & push image DockerHub (optionnel)
```

**Total :** ~30 fichiers Python, ~10 fichiers de config/docs, 4 services Docker. **C'est dense mais pas énorme — chaque fichier a un rôle clair.**

---

## 5. Roadmap weekend par weekend

### 🗓 W1 — Setup + Ingestion (≈ 8-10 h)

**Samedi matin (4h)**
- [ ] Créer repo GitHub `PDUCLOS/industrial-knowledge-copilot` (public, MIT)
- [ ] Setup local : `pyenv install 3.12`, `venv`, `pip install -r requirements.txt`
- [ ] Premier `docker-compose.yml` avec 4 services qui démarrent (Ollama + Chroma + API + UI)
- [ ] `ollama pull mistral:7b-instruct-q4_K_M` et `ollama pull nomic-embed-text`
- [ ] Test : `curl http://localhost:11434/api/tags` → 2 modèles listés

**Samedi après-midi (4h)**
- [ ] Télécharger NASA CMAPSS (4 sous-datasets + readme.txt)
- [ ] `src/ingestion/cmapss_loader.py` : parse les .txt → pandas DataFrame avec colonnes (unit, cycle, op1-3, sensor1-21)
- [ ] `src/ingestion/pdf_loader.py` : PyMuPDF → texte par page
- [ ] `src/ingestion/chunker.py` : recursive chunking 500 tokens, overlap 50

**Dimanche (3h)**
- [ ] `scripts/02_ingest.sh` : pipeline complet (loaders → chunker → embeddings → ChromaDB)
- [ ] Vérif : compter les chunks ingérés, sample de 5 chunks dans `data/processed/chunks.jsonl`
- [ ] Test retrieval brut (CLI Python) : top-5 chunks sur 3 questions manuelles
- [ ] Premier commit propre sur `main`

**Livrable W1 :** ingestion qui marche, ChromaDB peuplé (~500-2000 chunks), retrieval top-k fonctionnel.

---

### 🗓 W2 — Chaîne RAG + API (≈ 8-10 h)

**Samedi matin (4h)**
- [ ] `src/rag/embeddings.py` : wrapper OllamaEmbeddings (nomic-embed-text)
- [ ] `src/rag/vectorstore.py` : wrapper Chroma (similarity_search, persist)
- [ ] `src/rag/chain.py` : chaîne LangChain (retriever → prompt → LLM → output parser)
- [ ] `src/rag/prompts/system_fr.txt` : system prompt (rôle : expert maintenance industrielle)

**Samedi après-midi (4h)**
- [ ] `src/api/main.py` : FastAPI app + CORS
- [ ] `src/api/routes/query.py` : `POST /query {"question": "..."} → {"answer": "...", "sources": [...], "scores": [...]}`
- [ ] `src/api/schemas.py` : Pydantic models
- [ ] Test manuel : `curl -X POST http://localhost:8000/query -d '{"question": "Quelle est la température optimale du turbofan ?"}'`

**Dimanche (3h)**
- [ ] Tests pytest : `test_rag_chain.py` (assertion sur format réponse), `test_api.py` (endpoint smoke)
- [ ] Premier `src/rag/agent.py` : agent LangChain avec 1 tool Python (`query_cmapss`) → l'agent peut interroger les données CMAPSS via du code Python généré
- [ ] Commit propre

**Livrable W2 :** API FastAPI qui répond à des questions sur CMAPSS via RAG + agent, 5-10 tests pytest verts.

---

### 🗓 W3 — UI Streamlit + Évaluation RAGAS (≈ 8-10 h)

**Samedi matin (4h)**
- [ ] `src/ui/streamlit_app.py` : chat interface (input box, historique, streaming response)
- [ ] Sidebar : afficher les sources (chunks retrieved) + scores cosine
- [ ] Mode "toggle" : RAG pur / Agent avec tools
- [ ] Test manuel : 10 questions types, vérifier que l'UI répond en < 20 sec

**Samedi après-midi (4h)**
- [ ] `src/eval/dataset.py` : génère 20-30 Q&R depuis CMAPSS (questions sur RUL, capteurs, conditions opérationnelles)
- [ ] `src/eval/ragas_runner.py` : lance RAGAS, calcule faithfulness, answer_relevancy, context_precision, context_recall
- [ ] `scripts/03_run_eval.sh` + export JSON dans `reports/`
- [ ] **Premier baseline RAGAS** → noter les scores (objectif : faithfulness > 0.7, relevancy > 0.7)

**Dimanche (3h)**
- [ ] README.md complet : pitch + archi diagram (Mermaid) + quickstart (3 commandes)
- [ ] `docs/architecture.md` : détails techniques
- [ ] `docs/evaluation.md` : protocole + résultats baseline
- [ ] Screenshots de l'UI Streamlit dans `docs/screenshots/`
- [ ] Commit propre

**Livrable W3 :** démo end-to-end qui marche, baseline RAGAS documenté, README pro.

---

### 🗓 W4 — Polish + Tuning + Push final (≈ 6-8 h)

**Samedi (4h)**
- [ ] Tuning des scores RAGAS :
  - Ajuster le system prompt (plus direct, citer les sources)
  - Hybrid search (BM25 + dense via Reciprocal Rank Fusion)
  - Reranking avec cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
  - Recalculer RAGAS, viser +0.1 sur chaque métrique
- [ ] Logs structurés loguru (latence, nombre de chunks retrieved, score moyen)
- [ ] Endpoint `/eval` exposé dans l'API

**Dimanche (3h)**
- [ ] GitHub Actions : CI pytest + ruff
- [ ] `docs/pitch_entrevue.md` : script 90 sec + checklist démo live
- [ ] Vidéo/GIF démo 30 sec
- [ ] Tag Git `v1.0.0` + release notes
- [ ] **Push final** : repo public, README, screenshots, vidéo
- [ ] Mise à jour CV (`cv.md`) : ajout du projet dans "Projets" / "Atouts différenciants"
- [ ] Mise à jour LinkedIn : section "Featured" + "Projets" → lien repo

**Livrable W4 :** projet public et propre, évaluable en entretien avec démo live.

---

## 6. Critères d'acceptation (Definition of Done)

Le projet est "livrable" en entretien SI :

- [ ] Repo public GitHub avec README pro (archi diagram + quickstart)
- [ ] `git clone && docker-compose up` → 4 services tournent en < 5 min
- [ ] Question type "Quelle est la RUL moyenne du turbofan en cycle 150 ?" → réponse correcte en < 20 sec
- [ ] Sources affichées (top-3 chunks + score cosine)
- [ ] Au moins 20 questions dans `eval_dataset.jsonl`
- [ ] RAGAS : faithfulness > 0.7, answer_relevancy > 0.7
- [ ] Tests pytest verts (CI GitHub Actions)
- [ ] Demo live fonctionne (capture vidéo 30 sec)
- [ ] Pitch 90 sec rodé (dans `docs/pitch_entrevue.md`)

---

## 7. Métriques de succès (à tracker)

| Métrique | Baseline W3 | Objectif W4 | Comment mesurer |
|----------|-------------|-------------|-----------------|
| Faithfulness RAGAS | ~0.5 | **> 0.75** | `ragas_runner.py` |
| Answer relevancy | ~0.6 | **> 0.75** | `ragas_runner.py` |
| Context precision | ~0.5 | **> 0.70** | `ragas_runner.py` |
| Latence query (CPU M-series) | ~15 sec | **< 10 sec** | loguru timing decorator |
| Latence query (GPU MPS) | ~5 sec | **< 5 sec** | idem |
| % questions répondues correctement | 60% | **> 80%** | eval manuelle 20 questions |
| Tests pytest | 5 tests | **> 15 tests** | CI |
| Coverage | 40% | **> 60%** | pytest-cov |

---

## 8. Plan d'évaluation (RAGAS)

### Dataset d'évaluation (20-30 Q&R générées)

Format JSONL :
```json
{"question": "Quelle est la température moyenne du capteur 11 en cycle 100 ?", "ground_truth": "La température moyenne est de X°C"}
{"question": "Quel est le RUL estimé pour l'unité 5 au cycle 150 ?", "ground_truth": "Environ 80 cycles restants"}
```

**Stratégie de génération :**
- 10 questions factuelles sur CMAPSS (stats simples, conditions opérationnelles)
- 10 questions de raisonnement (corrélation entre capteurs, dégradation)
- 5 questions multi-hop (croiser plusieurs capteurs/cycles)
- 5 questions pièges (hors scope → vérifier que l'agent dit "je ne sais pas")

### Métriques RAGAS

| Métrique | Signification | Objectif |
|----------|--------------|----------|
| **Faithfulness** | La réponse est-elle fidèle au contexte retrieved ? | > 0.75 |
| **Answer relevancy** | La réponse est-elle pertinente pour la question ? | > 0.75 |
| **Context precision** | Les chunks retrieved sont-ils les bons ? | > 0.70 |
| **Context recall** | A-t-on récupéré tous les chunks nécessaires ? | > 0.65 |

---

## 9. Plan de documentation

### README.md (300 lignes max)
1. **Hero banner** : titre + tagline + archi diagram + GIF démo
2. **Pitch 1 paragraphe** : qui, quoi, pourquoi
3. **Quickstart** : `git clone && docker-compose up && open localhost:8501`
4. **Architecture** : diagramme Mermaid + tableau des services
5. **Stack** : versions pinning
6. **Démo** : 5 questions type + captures d'écran
7. **Évaluation RAGAS** : tableau des scores + lien vers `docs/evaluation.md`
8. **Roadmap** : statut actuel + TODO
9. **Auteur** : lien LinkedIn + CV

### docs/architecture.md
- Diagrammes Mermaid (composants + séquence query)
- Décisions techniques (pourquoi Ollama vs OpenAI, Chroma vs Qdrant…)

### docs/evaluation.md
- Méthodologie RAGAS
- Tableau baseline vs v1
- Analyse des erreurs (5 plus mauvais résultats + pourquoi)

### docs/pitch_entrevue.md
- Script 90 sec (à apprendre par cœur)
- Checklist démo live (5 questions prêtes)
- 3 questions pièges recruteur + réponses

---

## 10. Ce que je peux faire vs ce que tu fais

| Tâche | Qui | Pourquoi |
|-------|-----|----------|
| Squelette repo + arbo | **Moi** | Boilerplate rapide |
| Code Python (chain, API, UI, eval) | **Moi** | Mon terrain |
| Tests pytest | **Moi** | Discipline code |
| Tuning RAGAS | **Moi + toi** | Décisions prompt = ton vécu métier |
| Décision data source (CMAPSS vs autre) | **Toi** | Impact narrative CV |
| Récupération docs PDF industriels | **Toi** | Possible seulement si tu as accès (Michaud) |
| GitHub commits (attribution Patrice seul) | **Moi** | Convention 2026-06-30 (user.md) |
| Mise à jour CV + LinkedIn après | **Toi** | Ton image perso |
| Pitch entretien | **Toi** | Tu le fais vivre |

---

## 11. Risques identifiés

| Risque | Impact | Mitigation |
|--------|--------|------------|
| Ollama pas dispo / pas rapide sur Mac | Latence > 30 sec | Fallback Mistral API payante (0.2€/1k tokens) ou modèle plus petit (Phi-3 mini) |
| Qualité RAG insuffisante sur CMAPSS | RAGAS baseline < 0.5 | Tuning prompt + chunking + reranking. Si toujours < 0.5, pivoter Option B (PDF industriels) |
| NASA CMAPSS = data connue des recruteurs | "C'est pas original" | Ajouter 1 cas d'usage PDF industriels (même 2-3 PDF) pour montrer multi-source |
| Agent avec tool calling instable | Hallucinations de code | Limiter à 1 tool simple, valider output, fallback "réponse sans tool" |
| Temps > 4 weekends | Fatigue | Couper W4 (polish) et publier en v0.9 "MVP qui marche" |

---

## 12. Décisions à prendre avant de démarrer

J'ai besoin que tu tranches sur 2-3 points pour qu'on attaque lundi :

1. **Cas d'usage :** Option A (CMAPSS maintenance), B (PDF industriels), ou C (assistant RH générique) ?
2. **Repo GitHub :** `industrial-knowledge-copilot`, `b2b-rag-copilot`, autre nom ?
3. **LLM :** Ollama local confirmé (gratuit) ou OpenAI API malgré coût (~5-10€ pour le projet) ?
4. **Niveau de polish attendu :** "v1 propre et public" (W3 + W4 complets) ou "MVP qui marche en 2 weekends" (W1 + W2 + UI minimal) ?

---

**Prochaine étape :** dès que tu valides les 4 points ci-dessus, je crée le repo + le squelette + les 4 services Docker qui démarrent, et on lance W1.
