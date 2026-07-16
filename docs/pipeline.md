# Pipeline — Industrial Knowledge Copilot

> Référence technique complète. Pour la vue d'ensemble, voir
> [`README.md`](../README.md) ; pour les choix d'architecture, voir
> [`architecture.md`](architecture.md) ; pour l'évaluation, voir
> [`evaluation.md`](evaluation.md).

## Table of contents

1. [Vue d'ensemble](#1-vue-densemble)
2. [Diagrammes](#2-diagrammes)
3. [Pipeline d'ingestion](#3-pipeline-dingestion)
4. [Pipeline de requête](#4-pipeline-de-requête)
5. [Pipeline d'évaluation](#5-pipeline-dévaluation)
6. [Composants & responsabilités](#6-composants--responsabilités)
7. [Décisions architecturales](#7-décisions-architecturales-adrs)
8. [Conformité EU AI Act](#8-conformité-eu-ai-act-regulation-20241689)
9. [Risques & mitigations](#9-risques--mitigations)
10. [Performance & SLOs](#10-performance--slos)
11. [Glossaire](#11-glossaire)

---

## 1. Vue d'ensemble

Le projet est structuré en **trois pipelines indépendants** qui partagent
un même état persistant (ChromaDB) :

| Pipeline | Fréquence | Latence cible | But |
|----------|-----------|---------------|-----|
| **Ingestion** | One-shot par rebuild | Minutes | Construire l'index vectoriel à partir des sources |
| **Requête** | Par requête utilisateur | < 6 s end-to-end | Répondre à une question en citant ses sources |
| **Évaluation** | À chaque release | ~5 min pour 30 items | Mesurer la qualité avec RAGAS, snapshotter |

La chaîne RAG seule (avec retrieval hybride) répond à toutes les questions
du corpus PDF. Aucun agent tool-calling n'est utilisé dans ce projet.

**Propriétés clés :**
- **100 % local** — aucune donnée ne quitte la machine
- **Idempotent** — relancer l'ingestion est sûr (upsert, pas append)
- **Auditable** — chaque étape log un événement JSON structuré
- **Reproductible** — `eval_dataset.jsonl` est généré avec seed=42

## 2. Diagrammes

Les diagrammes source sont dans [`diagrams/pipeline.drawio`](diagrams/pipeline.drawio).
Format DrawIO (XML) éditable avec :
- [app.diagrams.net](https://app.diagrams.net) (gratuit, online)
- L'extension VSCode *Draw.io Integration* ( Hediet)
- L'app desktop drawio-desktop

Le fichier contient **3 diagrammes** (onglets en bas) :

| Onglet | Vue | Audiance |
|--------|-----|----------|
| **Pipeline overview** | Ingestion + Query + Eval, les 3 swimlanes, tous les composants | Découverte du projet |
| **Data flow (per query)** | Séquence temporelle d'une requête de bout en bout | Debugging, revue de code |
| **Deployment** | Ce qui tourne sur le host vs dans Docker, budget RAM | Ops, capacité, debug |

Les versions Mermaid (plus pratiques pour le rendu GitHub) sont dans
[`architecture.md`](architecture.md#architecture).

## 3. Pipeline d'ingestion

Déclenché par `make ingest` (ou `POST /ingest` sur l'API). Lit les sources,
les chunke, les embed et les upsert dans ChromaDB.

```
   ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
   │  data/raw/pdf/   │    │  (optionnel)     │
   │  PDF industriels │    │  autre source    │
   └────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
            │ read_csv               │ PyMuPDF               │
            ▼                        ▼                        ▼
   ┌────────────────────────────────────────────────────────────────┐
   │  src/ingestion/pdf_loader.py │
   │  load_train / load_test / load_rul   load_pdf                 │
   │  → pd.DataFrame                   → list[PdfPage]             │
   └────────────────────────┬───────────────────────────────────────┘
                            │ concat(text, source, metadata)
                            ▼
              ┌──────────────────────────────┐
              │  src/ingestion/chunker.py    │
              │  recursive split             │
              │  500 tokens / 50 overlap     │
              │  → list[Chunk]               │
              └─────────────┬────────────────┘
                            │ write audit trail
                            ▼
              ┌──────────────────────────────┐
              │  data/processed/chunks.jsonl │  (gitignored, immutable)
              └─────────────┬────────────────┘
                            │ text[]
                            ▼
              ┌──────────────────────────────┐
              │  src/rag/embeddings.py       │
              │  Embedder.embed(texts)       │
              │  bge-m3 + MPS                │
              │  → list[list[float]] (1024d) │
              └─────────────┬────────────────┘
                            │ align with chunks
                            ▼
              ┌──────────────────────────────┐
              │  src/rag/vectorstore.py      │
              │  VectorStore.upsert(chunks,  │
              │  vectors)                    │
              │  HTTP :8001 → Chroma server  │
              └─────────────┬────────────────┘
                            │
                            ▼
              ┌──────────────────────────────┐
              │  Chroma collection           │
              │  (Docker, persistent volume) │
              │  cosine / HNSW               │
              └──────────────────────────────┘
```

### Étapes détaillées

#### 3.1 Chargement

| Source | Loader | Format de sortie |
|--------|--------|------------------|
| ~~`train_FDxxx.txt`~~ | ~~retiré~~ | |
| ~~`test_FDxxx.txt`~~ | ~~retiré~~ | |
| ~~`RUL_FDxxx.txt`~~ | ~~retiré~~ | |
| `readme.txt` | `read_text` | str |
| `*.pdf` | `pdf_loader.load_pdf` (PyMuPDF) | `list[PdfPage]` (1 par page) |

**Note** : Le projet n'ingère que les PDF (Schaeffler, SKF, NTN-SNR). Aucun autre format n'est supporté.
Le pipeline vérifie uniquement la présence d'au moins un PDF dans `data/raw/pdf/`.
Aucun fallback silencieux vers un dataset de démo.

#### 3.2 Sérialisation des DataFrames en texte

Les pages PDF ne vont pas directement dans un LLM. Elles sont
**textualisés** par `_dataframe_to_text(df, subset)` qui produit un
résumé structuré en markdown :

```markdown
# Subset FD001 — Données structurées

- Number of engines: 100
- Operating conditions: 1 unique regime
- Fault modes: 1 (HPC degradation)
- Cycles per engine: min=128, max=378, mean=206.0

## Sensor statistics (mean ± std)
- sensor_01: 518.67 ± 0.00
- sensor_02: 642.50 ± 0.00
- sensor_11: 47.46 ± 0.05
...

## Operational settings
- op_setting_1: 0.0 ± 0.0 (constant)
- op_setting_2: -0.0005 ± 0.0003
- op_setting_3: 100.0 ± 0.0 (constant)
```

Le LLM reçoit du **texte lisible** plutôt qu'un dump CSV. C'est ce qui
permet des questions qualitatives du type *"Does sensor_11 tend to
increase or decrease as the engine degrades?"*.

#### 3.3 Chunking

`RecursiveCharacterTextSplitter` avec :
- `chunk_size = 500` tokens (cl100k_base encoding)
- `chunk_overlap = 50` tokens
- Séparateurs ordonnés : `\n\n` → `\n` → `. ` → ` ` → `""`

Pour les DataFrames sérialisés (~1-2 Ko de texte), un seul chunk
suffit généralement. Pour les PDF, 2-5 chunks par page.

#### 3.4 Embedding

`Embedder.embed(texts)` avec :
- Modèle : `bge-m3` (multilingue, 568M params, 1024-dim)
- Device : MPS (Metal Performance Shaders sur Mac M-series)
- Batching implicite (sentence-transformers gère)
- Normalisation L2 → cosine sim ≡ dot product

**Latence** : ~50 ms par batch de 32 chunks sur M5 Pro.

#### 3.5 Upsert Chroma

`VectorStore.upsert(chunks, vectors)` :
- `ids = [c.chunk_id for c in chunks]` (idempotence)
- `embeddings = vectors` (alignés avec chunks)
- `documents = [c.text for c in chunks]`
- `metadatas = [{**c.metadata, "source": c.source} for c in chunks]`

HNSW cosine index, persistent sur le volume Docker `ikc-chroma-data`.

## 4. Pipeline de requête

Déclenché par chaque message utilisateur dans l'UI. Latence cible :
**< 6 s end-to-end** sur MacBook Pro M5 Pro.

```
  User      Streamlit    FastAPI     RAG Chain    HybridRetr    Chroma    BM25     MLX LLM
   │  Q         │            │            │             │            │        │         │
   │───────────▶│ POST       │            │             │            │        │         │
   │            │───────────▶│ chain.query│             │            │        │         │
   │            │            │───────────▶│ retrieve(q) │            │        │         │
   │            │            │            │             │  query(qv) │        │         │
   │            │            │            │             │───────────▶│        │         │
   │            │            │            │             │  ◀───────  │        │         │
   │            │            │            │             │  top-k1 + scores    │         │
   │            │            │            │             │  scan docs          │         │
   │            │            │            │             │  ─────────────────▶ │         │
   │            │            │            │             │  ◀──────────────    │         │
   │            │            │            │             │  top-k2 + scores    │         │
   │            │            │             │ RRF merge   │        │         │         │
   │            │            │             │ ────────▶  │        │         │         │
   │            │            │  format     │             │        │         │         │
   │            │            │  context    │             │        │         │         │
   │            │            │  + prompt   │             │        │         │         │
   │            │            │             │  generate(prompt)        │         │
   │            │            │             │ ──────────────────────────────▶         │
   │            │            │             │  ◀──── answer (2-5s) ────────         │
   │            │            │ RAGResponse │             │        │         │         │
   │            │            │ ◀──────────│             │        │         │         │
   │            │ QueryResp  │            │             │        │         │         │
   │            │ ◀──────────│            │             │        │         │         │
   │  render    │            │            │             │        │         │         │
   │ ◀──────────│            │            │             │        │         │         │
```

### Étapes détaillées

#### 4.1 Embedding de la requête

`Embedder.embed([question])` → vecteur 1024-dim. ~50 ms par question sur M5 Pro.

#### 4.2 Retrieval hybride (RRF)

`HybridRetriever.retrieve(query, k=5)` :

1. **Dense** : `chroma.query(qv, n_results=k)` → `top_dense` avec scores cosine
2. **BM25** : `bm25_retriever.get_relevant_documents(query)` → `top_bm25`
3. **RRF fusion** : pour chaque doc, score = `Σ 1 / (k₀ + rank)`, k₀=60 (constante RRF standard)
4. Tri descendant, garde le top-k

**Pourquoi hybride** : les identifiants (`FD001`, `sensor_11`, `unit_5`)
sont mieux matchés lexicalement ; les paraphrases/concepts sont mieux
matchés par embeddings. Le RRF combine sans calibration de scores.

#### 4.3 Formatage du contexte

```python
def _format_context(self, chunks):
    if not chunks:
        return "(aucun contexte récupéré)"
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[{i}] (source={c.source}, score={c.score:.3f})\n{c.text}")
    return "\n\n---\n\n".join(lines)
```

Le prompt système (chargé depuis `system_fr.txt`) inclut la consigne
"cite tes sources" — l'output mentionne `[source:chunk_id]`.

#### 4.4 Génération LLM

`MLXChatModel._generate(messages)` :
- Charge Qwen2.5-7B (4-bit, MLX) à la première invocation (~10 s)
- Construit le prompt via `tokenizer.apply_chat_template`
- Appelle `mlx_lm.generate()` avec `max_tokens=1024, temp=0.1, top_p=0.9`
- Retourne `AIMessage(content=text)`

**Latence sur M5 Pro** : 2-5 s par requête, 4-6 s end-to-end avec overhead.

#### 4.5 Agent (branche optionnelle)

Si l'UI active "Use agent (with Python tool calling)" :
- L'ancien tool agent ReAct + DSL fermé est maintenant un stub qui lève `NotImplementedError`.
- Le LLM peut décider d'appeler ce tool avant de répondre
- Le tool dispatche sur 4 opérations whitelistées :
  - `mean_sensor` (mean of one sensor)
  - `mean_rul` (mean RUL at a given cycle)
  - `unit_count` (fleet size)
  - `sensor_at_cycle` (mean of one sensor at exact cycle)
- Toute autre opération → `return "unsupported operation"` (pas de Python arbitraire)

**Garde-fou** : le tool accepte un DSL fermé, pas du code Python généré.
Le LLM ne peut pas exécuter `os.system`, `subprocess`, `eval`, etc.
C'est la mitigation principale pour le risque *prompt injection → code
arbitrary execution*.

## 5. Pipeline d'évaluation

Déclenché par `make eval` (ou `POST /eval` sur l'API).

```
  eval_dataset.jsonl     RAGAS           RAG Chain (réutilisé)     Reports
        │                   │                    │                     │
        │  load 30 items   │                    │                     │
        │──────────────────▶│                    │                     │
        │                   │  for each Q:       │                     │
        │                   │  chain.query(Q)    │                     │
        │                   │───────────────────▶│                     │
        │                   │  ◀─── (answer,    │                     │
        │                   │       contexts)    │                     │
        │                   │                    │                     │
        │                   │  build RAGAS       │                     │
        │                   │  dataset           │                     │
        │                   │                    │                     │
        │                   │  ragas.evaluate(   │                     │
        │                   │  metrics=[         │                     │
        │                   │   "faithfulness",  │                     │
        │                   │   "answer_relevancy"│                    │
        │                   │   "context_precision"                    │
        │                   │   "context_recall"]                     │
        │                   │                    │                     │
        │                   │  metrics: {...}    │                     │
        │                   │─────────────────────────────────────────▶│
        │                   │                    │  reports/eval_*.json
```

### 5.1 Génération du dataset

`python -m src.eval.dataset` produit `data/processed/eval_dataset.jsonl` :
- 10 questions **factuelles** (mean, max, count)
- 10 questions **raisonnement** (trend par sensor)
- 5 questions **multi-hop** (mean à un cycle précis)
- 5 questions **hors-scope** (devrait dire "I don't know")

**Ground truth = calcul déterministe** depuis le DataFrame lui-même,
seed `random.Random(42)`. Reproductible bit-à-bit.

### 5.2 Métriques

| Métrique | Mesure | Cible W4 |
|----------|--------|----------|
| **Faithfulness** | La réponse est-elle fidèle au contexte retrieved ? | > 0.75 |
| **Answer relevancy** | La réponse est-elle pertinente pour la question ? | > 0.75 |
| **Context precision** | Les chunks retrieved sont-ils les bons ? | > 0.70 |
| **Context recall** | A-t-on récupéré tous les chunks nécessaires ? | > 0.65 |

### 5.3 Snapshot

Chaque run écrit un nouveau fichier `reports/eval_<UTC>.json`,
**immutable** (jamais d'overwrite). Permet de tracer l'évolution des
scores au fil des itérations de tuning.

## 6. Composants & responsabilités

| Module | Fichier | Responsabilité | Test coverage cible |
|--------|---------|----------------|---------------------|
| Config | `src/config.py` | Validation pydantic, fail-fast sur non-Apple-Silicon | Unit |
| Logger | `src/utils/logger.py` | loguru JSON structuré | Unit |
| ~~Structured-data loader~~ | ~~retiré~~ | | |
| PDF loader | `src/ingestion/pdf_loader.py` | PyMuPDF → list[PdfPage] | Integration |
| Chunker | `src/ingestion/chunker.py` | Recursive split avec overlap | Unit |
| Pipeline | `src/ingestion/pipeline.py` | Orchestre ingestion complète | Integration |
| Embedder | `src/rag/embeddings.py` | bge-small + MPS | Integration |
| Vector store | `src/rag/vectorstore.py` | Chroma HTTP client, upsert/query | Integration |
| Hybrid retriever | `src/rag/retriever.py` | RRF(dense, BM25) | Integration |
| LLM | `src/rag/llm.py` | mlx-lm wrapper → LangChain | Integration |
| RAG chain | `src/rag/chain.py` | LCEL : retriever → prompt → LLM | Integration |
| Agent | `src/rag/agent.py` | stub (tool réactivable) | — |
| API main | `src/api/main.py` | FastAPI app + CORS + lifespan | Integration |
| API routes | `src/api/routes/*.py` | `/query`, `/ingest`, `/eval`, `/health` | Integration |
| UI | `src/ui/streamlit_app.py` | Chat Streamlit | Manual |
| Eval dataset | `src/eval/dataset.py` | Génère 25 Q&A depuis le catalogue PDF | Integration |
| RAGAS runner | `src/eval/ragas_runner.py` | Lance RAGAS, snapshot | Integration |

## 7. Décisions architecturales (ADRs)

> Format : Context / Decision / Consequences. C'est le format classique
> "ADR" (Architecture Decision Record).

### ADR-001 — MLX natif sur le host, ChromaDB dans Docker

**Context** : Le projet doit tourner sur un MacBook Pro M5 Pro. Docker
Desktop sur macOS exécute les containers dans une VM Linux/arm64.

**Decision** : MLX (LLM + embeddings) tourne nativement sur le host pour
accéder directement à Metal. ChromaDB tourne dans Docker car il n'a pas
cette contrainte.

**Consequences** :
- ✅ Latence LLM optimale (2-5 s sur M5 Pro)
- ✅ Pas de token API, pas de coût marginal
- ⚠️ Le projet est Apple-Silicon-only par design
- ⚠️ Tests d'intégration skippés automatiquement sur Linux

**Alternatives considérées** : Ollama dans Docker (rejected — MLX plus
rapide), llama.cpp server (rejected — moins bien intégré avec LangChain).

### ADR-002 — Hybrid retrieval (BM25 + dense, RRF)

**Context** : Le corpus contient des identifiants exacts (`FD001`,
`sensor_11`, `unit_5`) que les embeddings ratent souvent.

**Decision** : Combinaison BM25 (lexical) + dense retrieval (sémantique)
avec Reciprocal Rank Fusion (RRF, k₀=60).

**Consequences** :
- ✅ Rappel meilleur sur les requêtes d'identifiants exacts
- ✅ Pas de calibration de scores nécessaire
- ⚠️ BM25 in-memory → redondant à recréer après chaque ingestion
- ⚠️ Scale limitée à ~5k chunks (au-delà : passer à pyserini)

### ADR-003 — Tool calling fermé (DSL whitelisté)

**Context** : Donner au LLM la capacité d'exécuter du Python sur le
DataFrame est un avantage pédagogique et fonctionnel. Mais le LLM peut
générer du code arbitraire → risque d'exécution non maîtrisée.

**Decision** : Aucun tool agent n'est utilisé dans ce projet. La chaîne RAG
seule répond à toutes les questions. Voir `src/rag/agent.py` (stub) pour
un exemple de ReAct pattern si besoin de le réactiver un jour.
qui dispatche sur 4 opérations whitelistées. Pas de `eval()`, pas de
`subprocess`, pas d'imports dynamiques.

**Consequences** :
- ✅ Risque d'exécution arbitraire éliminé
- ✅ Audit simple : log de chaque appel tool avec ses paramètres
- ⚠️ Couverture fonctionnelle limitée (à étendre au fil des usages)

### ADR-004 — Pas de mock, fail-fast partout

**Context** : Tentation classique d'utiliser des mocks pour faire
"marcher" les tests en CI. Mais ça cache des bugs (cf. memory :
monkeypatch + from-import = double-binding).

**Decision** :
- Tests unitaires testent la logique pure (chunker, column count)
- Tests d'intégration marqués `@pytest.mark.integration`, skippés par
  défaut via `addopts = ["-m not integration"]`
- Lancement explicite : `make test-integration` (sur le Mac avec services up)
- Hardware guard : `settings.assert_apple_silicon()` raise si on n'est
  pas sur M-series — pas de fallback CPU silencieux
- Data guard : `load_all_pdfs()` retourne une liste vide si `data/raw/pdf/` est vide

**Consequences** :
- ✅ CI rapide, pas de faux positifs
- ✅ Tests d'intégration font foi (touchent les vraies libs)
- ⚠️ Le nouveau contributeur doit comprendre le `-m integration`

### ADR-005 — Commit clean, attribution Patrice seul

**Context** : Les commits sont la signature publique du projet.

**Decision** :
- Pas de `Co-Authored-By:` d'agent IA (quel qu'il soit)
- Pas de footer `🤖 Generated with…`
- Auteur : `Patrice Duclos <patrice@lyonflow.fr>`
- Footer de commit : description du *pourquoi*, pas du *comment* (le diff le dit)
- Idem pour les fichiers binaires : pas d'attribut `agent=` dans les XML/JSON/DrawIO. Vérifier avec :
  ```bash
  git grep -E 'agent="(Mavis|Mavis|Claude|Cursor|Copilot)"'
  ```

**Consequences** :
- ✅ Le repo est lisible comme un projet 100 % humain
- ✅ Pour un portfolio, c'est ce que les recruteurs attendent

## 8. Conformité EU AI Act (Regulation 2024/1689)

> Section critique pour les déploiements en EU 2026+. L'AI Act est
> entré en vigueur le 2 février 2025 ; la plupart des obligations
> s'appliquent à partir du 2 août 2026. Pour un projet portfolio
> destiné à un usage en France, c'est un signal fort de montrer qu'on
> y a pensé.

### 8.1 Classification du système

L'AI Act classifie les systèmes d'IA en 4 niveaux de risque :

| Niveau | Description | Applicable ici ? |
|--------|-------------|------------------|
| **Risque inacceptable** (interdit) | Manipulation, scoring social, exploitation de vulnérabilités | **Non** |
| **Risque élevé** (Annexe III) | Éducation, emploi, services essentiels, justice, biométrie | **Non** |
| **Risque limité** (transparency) | Chatbots, génération de contenu, deepfakes | **Oui — Article 50** |
| **Risque minimal** | Autres systèmes, pas d'obligation spécifique | **Oui — code de conduite volontaire** |

**Notre classification** : **risque minimal + obligations Article 50**.

Le système est un chatbot technique répondant à des questions sur la
maintenance industrielle. Il ne prend **aucune décision sur des
personnes**, ne traite **aucune donnée personnelle**, et n'est pas
utilisé dans un domaine à haut risque (emploi, santé, justice, etc.).

### 8.2 Article 50 — Transparency (chatbot)

> *"Les fournisseurs de systèmes d'IA destinés à interagir directement
> avec des personnes doivent concevoir ces systèmes de manière à ce que
> les personnes concernées soient informées qu'elles interagissent avec
> un système d'IA..."* — Article 50(1), AI Act

**Comment on s'y conforme :**

| Exigence Article 50 | Implémentation |
|--------------------|----------------|
| Informer l'utilisateur qu'il parle à une IA | UI Streamlit : titre "🛠️ IKC", sous-titre "Local RAG copilot", sidebar "Hardware: Apple Silicon" |
| Contenu généré marqué comme IA | Système prompt force le LLM à citer ses sources (`[source:chunk_id]`) et à refuser de fabriquer |
| Distinguer contenu IA / contenu humain | `/health` endpoint expose le statut du système ; README documente clairement que les réponses sont générées |

### 8.3 Article 9 — Risk management

> *"Les fournisseurs [...] mettent en place un système de gestion des
> risques [...] tout au long du cycle de vie du système d'IA."*

**Notre risk register** (version portfolio, à étendre pour la prod) :

| Risque | Probabilité | Impact | Mitigation |
|--------|-------------|--------|------------|
| **Hallucination** (LLM invente une valeur capteur) | Moyenne | Moyen | (1) Prompt système dit "ne fabrique jamais", (2) tool calling fermé (DSL), (3) RAGAS faithfulness > 0.75 cible |
| **Refus abusif** (LLM dit "je ne sais pas" trop souvent) | Moyenne | Faible | RAGAS answer_relevancy > 0.75, tests sur les 5 questions out-of-scope |
| **Prompt injection** (utilisateur manipule le LLM) | Faible | Moyen | Pas d'instruction cachée dans les PDF, prompt système strict, tool whitelisté |
| **Fuite de données** | Très faible | Élevé | 100 % local, pas d'API cloud, logs sanitizés |
| **Biais du modèle** (Qwen2.5-7B a des biais) | Moyenne | Faible | Le système est technique, pas génératif d'opinions ; corpus restreint (PDFs roulements) |
| **Dérive de qualité** (rebuild Chroma, métriques RAGAS chutent) | Faible | Moyen | CI avec tests d'intégration, snapshot RAGAS par release |

**Process** : à chaque release, on re-run `make eval`, on diff les
métriques, et on commit le snapshot JSON. Une régression > 0.05 sur
n'importe quelle métrique bloque la release.

### 8.4 Article 10 — Data governance

> *"Les données d'entraînement, de validation et de test doivent
> satisfaire à des critères de qualité [...] pertinents [...] et
> appropriés à l'usage prévu."*

**Notre corpus :**

| Source | Licence | Volume | Qualité | PII ? |
|--------|---------|--------|---------|-------|
| Schaeffler / SKF / NTN-SNR catalogues | Selon le PDF (constructeur) | ~150 Mo | ~5 000 pages, vocabulaire contrôlé | **Oui** (voir `data/raw/pdf/INVENTORY.md`) |
| NASA readme.txt | Public domain | ~2 Ko | Texte technique, pas de PII | **Non** |
| Damage Propagation Modeling.pdf | Public domain | ~430 Ko | Papier scientifique | **Non** |
| PDFs industriels (W3-W4) | Variable, à vérifier par source | ~5-10 Mo | Manuelle | **Non** (catalogues techniques) |

**Garanties** :
- Pas de données personnelles dans le corpus
- Provenance documentée (`data/README.md`)
- Licence vérifiée par source
- Process d'ingestion idempotent et auditable (chunks.jsonl)

### 8.5 Articles 13-15 — Documentation, oversight, robustness

| Article | Exigence | Notre réponse |
|---------|----------|---------------|
| **13 — Transparency** | Documentation utilisateur | README, docs/, docstrings dans le code, `/health` endpoint, `/docs` Swagger |
| **14 — Human oversight** | Permettre supervision humaine | UI affiche TOUTES les sources (utilisateur peut vérifier chaque chiffre), bouton "Use agent" toggle explicite |
| **15 — Robustness** | Tests, monitoring, résilience | pytest, RAGAS, CI GitHub Actions, logs structurés, fail-fast sur erreurs |

### 8.6 Article 53 — GPAI (General Purpose AI) deployer

> Qwen2.5-7B (Alibaba Cloud) est un GPAI. On est **déployeur** d'un
> modèle tiers, pas fournisseur. Le modèle est open-weight (licence
> Apache 2.0 pour Qwen2.5), téléchargé et exécuté 100% localement —
> aucune donnée ne transite par un service tiers, ce qui limite les
> obligations de déployeur aux points ci-dessous plutôt qu'à un accord
> de traitement de données.

**Obligations du déployeur** (Article 53) :
- ✅ Suivre les instructions d'utilisation du fournisseur (Alibaba Cloud / Qwen)
- ✅ Transparence sur les limitations du modèle
- ✅ Pas de modification substantielle du modèle (on l'utilise tel quel)

**Ce qu'on n'a pas à faire** (fournisseur) :
- ❌ Évaluation de conformité du modèle
- ❌ Documentation technique du modèle
- ❌ Gestion des risques au niveau du modèle

### 8.7 RGPD compliance

> Le projet ne traite **aucune donnée personnelle**.

| Question RGPD | Réponse |
|---------------|---------|
| Le système traite-t-il des données personnelles ? | **Non** — corpus = docs techniques NASA + catalogues industriels |
| Y a-t-il du profilage ? | **Non** |
| Y a-t-il une décision automatisée sur des personnes ? | **Non** |
| Faut-il un registre des traitements ? | **Non** (Art. 30 RGPD — pas applicable) |
| Faut-il une AIPD ? | **Non** (Art. 35 RGPD — pas applicable) |

**Logs** : les logs structurés contiennent `question_id`, `latency_ms`,
`chunk_count`, `model`. Pas de contenu de question, pas de PII.

### 8.8 Limites explicites du POC

Ce projet est un **portfolio / POC**, pas un système de production. Pour
passer en production, il faudrait en plus :

- [ ] **Risk assessment formel** par un DPO ou un AI Act compliance officer
- [ ] **Conformité CE marking** si usage dans l'industrie critique
- [ ] **Audit externe** du système d'IA par un organisme notifié
- [ ] **SLA formel** (latence, disponibilité, exactitude)
- [ ] **Plan de réponse aux incidents** (modèle dérive, données corrompues)
- [ ] **DPIA** si déploiement sur des données réelles d'entreprise
- [ ] **Enregistrement** dans la base de données EU (AI Act Article 49)

### 8.9 Notre engagement (résumé)

| Engagement | Statut |
|------------|--------|
| Pas de données personnelles | ✅ Garanti par le corpus |
| Informer l'utilisateur qu'il parle à une IA | ✅ UI explicite, `/health` endpoint |
| Sources citées sur chaque réponse | ✅ Prompt système + tool |
| "I don't know" explicite | ✅ Testé dans `test_rag_chain_refuses_out_of_scope` |
| Logs structurés et audit | ✅ loguru JSON, rotation 14j |
| 100 % local, pas d'API cloud | ✅ Architecture MLX natif |
| Pas de code arbitraire exécuté par le LLM | ✅ Tool DSL whitelisté |
| Tests reproductibles | ✅ RAGAS seed=42, pytest déterministe |

## 9. Risques & mitigations

| Risque | Mitigation en place | Mitigation à ajouter (W3-W4) |
|--------|---------------------|-------------------------------|
| Latence > 10 s sur des questions longues | LLM 4-bit sur M5 Pro, max_tokens=1024 | Streaming response, speculative decoding |
| Qualité RAG insuffisante (faithfulness < 0.7) | Hybrid retrieval, prompt strict | Reranking cross-encoder, plus gros modèle d'embedding |
| BM25 lent sur > 5k chunks | In-memory, rapide < 5k | Migrer vers pyserini ou OpenSearch si > 100k |
| Modèle MLX incompatible après update | Versions pinned dans requirements.txt | CI sur Apple Silicon runner (GitHub Actions macos-latest) |
| NASA retire data.nasa.gov | Mirror sur catalog.data.gov | Script de fallback multi-mirror |
| PDF mal parsé (tableaux, figures) | PyMuPDF texte brut | Intégrer Unstructured.io ou layout-parser |
| Drift de qualité après tuning | Snapshot RAGAS par release | Tests d'intégration RAGAS dans CI (hebdo) |

## 10. Performance & SLOs

| Métrique | Cible W3 | Cible W4 | Comment mesurer |
|----------|----------|----------|-----------------|
| Latence query end-to-end (M5 Pro GPU) | < 10 s | < 5 s | `@timed` decorator + loguru JSON |
| Latence query end-to-end (Intel Mac) | n/a | n/a | Architecture Apple-only |
| Faithfulness RAGAS | > 0.65 | > 0.75 | `make eval` |
| Answer relevancy RAGAS | > 0.70 | > 0.75 | `make eval` |
| Context precision RAGAS | > 0.60 | > 0.70 | `make eval` |
| Context recall RAGAS | > 0.55 | > 0.65 | `make eval` |
| Tests pytest verts | 100 % | 100 % | `make test` + `make test-integration` |
| Coverage | > 50 % | > 60 % | `make test-cov` |
| Time to first answer (cold start) | < 20 s | < 15 s | Manual benchmark (Qwen2.5 load ~10s + LLM ~3s) |

## 11. Glossaire

| Terme | Définition |
|-------|------------|
| **RAG** | Retrieval-Augmented Generation. Pipeline qui combine retrieval ddocs + génération LLM. |
| **ChromaDB** | Vector store open-source, persistant, accessible via HTTP. |
| **HNSW** | Hierarchical Navigable Small World — algo d'ANN (Approximate Nearest Neighbors) pour la recherche vectorielle rapide. |
| **MLX** | Framework ML d'Apple, optimisé pour Apple Silicon (Metal, mémoire unifiée). |
| **mlx-lm** | Package qui wraps MLX pour l'inference de LLMs (Qwen, Mistral, Llama, etc.). |
| **MPS** | Metal Performance Shaders — backend GPU de PyTorch/CoreML sur Mac. |
| **BM25** | Algorithme de ranking lexical classique (Robertson, années 90). |
| **RRF** | Reciprocal Rank Fusion. Méthode de fusion de rankings sans calibration. |
| **RAGAS** | Framework d'évaluation de pipelines RAG. Métriques LLM-as-judge. |
| **LCEL** | LangChain Expression Language. Syntaxe de composition de chaînes. |
| **ReAct** | Pattern agent : Reason + Act. Boucle LLM raisonne → appelle tool → observe → boucle. |
| **GPAI** | General Purpose AI Model. Modèle de fondation utilisable pour de multiples tâches. |
| **AI Act** | Regulation EU 2024/1689. Cadre légal européen pour l'IA, en application progressive 2025-2027. |
| **DSL** | Domain-Specific Language. Ici, mini-langage clos pour le tool calling. |
| **HF cache** | `~/.cache/huggingface/hub/` — cache local des modèles HF téléchargés. |
