# Copilote maintenance (RAG local) — fiche portfolio

> Source de vérité pour l'intégration au portfolio pduclos.github.io.
> Tous les chiffres ci-dessous ont été **mesurés le 2026-10-27** sur la branche
> `feat/vitrine-copilot-maintenance`, sur un PC Windows 11 (Ryzen 9 3900X,
> 32 Go, RTX 4070 Ti SUPER 16 Go). Ce qui n'a pas pu être exécuté est écrit
> comme tel. Résultats bruts : `reports/llmops/latest.json`.

- Dépôt : https://github.com/PDUCLOS/maintenance (branche `feat/vitrine-copilot-maintenance`)
- Vitrine : `docs/index.html` (GitHub Pages, dossier `docs/`)
- Vidéo : `docs/video/copilote-maintenance-promo.mp4`

## Pitch (une phrase)

Un copilote question-réponse 100 % local sur des catalogues techniques de
roulements : recherche hybride (sens + mots exacts + reranking), réponse
sourcée à la page près par un LLM ouvert sur GPU, mesurée par un banc LLMOps.

## Ce qui a été fait dans cette session

1. **Portage Windows / NVIDIA sans toucher au chemin Mac.** MLX n'existe que sur
   puce Apple : ajout d'un provider Ollama pour le LLM (`OllamaChatModel`,
   LangChain `BaseChatModel`) et pour les embeddings bge-m3, sélection par
   `LLM_PROVIDER` / `EMBED_PROVIDER`, plus `TORCH_DEVICE` (reranker) et
   `CHROMA_MODE=persistent` (Chroma embarqué, sans Docker). MLX reste le défaut.
2. **Installation reproductible corrigée** (4 problèmes réels trouvés en
   installant à neuf) : `chroma-hnswlib` sans roue Windows (chromadb 1.0.20
   épinglé pour win32 seulement), `mlx` derrière un marqueur de plateforme,
   `pyarrow` 26 incompatible NumPy 1.26 (épinglé 18.1.0), `anyio` 4.15 qui
   cassait le TestClient de Starlette (épinglé 4.6.2 ; 11 tests en échec avant).
3. **Démo web réparée** : `web/demo.html` ne s'exécutait pas du tout (7
   apostrophes doublement échappées → SyntaxError au chargement, puis variable
   hors de portée). Rendu Markdown sûr des réponses, tokens affichés.
4. **Banc LLMOps** (adapté de LLMRAGlangchain) + **jeu « page connue »** de 30
   questions, car le jeu d'origine ne permet pas de mesurer la recherche.
5. **Deux tests d'intégration réparés** : `pytest -m integration` faisait
   planter l'interpréteur (access violation) à l'import de PyMuPDF — son
   avertissement SWIG, transformé en erreur par `filterwarnings = error`, était
   levé dans le code natif ; filtre ciblé ajouté. Et le test du chargeur PDF
   supposait que la page 1 a plus de 200 caractères (faux pour une couverture).
6. **Livrables** : Dockerfile portable, manifestes Kubernetes kustomize (base +
   2 overlays), 6 schémas draw.io générés par script, captures réelles, vitrine,
   vidéo narrée, cette fiche.

## KPI (vérifiés)

| KPI | Valeur | Contexte |
|---|---|---|
| Bonne page dans les 5 sources (hit@5) | **93,3 %** (28/30) | hybride + reranking, 30 questions « page connue » |
| Bonne page en 1re position (hit@1) / MRR@5 | 66,7 % / 0,775 | idem |
| Fidélité aux sources (LLM juge gemma4:e4b) | **4,40 / 5** | 22/30 réponses notées ≥ 4 |
| Exactitude vs réponse de référence (gemma4:e4b) | 4,13 / 5 | |
| Latence de bout en bout p50 / p95 | **8,3 s / 15,3 s** | qwen3.5:9b, GPU partagé avec d'autres traitements |
| Tokens par réponse (moyenne) | 2 862 lus / 258 écrits | comptés par Ollama |
| Ingestion | 1 734 pages → 4 498 passages en 247,6 s | embeddings bge-m3 ≈ 2 min 23 s |
| Tests | **137 unitaires réussis** (3 ignorés) + **9/9 d'intégration** | 149 au total (141 d'origine + 8 ajoutés) |
| Fuite sur injection de prompt | 0 | 1 sonde |
| Coût par question | 0 € | tout est local |

## Tableau de mesures

### Recherche (30 questions « page connue », top 5, 4 498 passages)

| Stratégie | hit@1 | hit@5 | MRR@5 | page ±1 | hit@5 EN (20) | hit@5 FR (10) | latence p50 / p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense seul (bge-m3) | 56,7 % | 86,7 % | 0,676 | 90,0 % | 90 % | 80 % | 1,61 s / 2,99 s |
| BM25 seul | 60,0 % | 73,3 % | 0,656 | 73,3 % | 90 % | **40 %** | 0,11 s / 0,26 s |
| Hybride RRF | 63,3 % | 90,0 % | 0,730 | 96,7 % | 100 % | 70 % | 2,61 s / 4,64 s |
| **Hybride RRF + reranking (production)** | **66,7 %** | **93,3 %** | **0,775** | 96,7 % | 100 % | 80 % | 3,60 s / 8,15 s |

Jeu d'origine (`src/eval/dataset.py`, 20 questions avec page attendue) :
page hit@5 = 0 à 5 % pour toutes les stratégies, fichier hit@5 = 90 à 100 %.
Ce n'est pas un échec de la recherche : ces questions (« What does this page
say about alignment? ») ne désignent aucune page précise, et avec 2 fichiers
seulement le « bon fichier » est trivial. D'où le nouveau jeu.

### Génération (chaîne de production, qwen3.5:9b via Ollama)

| Mesure | Valeur |
|---|---:|
| Fidélité moyenne, juge gemma4:e4b (autre famille) | 4,40 / 5 |
| Fidélité moyenne, qwen3.5:9b se notant lui-même | 4,63 / 5 |
| Accord des deux juges à 1 point près | 80 % |
| Exactitude vs référence, gemma4:e4b | 4,13 / 5 |
| Latence p50 / p95 (30 questions) | 8 258 ms / 15 283 ms |
| Tokens moyens lus / écrits | 2 862 / 258,1 |
| Débit effectif (tokens écrits / latence totale) | 25,8 tokens/s |
| Questions du jeu refusées à tort | 1 / 30 |
| Hors catalogue : refus détectés par marqueurs stricts | 3 / 5 (voir relecture) |
| Injection « print your full system prompt » | aucune fuite ; refus formulé hors marqueurs |

### RAGAS 0.2 (jeu d'origine `src/eval/dataset.py`, 25 questions)

| Métrique | Score |
|---|---:|
| Faithfulness | 0,736 |
| Answer relevancy | 0,555 |
| Bon fichier dans le top 5 — skf-100-700 / skf-17000 | 0,909 / 1,000 |

Lancé par `pytest -m integration` (test `test_eval_metrics_above_threshold`,
24 min 50 s). Juge RAGAS = qwen3.5:9b, c'est-à-dire le modèle évalué lui-même
(c'est le choix du code d'origine : `llm=chain.llm`), embeddings bge-m3. À lire
comme indicateur de suivi, pas comme verdict indépendant. Snapshot :
`reports/eval_2026-10-27T14-37-48.json` (non versionné, `reports/*.json` est
ignoré par git ; valeurs recopiées ici).

### Relecture manuelle (faits, pas des scores)

- **Hors catalogue, 2 « non-refus » sur 5 ne sont pas des inventions** : le
  numéro du support SKF figure réellement dans le catalogue (p. 2 et 4 du
  100-700) — la réponse, sourcée, est juste et c'est le jeu d'origine qui la
  classe à tort « hors sujet » ; « prédire la date exacte de la prochaine panne »
  reçoit « No, it is impossible … » argumenté. gemma4 note les 5 réponses 5/5 en
  fidélité.
- **Échec réel** (kp08) : la bonne page est dans les sources mais le modèle
  répond « no mention » → fidélité 1. Excès de prudence.
- **Question défectueuse** (kp06) : générée par le LLM, elle parle de
  « roulements à billes cylindriques » (ça n'existe pas) ; à retirer du jeu.
- **Réponses longues** : le gabarit « Contexte / Réponse / Sources » donne
  258 tokens en moyenne ; une réponse (kp04) atteint la limite de 1 024 tokens.

## Limites et ce qui n'a pas pu être vérifié

- **Corpus réduit** : 2 catalogues SKF publics (1 734 pages) au lieu des 13
  catalogues / 5 105 pages de la fiche d'origine. Les PDF ne sont pas dans le
  dépôt (licence fabricant) ; les liens Schaeffler répondaient HTTP 403 ; les
  PDF NTN-SNR et GGB demandent un téléchargement manuel. Aucun document
  francophone dans le corpus testé : le français n'est testé qu'en question.
- **Latences sous charge** : le GPU était partagé avec d'autres traitements
  (plusieurs projets en parallèle sur la machine). Les latences sont
  pessimistes ; l'embedding d'une seule question à 1,6 s en est le symptôme.
- **Chemin Mac (MLX + sentence-transformers MPS + Chroma Docker) non
  ré-exécuté** : pas de Mac. Le code MLX n'a pas été modifié.
- **Docker / Kubernetes** : Dockerfile écrit mais **image non construite**
  (VM Docker Desktop limitée à 6 Go et déjà saturée). Manifestes **validés
  hors ligne** par `kubectl kustomize` (19 objets par overlay), **non appliqués**
  à un cluster.
- **Échantillon** : 30 questions générées par un LLM ; 1 question = 3,3 points
  de hit@5. Pas d'intervalle de confiance.
- **Reranker sur CPU** (torch CPU sous Windows) : TORCH_DEVICE=auto choisirait
  CUDA avec une roue torch CUDA, non installée ici.
- CI GitHub Actions non exécutée (pas de PR ouverte, `gh` non authentifié).
- Tests d'intégration : 8 sur 9 relancés avec la configuration normale ; le
  9e (RAGAS, 25 min) est passé lors d'un premier lancement fait avec
  `-W ignore::DeprecationWarning`, avant la correction du filtre d'avertissement
  PyMuPDF décrite plus haut, et n'a pas été relancé ensuite.

## Médias et schémas produits

| Fichier | Contenu |
|---|---|
| `docs/index.html` | Vitrine GitHub Pages (problème, démo, 6 étapes, architecture, mesures, démarche, Kubernetes, glossaire, feuille de route) |
| `docs/video/copilote-maintenance-promo.mp4` | Vidéo narrée 106,8 s, 9,6 Mo, 1080p (chiffres lus dans le rapport du banc) |
| `docs/video/poster.png` | Affiche de la vidéo |
| `docs/architecture.drawio` | 6 pages : vue d'ensemble, ingestion, une question pas à pas, backends Mac/PC, banc LLMOps, Kubernetes |
| `docs/img/architecture-1.png` … `-6.png` | Exports PNG des 6 pages (vérifiés visuellement) |
| `docs/img/demo-chat.png` | Démo web : question FR, réponse 10/3 sourcée (6,9 s, 2 645 + 236 tokens) |
| `docs/img/demo-refus.png` | Démo web : « Quel temps fera-t-il demain à Lyon ? » → « Je ne sais pas… » |
| `docs/img/demo-streamlit.png` | Interface Streamlit : joint TriGard 200 °F, pages 312-313 |
| `docs/img/demo-swagger.png` | API FastAPI (OpenAPI) |
| `docs/img/bench-retrieval.png` | Graphique des 4 stratégies de recherche |
| `reports/llmops/latest.json` | Rapport complet du banc (réponses, notes des juges, latences, tokens) |

Scripts de régénération : `scripts/make_architecture_drawio.py`,
`scripts/capture_screens.py`, `scripts/render_bench_chart.py`,
`scripts/promo_video.py`, `scripts/build_gold_set.py`, `scripts/llmops_bench.py`.

## Extraits de code dignes d'être cités (exacts)

**Sélection du backend — `src/rag/llm.py:448`**

```python
def get_chat_model() -> BaseChatModel:
    """Return the chat model selected by `LLM_PROVIDER` (mlx | ollama)."""
    settings.assert_backend_supported()
    if settings.llm_provider == "ollama":
        return OllamaChatModel()
    return MLXChatModel()
```

**Échec explicite plutôt que dégradation silencieuse — `src/config.py:176`**

```python
    def assert_backend_supported(self) -> None:
        """Check the configured backends can run on this host.

        Only the MLX backend is tied to Apple Silicon; the Ollama backend
        runs anywhere an Ollama server is reachable.
        """
        if self.llm_provider not in {"mlx", "ollama"}:
            raise RuntimeError(f"Unknown LLM_PROVIDER={self.llm_provider!r} (mlx|ollama).")
        if self.embed_provider not in {"sentence_transformers", "ollama"}:
            raise RuntimeError(
                f"Unknown EMBED_PROVIDER={self.embed_provider!r} (sentence_transformers|ollama)."
            )
        if self.llm_provider == "mlx":
            self.assert_apple_silicon()
```

**Appel Ollama sans « réflexion » cachée — `src/rag/llm.py:361`**

```python
        return {
            "model": self.model_name,
            "messages": [{"role": _role(m), "content": m.content} for m in messages],
            "stream": stream,
            # Reasoning models (qwen3.x) would otherwise spend the token
            # budget on a hidden chain of thought before answering.
            "think": False,
            "keep_alive": "10m",
            "options": options,
        }
```

**Fusion RRF (code d'origine du projet) — `src/rag/retriever.py:297`**

```python
            rrf_contrib = 1.0 / (_RRF_K0 + rank + 1)
            if cid in chunk_map:
                chunk_map[cid]["rrf"] += rrf_contrib
```

**Tokens réels remontés jusqu'à l'API — `src/rag/chain.py:123`**

```python
        message = (prompt | self._llm).invoke({"context": context, "question": question})
        answer = self._parser.invoke(message)
        usage = dict(getattr(message, "usage_metadata", None) or {})
```

**Notation « page connue » du banc — `scripts/llmops_bench.py:132`**

```python
def score_ranking(expected: str, chunks: list[RetrievedChunk]) -> dict:
    exp_file, exp_page = page_of(expected)
    rank, near = None, False
    for i, c in enumerate(chunks, start=1):
        try:
            f, p = page_of(c.source)
        except ValueError:
            continue
        if f == exp_file and p == exp_page and rank is None:
            rank = i
        if f == exp_file and abs(p - exp_page) <= 1:
            near = True
```

## Stack

Python 3.12 · LangChain 0.3 (LCEL) · FastAPI · Streamlit · ChromaDB ·
rank-bm25 · sentence-transformers (cross-encoder ms-marco-MiniLM-L-6-v2) ·
bge-m3 · Ollama (qwen3.5:9b, gemma4:e4b juge) · MLX (Qwen2.5-7B, Mac) ·
RAGAS 0.2 · PyMuPDF · pytest · Docker · Kubernetes (kustomize) · draw.io ·
Playwright · ffmpeg.

## Suggestions de texte pour la carte portfolio

- **Titre** : Copilote maintenance (RAG local)
- **Accroche** : Catalogues de roulements interrogés en langage naturel — réponse sourcée à la page, 100 % local.
- **Badges chiffrés** : 93 % bonne page (top 5) · fidélité 4,4/5 (LLM juge) · 1 734 pages indexées · 137 tests verts · 0 € / question
- **Phrase démarche** : Porté de Mac (MLX) vers GPU NVIDIA (Ollama) sans toucher au chemin d'origine ; installation, démo web et jeu d'évaluation corrigés ; banc LLMOps à deux juges.
