# Pitch d'entretien — 90 secondes

> Apprendre par cœur. À dire calmement, sans réciter.

---

## Script (90 sec)

> "Sur ce projet, j'ai pris 14 ans de terrain B2B chez Michaud Chailly
> et 6 ans de pilotage data chez DEXIS BFC, et j'ai construit un
> **copilote RAG local** qui répond à des questions techniques sur des
> produits industriels.
>
> Concrètement : on ingère les **fiches techniques PDF** et les
> **catalogues de roulements Schaeffler / SKF / NTN-SNR** (~5 000 pages de doc technique), on
> les chunk, on les embed, on les stocke dans **ChromaDB**, et un
> **Qwen2.5-7B** tournant en local sur mon Mac — pas d'API payante —
> génère la réponse avec les sources citées.
>
> Le truc différenciant, c'est l'**agent avec tool calling Python** : si
> la question demande une moyenne ou un calcul, l'agent génère du code
> pandas qui s'exécute sur le DataFrame, et combine ça avec le RAG pour
> répondre.
>
> J'évalue la qualité avec **RAGAS** — métriques standardisées
> (faithfulness, answer relevancy) — et j'ai une stack industrialisée :
> Docker Compose, logs structurés, tests pytest, CI GitHub Actions.
>
> C'est exactement le scope d'un POC d'IA générative appliqué à
> l'industrie, tel qu'on le voit dans les JDs Data Scientist / Architecte
> IA 2026."

---

## Démo live (checklist)

**Avant l'entretien, avoir dans un terminal :**
- `make chroma-up` (ChromaDB tourne sur :8001)
- `make api` (FastAPI sur :8000)
- `make ui` (Streamlit sur :8501)
- L'UI ouverte dans le navigateur

**5 questions prêtes (à tester en avance) :**

1. **Factuel pur** : "How many turbofan engines are in the FD001 training set?"
   - Attend : une définition + source `[pdf:skf-17000-rolling-bearings.pdf:p42]`

2. **Statistique sur capteur** : "What is the mean of sensor_11 in FD002?"
   - Attend : une valeur numérique + source

3. **Raisonnement** : "Does sensor_04 tend to increase or decrease as the engine degrades in FD003?"
   - Attend : "increases" ou "decreases" + explication courte

4. **Multi-hop** : "For FD001 at cycle 150, what is the mean of sensor_07?"
   - Attend : une valeur numérique

5. **Out-of-scope (test de l'honnêteté)** : "What is the price of a new turbofan engine?"
   - Attend : "I don't know from the available data" — pas d'invention

---

## 3 questions pièges recruteur + réponses

### 1. "Pourquoi Qwen2.5-7B et pas GPT-4 ou Claude ?"

> "Trois raisons. Premièrement, c'est local : pour un POC industriel qui
> touche à des données de maintenance, on ne veut pas que les données
> sortent de la machine. Deuxièmement, c'est gratuit à l'usage, ce qui
> rend le POC industrialisable sans coût marginal. Troisièmement, en 4-bit
> quantisé, il passe sur mon MacBook Pro M5 Pro avec 2 à 5 secondes de
> latence par requête, ce qui est utilisable pour de la démo. Le code est
> agnostique au modèle — un seul adapter à changer pour swap vers un LLM
> distant."

### 1bis. "Pourquoi Qwen et pas Mistral, vu que le projet est présenté à des entreprises françaises ?"

> "J'ai d'abord testé Mistral-7B-Instruct-v0.3, cohérent avec l'ancrage
> français du projet. Mais sur l'agent avec tool calling (format ReAct
> strict), j'ai mesuré empiriquement 1 bonne réponse sur 3 questions
> quantitatives — le modèle enrobait le nom de l'outil en backticks
> markdown, ce qui cassait le matching exact, ou épuisait la limite
> d'itérations. Qwen2.5-7B, même taille, même empreinte mémoire, a
> obtenu 3/3 avec le même harnais de test. J'ai documenté la comparaison
> dans `PLAN.md` §8 plutôt que de deviner — c'est un point que j'assume
> et que je peux détailler si la question revient : le choix du modèle
> vient d'une mesure, pas d'une préférence de marque."

### 2. "Comment tu gères les hallucinations ?"

> "Trois niveaux. Le prompt système dit explicitement 'ne fabrique
> jamais de valeurs numériques, dis "je ne sais pas" si l'info n'est pas
> dans le contexte'. Ensuite, RAGAS faithfulness mesure si la réponse
> est fidèle au contexte retrieved, donc on détecte les dérives. Enfin,
> l'agent avec tool calling a un DSL fermé : le LLM ne peut pas exécuter
> du Python arbitraire, juste un petit nombre d'opérations pandas
> prédéfinies — pas de code runaway."

### 3. "Et si demain tu dois passer à 100k chunks ?"

> "Trois leviers. D'un, on passe de BM25 in-memory à un index persistant
> type pyserini ou OpenSearch. De deux, on remplace ChromaDB par Qdrant
> ou pgvector qui montent mieux en charge. De trois, on ajoute un
> cross-encoder reranker sur le top-50 avant de renvoyer le top-5 au
> LLM. Le code est déjà structuré pour que chacun de ces leviers soit
> un swap local — pas de réécriture."

---

## Si on me demande de montrer le code

Ouvrir dans l'ordre :
1. `src/rag/llm.py` — wrapper MLX → LangChain (le cœur du projet)
2. `src/rag/chain.py` — chaîne RAG LCEL
3. `src/rag/agent.py` — agent avec tool calling
4. `src/eval/ragas_runner.py` — évaluation standardisée
5. `Makefile` — orchestration de tout le pipeline

Ne pas perdre de temps sur la config ou les loaders. Aller droit au
différenciateur.
