# Demo questions — Industrial Knowledge Copilot

> 10 questions de référence, classées par catégorie. À utiliser en démo
> d'entretien (cf. `docs/pitch_entrevue.md`) ou pour smoke-tester le
> pipeline après une release.

---

## 1. CMAPSS — Factuel pur (RAG + retrieval)

Ces questions testent la capacité du RAG à retrouver des valeurs exactes
depuis le texte sérialisé des datasets CMAPSS.

- **Q1** : How many turbofan engines are in the FD001 training set?
  - Attendu : "100 engines."

- **Q2** : What is the mean of sensor_11 across all cycles in FD002?
  - Attendu : une valeur numérique, ~47-48 (sensor_11 est très stable).

- **Q3** : What is the maximum number of cycles observed for any unit in FD003?
  - Attendu : un nombre de cycles, ~500.

## 2. CMAPSS — Raisonnement (RAG + retrieval)

Ces questions testent la capacité du LLM à interpréter les trends.

- **Q4** : Does sensor_04 tend to increase, decrease, or stay stable as the engine degrades in FD001?
  - Attendu : "decreases." (le sensor_4 = T2 - température HPC outlet, qui baisse avec l'usure).

- **Q5** : What is the mean of sensor_11 at cycle 100 in FD001?
  - Attendu : ~47.5 (très stable, peu de variation).

## 3. CMAPSS — Multi-hop (agent with tool calling)

Ces questions testent l'agent avec tool calling Python. **Activer le toggle "Use agent" dans l'UI**.

- **Q6** : For FD001, what is the mean of sensor_07 across all units?
  - Attendu : une valeur numérique précise.

- **Q7** : What is the mean RUL in FD002?
  - Attendu : ~110-120 cycles.

## 4. Schaeffler + SKF — RAG sur PDFs techniques

Ces questions testent la capacité du RAG à retrouver des infos depuis
les catalogues industriels.

- **Q8** : What is the recommended mounting temperature for FAG induction heating devices?
  - Attendu : ~120°C (info du PDF FAG Equipment and Services).

- **Q9** : What is the basic dynamic load rating formula used by SKF for rolling bearings?
  - Attendu : "L10 = (C/P)^p" (info du PDF SKF 17000/1, section A.1).

## 5. Out-of-scope (test d'honnêteté)

**Aucune de ces questions ne doit recevoir une réponse confiante.**

- **Q10** : What is the phone number of NASA support?
  - Attendu : "I don't know from the available data."

---

## Comment lancer la démo (5 minutes)

```bash
# 1. Une seule fois (à faire avant l'entretien)
cd "/Users/patriceduclos/Library/CloudStorage/GoogleDrive-patrice.noel.duclos@gmail.com/Mon Drive/PromptAI/rag-copilot"
make setup && make pull-models && make data && make chroma-up
make ingest && make eval-dataset

# 2. Le jour J, dans 3 terminaux
make api    # terminal 1 — FastAPI :8000
make ui     # terminal 2 — Streamlit :8501
# terminal 3 — libre, pour les logs / debug

# 3. Poser les questions dans l'ordre 1 → 10 dans l'onglet Chat
#    Activer le toggle "Use agent" pour Q6 et Q7
```

## Checklist "démo qui marche" (à checker 1h avant)

- [ ] `make health` (ou `/health` endpoint) → `status: ok`, `chroma: True`, `mlx_ready: True`
- [ ] `make eval` produit un snapshot récent dans `reports/`
- [ ] L'UI Charge bien l'inventaire (Schaeffler + SKF visibles)
- [ ] Q1 répond en < 5 sec avec "100 engines"
- [ ] Q4 répond avec "decreases" ou "increases" + justification
- [ ] Q10 dit "I don't know"
- [ ] L'agent (Q6, Q7) renvoie une valeur numérique quand le toggle est activé

## Erreurs courantes et fix rapide

| Erreur | Cause | Fix |
|--------|-------|-----|
| "API n'a pas répondu" | `make api` pas lancé | `make api` dans un autre terminal |
| Latence > 20 sec | Cold start Mistral 7B | 1er appel charge le modèle (~10s), c'est normal |
| RAGAS baseline < 0.5 | Tuning pas fait | Section W4 du PLAN.md, méthodes documentées dans `docs/evaluation.md` |
| "Chroma not reachable" | Container pas démarré | `make chroma-up` |
| MLX refuse de charger | Pas Apple Silicon | Le projet est Apple-Silicon-only par design |
