# Demo questions — Industrial Knowledge Copilot

> 10 questions de référence, classées par catégorie. À utiliser en démo
> d'entretien (cf. `docs/pitch_entrevue.md`) ou pour smoke-tester le
> pipeline après une release.

---

## 1. Capacité de charge (RAG sur catalogues)

- **Q1** : What is the basic dynamic load rating (C) of a rolling bearing?
  - Attendu : définition de C, mention de newtons, lien avec la durée de vie L10.

- **Q2** : How is the rating life (L10) of a rolling bearing calculated?
  - Attendu : formule L10 = (C/P)^p, facteurs correctifs (température, fiabilité, lubrification).

- **Q3** : What is the difference between static (C0) and dynamic (C) load rating?
  - Attendu : C0 = charge statique admissible sans déformation permanente, C = charge dynamique pour 1 million de tours.

## 2. Lubrification (RAG sur catalogues)

- **Q4** : How do I select a lubricant (grease or oil) for a rolling bearing?
  - Attendu : critères (température, vitesse, charge, environnement), exemples de graisses.

- **Q5** : What is the recommended re-lubrication interval for a deep groove ball bearing?
  - Attendu : intervalle en heures ou mois, en fonction des conditions.

## 3. Montage / démontage (RAG sur catalogues)

- **Q6** : What is the recommended procedure to mount a deep groove ball bearing?
  - Attendu : chauffage, ajustements, force de montage, outillage.

- **Q7** : What is the difference between a loose fit and an interference fit for a bearing on a shaft?
  - Attendu : ajustement glissant vs serré, conséquences sur le démontage.

## 4. Diagnostic (RAG sur catalogues)

- **Q8** : How do I diagnose a bearing defect through vibration analysis?
  - Attendu : FFT, fréquences caractéristiques, seuils de sévérité ISO 10816.

- **Q9** : What are the operating temperature limits for a rolling bearing?
  - Attendu : plage normale, alerte, alarme, influence du lubrifiant.

## 5. Hors-scope (test d'honnêteté)

**Aucune de ces questions ne doit recevoir une réponse confiante.**

- **Q10** : What is the phone number of NASA support?
  - Attendu : "I don't know from the available data."

---

## Comment lancer la démo (5 minutes)

```bash
# 1. Une seule fois (à faire avant l'entretien)
cd "/Users/patriceduclos/Library/CloudStorage/GoogleDrive-patrice.noel.duclos@gmail.com/Mon Drive/PromptAI/rag-copilot"
make setup && make pull-models
# Drop Schaeffler / SKF / NTN-SNR catalogues into data/raw/pdf/
make chroma-up
make ingest && make eval-dataset

# 2. Le jour J, dans 2 terminaux
make api    # terminal 1 — FastAPI :8000
make ui     # terminal 2 — Streamlit :8501

# 3. Poser les questions dans l'ordre 1 → 10 dans l'onglet Chat
#    (pas besoin de toggle "agent" — il a été supprimé en juillet 2026
#    (le tool agent a été retiré en juillet 2026)
```

## Checklist "démo qui marche" (à checker 1h avant)

- [ ] `/health` → `status: ok`, `chroma: True`, `mlx_ready: True`
- [ ] `make eval` produit un snapshot récent dans `reports/`
- [ ] L'UI charge bien l'inventaire (Schaeffler + SKF + NTN-SNR visibles)
- [ ] Q1 répond en < 5 sec avec une définition de C
- [ ] Q4 mentionne les critères température / vitesse / charge
- [ ] Q10 dit "I don't know"
- [ ] Une question FR reçoit une réponse FR (mirror)

## Erreurs courantes et fix rapide

| Erreur | Cause | Fix |
|--------|-------|-----|
| "API n'a pas répondu" | `make api` pas lancé | `make api` dans un autre terminal |
| Latence > 20 sec | Cold start Qwen2.5-7B | 1er appel charge le modèle (~10s), c'est normal |
| RAGAS baseline < 0.5 | Tuning pas fait | Section W4 du PLAN.md, méthodes documentées dans `docs/evaluation.md` |
| "Chroma not reachable" | Container pas démarré | `make chroma-up` |
| MLX refuse de charger | Pas Apple Silicon | Le projet est Apple-Silicon-only par design |
