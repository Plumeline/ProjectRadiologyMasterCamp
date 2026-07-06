# Guide de soutenance — preuves exécutables

> Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise.

Ce document liste les commandes à exécuter en direct et les sorties attendues. Le cahier des
charges exige des **preuves** : commandes exécutables, sorties JSON, métriques, erreurs,
avertissements et limites. Tout tourne sur CPU (l'inférence MedGemma a été faite une fois hors-ligne).

---

## 0. Avant la soutenance — smoke test (doit être vert)

```bash
pip install -r requirements-test.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python -m compileall -q src api app eval finetuning tests
```

Attendu : `8 passed`, `compileall` sans erreur.

---

## 1. Baseline reproductible (évaluation chiffrée)

```bash
python eval/run_evaluation.py --mode baseline --out-dir /tmp/eval_out --db-path /tmp/evidence.sqlite
```

Attendu : accuracy **80,0 %**, JSON valide 100 %, warning 100 %.

---

## 2. L'amélioration mesurée — les 4 versions

```bash
python eval/run_evaluation.py --mode baseline    --out-dir /tmp/eval_out --db-path /tmp/evidence.sqlite
python eval/run_evaluation.py --mode improved    --out-dir /tmp/eval_out --db-path /tmp/evidence.sqlite
python eval/run_evaluation.py --mode improved_v2 --out-dir /tmp/eval_out --db-path /tmp/evidence.sqlite
python eval/run_evaluation.py --mode improved_v3 --out-dir /tmp/eval_out --db-path /tmp/evidence.sqlite
```

| mode | accuracy | faux négatifs | histoire |
|---|---|---|---|
| baseline | 80,0 % | 6 | équilibré mais faible |
| improved (v1) | 66,7 % | 10 | trop permissif |
| improved_v2 | 36,7 % | 0 | trop strict |
| **improved_v3** | **86,7 %** | **1** | **équilibre trouvé** |

Message clé : une consigne « prudente » plausible (v1) a rendu le système plus dangereux ;
seule la mesure l'a révélé. v3 dépasse la baseline sur toutes les métriques.

---

## 3. Démo web — upload, warning, JSON

### API FastAPI
```bash
uvicorn api.main:app --reload
```
Dans un autre terminal :
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@data/sample_images/CXR_SYN_002_suspected_opacity.png"
```
Attendu : JSON avec `predicted_class = suspected_opacity`, une confiance, des observations, une
justification, des limites et l'avertissement. Le modèle affiché est `google/medgemma-4b-it`.

### Interface Streamlit
```bash
streamlit run streamlit_app.py
```
Déposer une image, sélectionner le mode `improved_v3` (par défaut), montrer la sortie et
l'avertissement à l'écran.

---

## 4. Preuves à montrer (fichiers)

- `docs/metrics_summary.csv` — métriques des 4 versions.
- `eval/error_register.csv` — 10 cas d'erreur commentés (résiduels v3 + ablations v1/v2).
- `docs/figures/confusion_matrices.png` — les 4 matrices de confusion.
- `docs/figures/metric_trajectory.png` — la trajectoire baseline → v1 → v2 → v3.
- `/tmp/evidence.sqlite` — journal des runs (généré à la volée, hors dépôt).
- `docs/rapport_final.pdf` — le rapport complet.

---

## 5. Montrer aussi les échecs (règle de soutenance)

Ne jamais montrer uniquement des réussites. Points à assumer ouvertement :

- **Images synthétiques**, non cliniques : les scores mesurent le pipeline, pas la médecine.
- **Petit échantillon** (30 images) : valeurs illustratives.
- **Confiance non calibrée** : valeurs peu variées (0,95 / 0,20).
- **v1 et v2 ont échoué** : c'est la preuve que la prudence doit être mesurée, pas supposée.

---

## Questions probables du jury — réponses courtes

- *« Ça tourne sans GPU ? »* Oui : l'inférence MedGemma est faite une fois hors-ligne en 4-bit ;
  l'app lit les prédictions en cache. Architecture volontaire, documentée.
- *« Pourquoi v1 est pire que la baseline ? »* La consigne anti-artefact a poussé le modèle à
  écarter les opacités en `normal` confiant (faux négatifs surconfiants). Diagnostic via matrice
  de confusion, corrigé en v3.
- *« Vos données sont-elles réelles ? »* Non, synthétiques et autorisées par le cahier des charges.
  C'est notre principale limite documentée.
