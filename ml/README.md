# ML - Multimodal fusion + XAI for learning-disability screening

Trains and explains screening models for **dyslexia** (game-interaction data) and **dysgraphia**
(handwriting images) using early, late and hybrid fusion, with nested cross-validation.

* Method: [../docs/ML_METHODOLOGY.md](../docs/ML_METHODOLOGY.md)
* Results: [../docs/RESULTS.md](../docs/RESULTS.md) (auto-generated)

```bash
pip install -r ml/requirements.txt
python ml/src/dyslexia_pipeline.py
python ml/src/dysgraphia_pipeline.py
python ml/src/make_report.py
```

Outputs: `ml/results/<task>/` (metrics CSV/JSON, figures) and `ml/models/<task>_fusion.joblib`.

Load a trained model:

```python
import joblib, sys; sys.path.insert(0, "ml/src")
bundle = joblib.load("ml/models/dyslexia_fusion.joblib")
fusion = bundle["fusion"]
probs = fusion.predict_proba_all(X)          # dict: model name -> p(risk)
name = "Early fusion | XGBoost"
flag = probs[name] >= fusion.thresholds_[name]
why = fusion.stacking_view_contributions(X)  # per-modality log-odds contributions
```
