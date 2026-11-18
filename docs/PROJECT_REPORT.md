# NeuroQuest: Explainable Multimodal AI for Game-Based Screening of Dyslexia, Dysgraphia and Dyscalculia

*Project report content. Copy each chapter into your report template. Figures referred to below are in
`ml/results/` and `docs/paper/figures/`.*

---

## Abstract

Specific learning disorders such as dyslexia (reading), dysgraphia (writing) and dyscalculia (mathematics) are
often identified late, after years of academic difficulty. Our project, NeuroQuest, is a game-based screening
platform. Children complete short interactive tasks, and the system estimates their risk for each disorder.

The original platform scored the game with fixed hand-written rules. In this work we replaced and supported those
rules with an explainable, multimodal machine-learning system. For each disorder we grouped the available evidence
into modalities (task accuracy, interaction behaviour, response time, handwriting geometry, deep visual features
from a pretrained CNN, and number-sense measures). We compared four families of models: single-modality baselines,
early fusion, late fusion and hybrid fusion. All of them were evaluated with nested cross-validation and metrics
suitable for imbalanced data.

Fusion improved dyslexia screening from a ROC-AUC of 0.804 to 0.862 on 5,039 children, and reached 0.933 for
dysgraphia on 249 handwriting samples. Explainable-AI techniques (SHAP and occlusion maps) show which modality,
feature, game question or handwriting region drives each prediction.

The trained models were integrated into the game through a new ML service. We added a model dashboard inside the
web application and a safe retraining loop that improves the models as confirmed outcomes are collected.

---

## Chapter 1: Introduction

### 1.1 Background
Learning disorders affect an estimated 5–15% of school-age children. Early support makes a large difference, but
formal assessment by a psychologist is expensive, slow and usually happens only after a child has struggled for a
long time. Screening games offer a scalable first step. A child plays a set of short tasks, and the system flags
those who may need a professional assessment.

### 1.2 Problem Statement
The existing NeuroQuest platform already had a working game and backend, but its scoring had five weaknesses:
- **Fixed rules.** Each answer was marked right or wrong, and fixed thresholds turned the score into a risk level
  (for example, "score ≥ 80 → low risk"). Nothing was learned from data.
- **One kind of evidence at a time.** How a child answered (timing, clicking pattern, stroke quality) was largely
  ignored.
- **No validation.** There was no measurement of how accurate the scoring actually was.
- **No explanation.** The report gave fixed text per risk level and could not say *why* a child was flagged.
- **No way to improve.** The system could not learn from children whose real outcome later became known.

### 1.3 Objectives
1. Train machine-learning models for dyslexia, dysgraphia and dyscalculia using the available datasets.
2. Use a **multimodal** approach, and compare **early fusion**, **late fusion** and **hybrid fusion** against
   single-modality baselines.
3. Evaluate the models **rigorously and honestly**, without data leakage, using metrics suited to imbalanced data.
4. Make every prediction **explainable** (XAI).
5. **Integrate** the best models into the game with minimal changes to it.
6. Build a **dashboard** that presents all experiments, results and the models in use.
7. Add a **retraining pipeline** so the models improve as real labelled data is collected.

### 1.4 Scope
The system is a **screening aid**, not a diagnostic tool. Its output is a risk level that suggests whether a
professional assessment is advisable.

---

## Chapter 2: Literature Review

**Game-based dyslexia screening.** Rello et al. (2020) showed that a 32-question online game (Dytective) can predict
dyslexia risk from how children interact with it. They released the desktop and tablet datasets that we use.

**Dysgraphia detection.** A recent scoping review (Fallah et al., 2025) divides existing work into two groups:
- *online* methods, which use pen movement recorded on a tablet;
- *offline* methods, which analyse scanned handwriting with either hand-crafted features or convolutional neural
  networks.

Few studies combine both feature types, and few check for confounds in how samples were collected.

**Dyscalculia detection.** AI-based dyscalculia screening is still rare and data-poor (Bhushan et al., 2024).
Cognitive research identifies several markers:
- weaker approximate number sense (Piazza et al., 2010);
- an immature, logarithmic "mental number line" (Siegler & Opfer, 2003);
- largely intact subitizing, i.e. instantly recognising small quantities (Decarli et al., 2020).

**Multimodal fusion.** Fusion strategies are commonly grouped into three types (Baltrušaitis et al., 2019):
- *early fusion*: combining features before training one model;
- *late fusion*: combining the decisions of separate models;
- *hybrid fusion*: a mix of the two.

**Explainable AI.** SHAP (Lundberg & Lee, 2017) attributes each prediction to its input features. Occlusion
sensitivity (Zeiler & Fergus, 2014) shows which image regions a CNN relies on.

**Gap.** Most existing screening systems use one modality and one model. They report plain accuracy on a single
train/test split, which is misleading on imbalanced data. They give no explanation and stop at a trained model,
with no deployment or improvement loop. Our project addresses each of these points.

---

## Chapter 3: Existing System and Its Limitations

**Existing system.** NeuroQuest consisted of:
- a React web game with 15 tasks, five per disorder. Examples include mirror-letter discrimination, rhyming, letter
  tracing, word spacing, subitizing, number line and addition;
- a Spring Boot backend that stored sessions in MongoDB;
- a rule-based scoring service that produced a report with a risk level per disorder.

**Limitations identified:**

| Limitation | Effect |
|---|---|
| Rule-based scoring with fixed thresholds | Not learned from data; accuracy unknown |
| Each answer judged in isolation | Rich behavioural signals ignored |
| Motor "jitter" hard-coded to 12 or 25 | Handwriting-quality measure was not real |
| No explanation beyond fixed text | Teachers and parents cannot see *why* |
| No use of outcome data | The system can never improve |
| Datasets in the repository unused | No evidence base behind the scoring |

---

## Chapter 4: Proposed System

### 4.1 Overview
The proposed system keeps the game and backend and adds three components:
1. an **offline ML pipeline** that trains, evaluates and explains multimodal fusion models;
2. an **ML service** that scores each game session with the best model for each disorder and handles retraining;
3. a **Model Dashboard** page inside the web application.

### 4.2 System Architecture

```
 Child ──plays──► Game (React, 17 tasks) ──answers + raw data──► Backend (Spring Boot) ──► MongoDB
                                                                     │                     ▲
                                                           session   ▼                     │ labels
                                                   ML Service (FastAPI) ◄── Model Registry  │
                                                                     │          ▲          │
                                                   risk + explanation│          │promote   │
                                                                     ▼          │if better │
                                                       Report (rules + AI)   Retraining ◄──┘
                                                                              (champion vs challenger)

 Offline: Research datasets ──► Modality features ──► Fusion study (nested CV) ──► XAI ──► initial models
```

**Online flow**
1. The child plays the game, and each answer is sent to the backend.
2. On submission, the backend computes the rule-based report, then asks the ML service to score the session.
3. Where a model trained on real data can read the session, its risk level is used for that disorder. The rules
   remain as a fallback, so the game always works.
4. The report shows the final risk levels and an "AI Model Assessment" panel that explains each model result.

**Learning flow**
1. A specialist later records the child's confirmed outcome.
2. When enough new confirmed outcomes arrive, the ML service retrains.
3. The new model goes live only if it performs better than the current one on children it has never seen.

### 4.3 Modules
| Module | Responsibility |
|---|---|
| Data preprocessing | Cleaning, normalisation and handling of missing values |
| Feature extraction | Building each modality's features (tabular, handwriting geometry, CNN, number sense) |
| Fusion framework | Unimodal, early, late and hybrid models; nested cross-validation |
| Model selection | Choosing the deployed model by a documented rule |
| Explainability | SHAP, permutation importance and occlusion maps |
| ML service | Scoring, model registry, retraining and dashboard data |
| Backend integration | Calling the ML service, storing raw data and labels |
| Game extensions | Two new tasks, real jitter measurement, labelling screens |
| Model Dashboard | Web page showing all experiments, decisions and live model status |

---

## Chapter 5: Datasets and Preprocessing

### 5.1 Datasets
| Dataset | Children / samples | At risk | Description |
|---|---|---|---|
| Dyslexia (Dyt desktop + tablet) | 5,039 | 540 (10.7%) | 32-question gamified test: clicks, hits, misses, accuracy per question, plus age, gender, language |
| Dysgraphia handwriting | 249 images | 114 | Scanned sentences labelled potential / low-potential dysgraphia |
| Dyscalculia (Castaldi et al., 2024) | 37 | 15 | Time-perception precision, age, IQ, gender |
| Dyscalculia (synthetic) | 3,000 | 401 | Simulated game data for the number tasks (pipeline testing only) |

We searched for a public dyscalculia dataset that matched the game's tasks. The best match, Lamb et al. (2024),
has no public data. The only usable real dataset was the small Castaldi et al. study, so we also generated a
synthetic dataset to build and test the pipeline (Section 6.6).

### 5.2 Preprocessing and Data-Quality Fixes
**Dyslexia data**
- About 5,000 accuracy and miss-rate values had lost their decimal point (for example, 0.875 was stored as 875).
  We recomputed them from the hit, miss and click counts.
- Question 29 was never shown on tablets, so we removed it.
- Other missing tablet values are filled during training, with indicator columns that mark which values were
  missing.

**Dysgraphia images**
- Converted to black and white.
- Ruled lines removed, and strokes that crossed a line reconnected.
- Rescaled so the median letter height is 40 pixels. Raw letter size in pixels alone predicted the label with
  AUC 0.90. Because the scan resolution is unknown, a model could have learned the *scanner* instead of the
  *handwriting*. Normalising the scale removes this shortcut.

**Dyscalculia data**
- Removed the `math` column, because it is the diagnostic criterion itself.
- Removed the `reading` column: it is 46% missing, and both its values and its missingness separate the groups.
  Keeping either column would have produced a falsely perfect model.

---

## Chapter 6: Methodology

### 6.1 Modalities
| Disorder | Modalities |
|---|---|
| Dyslexia | **Task accuracy** (what the child got right), **Interaction behaviour** (clicks, misses, miss rate), **Demographics** |
| Dysgraphia | **Handwriting geometry** (25 interpretable measures), **Deep visual features** (EfficientNet-B0 CNN) |
| Dyscalculia (real) | **Time perception**, **Cognitive profile** |
| Dyscalculia (game / synthetic) | **Task accuracy**, **Response time**, **Number-sense profile**, **Demographics** |

**Handwriting geometry features** describe letter-size variability, baseline drift, spacing regularity, slant,
stroke width, contour tremor and broken strokes. All lengths are measured relative to the child's own letter size.

**Deep visual features** come from a CNN pretrained on ImageNet (EfficientNet-B0). The CNN reads the handwriting
in overlapping square tiles, and the results are averaged.

**Number-sense features** are:
- precision of quantity comparison (the Weber fraction);
- number-line error, and whether the number line is linear or logarithmic;
- subitizing span;
- response times.

### 6.2 Models and Fusion Strategies
- **Unimodal (baseline):** one model per modality.
- **Early fusion:** all modalities' features are joined into one input, then a single classifier is trained
  (logistic regression, random forest, XGBoost, MLP or SVM).
- **Late fusion:** one model per modality; their risk scores are then combined in one of three ways:
  - a simple average (soft voting);
  - an average weighted by each modality's accuracy;
  - a small logistic-regression model that learns the combination (stacking).
- **Hybrid fusion:** stacking over the per-modality scores together with the early-fusion score.

Class imbalance is handled with class weights. Each model's decision threshold is set to maximise balanced
accuracy, using training data only.

### 6.3 Evaluation Protocol
We used **nested cross-validation**:
- The data is split into outer folds.
- Inside each training part, a second (inner) split makes every choice: which algorithm to use per modality, the
  fusion weights, the stacking model and the decision threshold.
- The outer test folds are used only for scoring. No test child influences any decision.

**Metrics:** ROC-AUC, balanced accuracy, sensitivity (the share of at-risk children caught), specificity (the share
of others correctly cleared), F1 and PR-AUC. Results are reported as mean ± standard deviation over test folds.

Plain accuracy is shown but not used to rank models. For dyslexia, always answering "no risk" already scores 89%.

### 6.4 Model Selection Rule
We do not simply deploy the top scorer. Among all models within **0.005 ROC-AUC** of the best, we deploy the
**simplest**, preferring fewer stages and simpler algorithms. Differences smaller than this are inside the normal
fold-to-fold variation, so extra complexity would buy nothing measurable.

### 6.5 Explainable AI
Explanations work at three levels:
1. **Which modality matters:** each modality's share of SHAP importance, and the drop in accuracy when a whole
   modality is shuffled.
2. **Which features or game questions matter:** SHAP values for every feature. For dyslexia, the values are summed
   per question to rank the game's questions.
3. **Why this child was flagged:**
   - a SHAP breakdown of the individual prediction;
   - the per-modality contribution to the late-fusion decision;
   - occlusion heatmaps on handwriting images.

### 6.6 Synthetic Dyscalculia Data
Each simulated child has hidden abilities: number sense, arithmetic skill, speed, attention and subitizing.
Children with dyscalculia are given weaker number sense and arithmetic, following published research.

The simulation also adds realistic complications:
- skills improve with age;
- some children without dyscalculia have attention problems that make them look similar to those with it;
- 5% of labels are deliberately wrong.

Each simulated child then "plays" the game's five number tasks over many trials. Because we know what was planted,
this data also tests whether the explanations are correct.

---

## Chapter 7: Implementation

### 7.1 Technology Stack
| Layer | Technology |
|---|---|
| Game (frontend) | React 19, Vite |
| Backend | Java 21, Spring Boot 3.3, MongoDB 8 (Docker) |
| ML pipeline | Python 3.10: scikit-learn, XGBoost, TensorFlow (EfficientNet-B0), OpenCV, SHAP |
| ML service | FastAPI, Uvicorn |
| Dashboard | React page with SVG charts, fed by the ML service |

### 7.2 ML Pipeline (`ml/src/`)
- `fusion.py`: a single reusable framework for all fusion strategies and nested cross-validation.
- `dyslexia_pipeline.py`, `dysgraphia_pipeline.py`, `dyscalculia_pipeline.py`: one experiment per disorder.
- `dysgraphia_features.py`: image preprocessing, geometry features and CNN embeddings.
- `dyscalculia_features.py`: turns trial-level game logs into features. The same code is used for synthetic and
  real game data.
- `model_selection.py`: the deployment selection rule.
- `main_model.py`: one entry point that scores all three disorders and maps probabilities to risk levels.

### 7.3 ML Service (`ml/service/`)
- **Scoring:** turns a game session into model inputs and returns, per disorder, a probability, a risk level,
  the model version and the per-modality contributions.
- **Model registry:** tracks every model version and which one is live.
- **Retraining:** champion/challenger retraining from confirmed outcomes.
- **Dashboard data:** serves all tables and figures to the Model Dashboard page.

### 7.4 Game and Backend Changes (kept minimal)
| Change | Purpose |
|---|---|
| New task Q16, "Copy the sentence" | Captures a handwriting image for the dysgraphia model |
| New task Q17, "Number-sense round" | Short repeated number trials for the dyscalculia model |
| Real stroke jitter in letter tracing | Replaces the hard-coded value |
| Optional "existing diagnosis" at registration | Parent-reported training labels |
| "Confirmed outcome" buttons in session history | Specialist training labels |
| "AI Model Assessment" panel in the report | Shows each model's result and what drove it |
| Backend calls the ML service on submit | Model risk replaces rules where a real-data model applies |
| New endpoints for labels and training data | Feed the retraining loop |

The original 15 tasks are unchanged.

### 7.5 Model Dashboard
A page inside the web app shows:
1. the model each disorder uses now, its scores, its risk thresholds and why it was chosen;
2. the live retraining status and version history;
3. a plain-language explanation of the four model families;
4. a story for each experiment: what each fusion type gave and which model was picked;
5. an interactive comparison of every model, explanation charts and figures, and the full metrics table.

### 7.6 Retraining Policy
- Training starts once a disorder has at least **40 confirmed sessions**, with at least **10** in each group.
- Retraining is checked after every **20 new** confirmed outcomes.
- A new model goes live only if it scores at least **0.65 ROC-AUC** and beats the live model by **0.01** on
  children the live model never saw.
- Specialist labels take priority over parent-reported ones.
- Every version is kept, so the system can roll back.

---

## Chapter 8: Results and Analysis

### 8.1 Effect of Fusion
Each cell shows the best model of that family (ROC-AUC, mean ± sd). **Bold** marks the deployed model.

| Experiment | Unimodal | Early fusion | Late fusion | Hybrid |
|---|---|---|---|---|
| Dyslexia | 0.804 ± 0.039 | **0.862 ± 0.023** | 0.846 ± 0.026 | 0.863 ± 0.020 |
| Dysgraphia | 0.920 ± 0.042 | **0.933 ± 0.034** | 0.923 ± 0.041 | 0.923 ± 0.041 |
| Dyscalculia (real, n=37) | **0.765 ± 0.174** | 0.766 ± 0.162 | 0.696 ± 0.211 | 0.697 ± 0.194 |
| Dyscalculia (synthetic) | 0.731 ± 0.008 | **0.741 ± 0.020** | 0.741 ± 0.020 | 0.739 ± 0.016 |

**Dyslexia.** Fusion clearly helped, adding +0.058 ROC-AUC over the best single modality. Early fusion beat late
fusion: it can combine *what* a child answered with *how* they interacted on the same question, while late fusion
only sees each modality's final score. Hybrid fusion was 0.001 higher, far inside the noise, so we deploy the
simpler early-fusion XGBoost.

The deployed model has balanced accuracy 0.778, sensitivity 0.811 and specificity 0.745. Its pooled ROC-AUC is
0.861, with a 95% confidence interval of 0.845–0.876. Results were better on desktop (0.883) than on tablet
(0.797), where some questions are missing.

**Dysgraphia.** Early fusion reached **0.933 ROC-AUC** and **0.850 balanced accuracy** (sensitivity 0.863,
specificity 0.837). That is +0.013 over the CNN alone. Both modalities look at the same handwriting, which explains
the small gain.

**Dyscalculia (real).** Time perception alone was the best choice. Late and hybrid fusion were worse: with only 37
children, the combiner overfits, and the second modality (age, IQ, gender) carries no signal.

**Dyscalculia (synthetic).** All fusion types gave a small gain of about +0.01. The best possible AUC given the
deliberate label noise is 0.847. These numbers test the pipeline only and are not a real-world accuracy.

### 8.2 Explainability Results
**Dyslexia.** Modality importance: interaction behaviour 42.6%, task accuracy 35.3%, demographics 22.1%. The
strongest features were native language, age, and click counts on questions 28, 24 and 25. The most informative
game questions were Q24, Q28, Q25, Q20 and Q22, which is directly useful for redesigning the game.

**Dysgraphia.**
- Shuffling the CNN features lowered held-out AUC by 0.340; shuffling the geometry features lowered it by 0.041.
- The key geometry features were the spread of letter heights, spacing variability, sentence length, contour shape
  and broken strokes.
- The heatmaps show that the model partly relies on pen-stroke thickness, so the score may be partly inflated by
  differences in how samples were collected.

**Dyscalculia.**
- On real data, time-perception precision was the strongest signal, matching the original study.
- On synthetic data, the explanations correctly ranked the planted effects: arithmetic first, number sense next,
  subitizing last. This confirms that the XAI layer reflects real model behaviour.

### 8.3 Models Deployed in the Game
| Disorder | Model in the game | Input | Effect on the report |
|---|---|---|---|
| Dyslexia | Rule-based (for now) | Game questions Q1–Q5 | Rules |
| Dysgraphia | Early fusion, logistic regression (AUC 0.933) | Q16 handwriting image | Sets the risk level |
| Dyscalculia | Early fusion, logistic regression (synthetic-trained) | Q17 number round | Shown only |

The dyslexia model needs the 32 Dyt test questions, which the game does not have. Dyslexia therefore uses rules
until enough labelled game sessions exist to train a game-specific model.

---

## Chapter 9: Testing

| Test | Result |
|---|---|
| Data integrity checks (corrupted values, leakage, resolution bias) | Issues found and fixed (Section 5.2) |
| Feature sanity check: single-feature AUCs after normalisation | No single feature above about 0.80; shortcut removed |
| Nested cross-validation for all models | Every reported number is a held-out estimate |
| XAI check on synthetic data with known effects | Explanations rank the planted effects correctly |
| End-to-end API test: register → submit (with handwriting image and number round) → report → label → training data | Passed; the model result was applied to the report |
| Retraining self-test on 160 simulated sessions (two rounds) | Round 1 trained and promoted game models; round 2 kept the live models because no challenger beat them by the margin |
| Fallback test: ML service unavailable | Game completes with the rule-based report |
| Frontend build and lint | Builds cleanly |

---

## Chapter 10: Improvements Over the Existing System

| Area | Before | After |
|---|---|---|
| Scoring | Fixed rules | Trained models per disorder, with rules as fallback |
| Evidence used | Right/wrong per question | Several modalities fused per child |
| Validation | None | Nested cross-validation and imbalance-aware metrics |
| Explanation | Fixed text | SHAP, modality importance, game-question ranking, heatmaps, per-child explanation |
| Motor measure | Hard-coded jitter | Measured from stroke points |
| Data use | Datasets unused | Cleaned and used; quality issues fixed |
| Model choice | — | Documented rule: simplest model within noise |
| Learning | Never improves | Safe champion/challenger retraining from confirmed outcomes |
| Visibility | — | Model Dashboard page and full documentation |

**Compared with typical published work**, the main differences are:
- several modalities instead of one;
- leakage-free nested evaluation instead of a single split;
- explanations at three levels;
- a deployed system that keeps learning, instead of a standalone model.

---

## Chapter 11: Limitations and Ethical Considerations

**Limitations**
- No child appears in more than one dataset, so the three disorders are modelled separately. A single
  cross-disorder model is not yet possible.
- The dysgraphia dataset is small, and its score may be inflated by how the samples were collected. The model
  learned from paper scans, while the game collects handwriting on a screen.
- The real dyscalculia dataset is very small (37 children). The deployed dyscalculia model is trained on synthetic
  data and is shown for information only.
- The retraining loop has been tested only on simulated sessions. Its real benefit depends on collecting confirmed
  outcomes.
- The labels are screening-level and not always formal clinical diagnoses.

**Ethical considerations**
- The system is a screening aid, not a diagnosis. Every report recommends professional assessment where risk is
  found.
- Handwriting samples and outcome labels are sensitive data about children. Real use requires:
  - parental consent;
  - authentication on the labelling and training-data endpoints;
  - a data-retention policy.
- A model trained on synthetic data never changes a child's report.

---

## Chapter 12: Conclusion and Future Work

**Conclusion.** We turned a rule-based screening game into an explainable, multimodal AI system. Fusing different
kinds of evidence improved dyslexia screening from 0.804 to 0.862 ROC-AUC and reached 0.933 for dysgraphia. The
evaluation was designed to prevent data leakage, and the data-quality fixes stop the models from relying on
shortcuts. The explanations show what drives each decision. The models now run inside the game, are presented on
a dashboard, and can improve safely as confirmed outcomes are collected.

**Future work**
1. Collect labelled game sessions to train game-specific models, especially for dyslexia, and to validate the
   handwriting model on screen input.
2. Extend the number-sense round into a validated dyscalculia test battery.
3. Build a cross-disorder model once children complete all tasks and have confirmed outcomes.
4. Study fairness across age, language and device (desktop vs tablet).
5. Add authentication, consent management and data-retention controls before real deployment.

---

## Appendix A: How to Run

1. Start MongoDB: `docker run -d --name neuroquest-mongo -p 27017:27017 mongo:8.0`
2. Start the backend: `cd backend` then `mvn spring-boot:run` (port 8080).
3. Start the ML service: `python -m uvicorn app:app --app-dir ml/service --port 8000`
4. Start the game: `cd frontend`, `npm install`, then `npm run dev`. Open the printed URL; the Model Dashboard is
   in the top bar.
5. Retrain the research models (optional):
   ```
   python ml/src/dyslexia_pipeline.py
   python ml/src/dysgraphia_pipeline.py
   python ml/src/dyscalculia_synthetic.py
   python ml/src/dyscalculia_pipeline.py
   python ml/src/main_model.py
   ```

## Appendix B: Project Structure

```
backend/                 Spring Boot API (scoring, sessions, labels, ML client)
frontend/                React game, report, Model Dashboard
ml/src/                  Research pipeline (features, fusion, XAI, model selection)
ml/service/              ML service (scoring, registry, retraining, dashboard API)
ml/results/              Metrics and figures for every experiment
ml/models/               Trained models and registry
datasets/                Dyslexia, dysgraphia, dyscalculia (real + synthetic)
docs/                    Methodology, results, integration guide, IEEE paper
```

## Appendix C: API Summary

| Service | Endpoint | Purpose |
|---|---|---|
| Backend | `POST /api/assessments/register` | Start a session (optional known diagnoses) |
| Backend | `POST /api/assessments/{id}/submit` | Submit answers; returns the report with model results |
| Backend | `POST /api/assessments/{id}/labels` | Record a confirmed outcome |
| Backend | `GET /api/assessments/training-data` | Labelled sessions for retraining |
| ML service | `POST /api/score` | Score one session |
| ML service | `POST /api/retrain`, `/api/retrain/maybe` | Retrain now, or when enough new labels exist |
| ML service | `GET /api/models` | Model registry and retraining history |
| ML service | `GET /api/dashboard` | Data for the Model Dashboard |

## References
1. L. Rello, R. Baeza-Yates, A. Ali, J. P. Bigham, M. Serra, "Predicting risk of dyslexia with an online gamified
   test," *PLoS ONE*, 15(12): e0241687, 2020.
2. S. Bhushan et al., "AI-Enhanced Dyscalculia Screening: A Survey of Methods and Applications for Children,"
   *Diagnostics*, 14(13): 1441, 2024.
3. A. Fallah, Y. ZandiyeVakili, H. Sajedi, "AI-driven approaches for dysgraphia diagnosis using online and offline
   handwriting data: A comprehensive scoping review," *PLoS ONE*, 20(12): e0328722, 2025.
4. E. Castaldi et al., "Auditory time perception impairment in children with developmental dyscalculia," dataset,
   Zenodo record 10966052, 2024.
5. M. Piazza et al., "Developmental trajectory of number acuity reveals a severe impairment in developmental
   dyscalculia," *Cognition*, 116(1): 33–41, 2010.
6. R. S. Siegler, J. E. Opfer, "The development of numerical estimation," *Psychological Science*, 14(3): 237–243,
   2003.
7. G. Decarli et al., "Impaired large numerosity estimation and intact subitizing in developmental dyscalculia,"
   *PLoS ONE*, 2020.
8. T. Baltrušaitis, C. Ahuja, L.-P. Morency, "Multimodal machine learning: A survey and taxonomy," *IEEE TPAMI*,
   41(2): 423–443, 2019.
9. S. M. Lundberg, S.-I. Lee, "A unified approach to interpreting model predictions," *NeurIPS*, 2017.
10. M. D. Zeiler, R. Fergus, "Visualizing and understanding convolutional networks," *ECCV*, 2014.
11. S. Varma, R. Simon, "Bias in error estimation when using cross-validation for model selection," *BMC
    Bioinformatics*, 7: 91, 2006.
12. T. Chen, C. Guestrin, "XGBoost: A scalable tree boosting system," *KDD*, 2016.
13. M. Tan, Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," *ICML*, 2019.
14. Dysgraphia handwriting dataset: add the source of `datasets/DATASET DYSGRAPHIA HANDWRITING`.

*Check each reference against the original before submitting.*
