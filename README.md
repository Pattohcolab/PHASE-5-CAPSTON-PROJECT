# Predicting Acute Food Insecurity Risk in Kenyan Counties using Climate and Market Signals

**Program:** Moringa School - Data Science Capstone Project

**Team:** Data Alchemists
- Patrick Njoroge
- Chelimo Ivy
- Alex Njihia
- Joel Andrew
- Halima Ahmed
- Rebecca Monchari

---

## Overview

Kenya's smallholder farmers and pastoralists face persistent food insecurity driven by rainfall failure and rising staple food prices. Official IPC (Integrated Food Security Phase Classification) assessments, the trigger for humanitarian response, are published 4–8 weeks *after* conditions have already deteriorated.

This project builds a **binary classification model** that predicts whether a Kenyan county will reach **IPC Phase 3 (Crisis) or worse** in a given month, using satellite-derived rainfall anomalies (CHIRPS) and WFP food price records as **lagged, leading indicators**, giving Kenya's National Drought Management Authority (NDMA) an early-warning signal ahead of the official IPC cycle.

**Prediction question:** Can satellite rainfall anomalies and food price data predict a county's IPC Phase 3+ status *before* the official classification is published?

## Data Sources

| # | Dataset | Source | Raw Rows | Role |
|---|---|---|---|---|
| 1 | FEWS NET IPC Classifications | HDX / FEWS NET | 27,694 | Target variable |
| 2 | CHIRPS Rainfall Indicators | HDX / WFP | 132,678 | Climate features |
| 3 | WFP Food Prices (Kenya) | HDX / WFP | 26,745 | Market features |

Expected local files (not included in this repo — see [Data Access](#data-access)).

## Repository Structure

```
.
├── notebook.ipynb                          # main CRISP-DM notebook (Phases 1-7)
│
├── data/
│   ├── ipcphase.xlsx                       # FEWS NET IPC classifications (raw)
│   ├── ken-rainfall-subnat-full.csv        # CHIRPS rainfall (raw)
│   ├── wfp_food_prices_ken.csv             # WFP food prices (raw)
│   └── model_ready.csv                     # final merged, feature-engineered dataset
│
├── Data_Alchemists_Capstone_Proposal.docx  # Original project proposal
│
├── models/                                 # deployment artifacts (Phase 7)
│   ├── deployed_model_Random_Forest.joblib
│   ├── deployed_imputer.joblib
│   ├── feature_columns.pkl
│   └── deployment_metadata.json
│
├── README.md
├── requirements.txt
```

> `model_ready.csv` is the single canonical modelling dataset produced by the notebook (Section 3.8) — it is written to `data/` and used directly as the input to Phase 5 modelling. No separate intermediate file is needed.

## Methodology (CRISP-DM)

| Phase | Description |
|---|---|
| 1. Business Understanding | Problem framing, stakeholders, success criteria |
| 2. Data Understanding | Structure, dtypes, missingness for all three raw datasets |
| 3. Data Preparation | Cleaning, geographic harmonization to 47 counties, target construction, lag alignment, merge |
| 4. EDA | Univariate, bivariate, and multivariate analysis of climate/price signals vs. crisis |
| 5. Modeling | Chronological train/validation/test split, 4 candidate models, threshold tuning |
| 6. Evaluation | Test-set metrics, calibration, confidence intervals, SHAP interpretability |
| 7. Deployment | Monthly scoring pipeline (implemented) + serving/monitoring architecture (designed, not implemented — see [Scope & Limitations](#scope--limitations)) |

Key technical decisions:
- **Geographic harmonization:** IPC (compound location strings), CHIRPS (PCODE), and WFP (admin names) were all mapped to Kenya's 47 official counties, including historical alias handling for post-devolution name changes.
- **One-month lag** applied to rainfall/price features relative to IPC reporting date, to prevent leakage and simulate a genuine early-warning scenario.
- **Chronological (not random) split:** train on pre-2021 data, validate on 2021, test on 2022+ — touched exactly once at the end.
- **Threshold selection rule:** sweep thresholds on validation data, keep those with recall ≥ 0.75 (humanitarian floor), and pick the one that maximizes precision.
- **Four deployment-ready candidates** compared under identical rules: Logistic Regression (class-weighted), Random Forest (class-weighted), XGBoost (`scale_pos_weight`), and LightGBM (`is_unbalance`).

## Key Results

- **Winning model:** Random Forest — Recall 0.804, Precision 0.752, F1-macro 0.858, PR-AUC 0.818, ROC-AUC 0.944 on the untouched 2022+ test set, at the validation-selected threshold (0.52). Roughly 3 crises caught for every 1 false alarm.
- **Top predictors (SHAP):** structural vulnerability (`is_asal`), recent crisis history (`prev_crisis`), 3-month rainfall anomaly (`r3q_mean`), and 1-month rainfall anomaly (`rfq_mean`) dominate; staple food price level is a weaker, supporting signal.
- **Live demonstration (October 2023):** of 47 counties scored, 18 were flagged; all 8 counties that actually reached Crisis+ that month were caught and ranked in the top 8 by predicted probability.

## Scope & Limitations

- **Implemented:** model training and selection, artifact persistence, a production-ready scoring function (`score_new_month`), and a live demonstration scoring an actual held-out month end-to-end.
- **Designed but not implemented:** a scheduled monthly job, a FastAPI/dashboard serving layer, and automated drift monitoring against newly-published IPC cycles. These are shown as an architecture diagram in the notebook's Phase 7 section and depend on infrastructure (hosting, a scheduler, a live NDMA data feed) outside the scope of this notebook. The scoring function and saved artifacts are written to be the building blocks those stages would call directly.
- **Interpretability caveat:** SHAP analysis shows the model relies most heavily on *where a county is* and *what it already was* (`is_asal`, `prev_crisis`) rather than on what's currently changing. It is strongest at confirming already-vulnerable, already-crisis-prone counties, and comparatively weaker at catching a new crisis emerging somewhere previously stable — an important limitation for an early-warning system to be transparent about, and a clear direction for future feature work (price *trends* rather than levels, longer rolling rainfall windows).

## Setup

### Requirements

- Python 3.9+
- Jupyter (VS Code Jupyter extension or JupyterLab)

### Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm shap joblib openpyxl
```

### Running the notebook

1. Place the three raw data files in a `data/` folder at the project root (see structure above).
2. Open `notebook.ipynb` in VS Code (with the Jupyter extension) or JupyterLab.
3. Run cells top to bottom — the notebook is written to execute sequentially through all 7 phases.
4. Generated artifacts (`data/model_ready.csv`, `models/deployed_model_*.joblib`, `models/deployed_imputer.joblib`, `models/feature_columns.pkl`, `models/deployment_metadata.json`) will be written automatically.

## Data Access

The raw datasets are publicly available from the Humanitarian Data Exchange (HDX):
- FEWS NET IPC classifications
- CHIRPS rainfall indicators (Kenya subnational)
- WFP food prices — Kenya

They are not bundled in this repository due to size; download them from HDX and place them under `data/` using the filenames listed above.

## Success Criteria

| Metric | Target | Achieved | Met? |
|---|---|---|---|
| Recall (crisis class) | ≥ 0.75 | 0.804 | ✅ |
| Precision (crisis class) | ≥ 0.60 | 0.752 | ✅ |
| PR-AUC (crisis class) | ≥ 0.75 | 0.818 | ✅ |
| F1 (macro) | ≥ 0.70 | 0.858 | ✅ |
| ROC-AUC | ≥ 0.80 | 0.944 | ✅ |
| Beat naive baseline | PR-AUC > 0.197 (test prevalence) | 0.818 | ✅ |

## Citations

1. Funk, C. et al. (2015). *The climate hazards infrared precipitation with stations.* Scientific Data, 2, 150066.
2. FAO (2021). *IPC Technical Manual Version 3.1.*
3. FEWS NET (2023). *Kenya Food Security Outlook, Oct 2022–Mar 2023.*

## License

This project is submitted as a Moringa School capstone deliverable. No license is currently specified for reuse; contact the authors before redistributing.

## Authors

Data Alchemists: Patrick Njoroge, Chelimo Ivy, Alex Njihia, Joel Andrew, Halima Ahmed, Rebecca Monchari
