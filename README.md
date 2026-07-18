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

Expected local files (not included in this repo — see [Data Access](#data-access)):


## Repository Structure

```
.
├── data/
│   ├── ipcphase.xlsx                       # IPC/FEWS NET phase classifications
│   ├── ken-rainfall-subnat-full.csv        # CHIRPS dekadal rainfall anomalies
│   └── wfp_food_prices_ken.csv             # WFP monthly staple food prices
├── catboost_info/                          # Auto-generated CatBoost training logs
│   ├── learn/
│   ├── catboost_training.json
│   ├── learn_error.tsv
│   └── time_left.tsv
├── notebook.ipynb                          # Main analysis notebook 
├── model_df.csv                            # Intermediate model-ready dataset 
├── model_ready.csv                         # Final cleaned modelling dataset 
├── deployed_model_Random_Forest.joblib     # Serialized winning classifier 
├── deployed_imputer.joblib                 # Fitted SimpleImputer 
├── deployment_metadata.json                # Threshold, feature list, recall floor 
├── Data_Alchemists_Capstone_Proposal.docx  # Original project proposal
└── README.md
```

> `catboost_info/` is created automatically the first time CatBoost trains in the notebook, it's safe to delete and regenerate, and worth adding to `.gitignore`.

## Methodology (CRISP-DM)

| Phase | Description |
|---|---|
| 1. Business Understanding | Problem framing, stakeholders, success criteria |
| 2. Data Understanding | Structure, dtypes, missingness for all three raw datasets |
| 3. Data Preparation | Cleaning, geographic harmonization to 47 counties, target construction, lag alignment, merge |
| 4. EDA | Univariate, bivariate, and multivariate analysis of climate/price signals vs. crisis |
| 5. Modeling | Chronological train/validation/test split, 4 candidate models, threshold tuning |
| 6. Evaluation | Test-set metrics, calibration, confidence intervals, SHAP interpretability |
| 7. Deployment | Monthly scoring pipeline design and live demonstration on held-out data |

Key technical decisions:
- **Geographic harmonization:** IPC (compound location strings), CHIRPS (PCODE), and WFP (admin names) were all mapped to Kenya's 47 official counties, including historical alias handling for post-devolution name changes.
- **One-month lag** applied to rainfall/price features relative to IPC reporting date, to prevent leakage and simulate a genuine early-warning scenario.
- **Chronological (not random) split:** train on pre-2021 data, validate on 2021, test on 2022+ - touched exactly once at the end.
- **Threshold selection rule:** sweep thresholds on validation data, keep those with recall ≥ 0.75 (humanitarian floor), and pick the one that maximizes precision.
- **Four deployment-ready candidates** compared under identical rules: Logistic Regression (class-weighted), Random Forest (class-weighted), XGBoost (`scale_pos_weight`), and LightGBM (`is_unbalance`).

## Key Results

- **Winning model:** Random Forest - recall 0.804, precision 0.745, F1-macro 0.858, PR-AUC 0.815 on the validation-selected threshold; roughly 3 crises caught for every 1 false alarm.
- **Top predictors (SHAP):** structural vulnerability (`is_asal`), 3-month rainfall anomaly (`r3q_mean`), 1-month rainfall anomaly (`rfq_mean`), and recent crisis history (`prev_crisis`) dominate; staple food price level is a weaker, supporting signal.
- All four candidates reached recall in the 0.84–0.90 range on the genuinely unseen 2022+ test set, indicating the feature set (not model architecture) carries most of the predictive signal.

## Setup

### Requirements

- Python 3.9+
- Jupyter (VS Code Jupyter extension or JupyterLab)

### Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost shap joblib imbalanced-learn
```

> The notebook installs `imbalanced-learn` itself in a setup cell if it isn't already present.

### Running the notebook

1. Place the three raw data files in a `data/` folder at the project root (see structure above).
2. Open `notebook.ipynb` in VS Code (with the Jupyter extension) or JupyterLab.
3. Run cells top to bottom — the notebook is written to execute sequentially through all 7 phases.
4. Generated artifacts (`model_df.csv`, `deployed_model_*.joblib`, `deployed_imputer.joblib`, `deployment_metadata.json`) will be written to the project root.

## Data Access

The raw datasets are publicly available from the Humanitarian Data Exchange (HDX):
- FEWS NET IPC classifications
- CHIRPS rainfall indicators (Kenya subnational)
- WFP food prices — Kenya

They are not bundled in this repository due to size; download them from HDX and place them under `data/` using the filenames listed above.

## Success Criteria

| Metric | Target | Rationale |
|---|---|---|
| Recall (crisis class) | ≥ 0.75 | Missing a crisis has far greater humanitarian cost than a false alarm |
| Precision (crisis class) | ≥ 0.60 | Prevents alert fatigue |
| PR-AUC (crisis class) | ≥ 0.75 | Imbalance-robust discriminative measure |
| F1 (macro) | ≥ 0.70 | Penalizes ignoring the minority crisis class |
| ROC-AUC | ≥ 0.80 | Overall discriminative ability |

## Citations

1. Funk, C. et al. (2015). *The climate hazards infrared precipitation with stations.* Scientific Data, 2, 150066.
2. FAO (2021). *IPC Technical Manual Version 3.1.*
3. FEWS NET (2023). *Kenya Food Security Outlook, Oct 2022–Mar 2023.*

## Authors

Data Alchemists : Patrick Njoroge, Chelimo Ivy, Alex Njihia, Joel Andrew, Halima Ahmed, Rebecca Monchari
