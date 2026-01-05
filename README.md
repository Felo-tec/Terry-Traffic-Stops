# Terry Traffic Stops — Data Analysis

This repository contains a Jupyter notebook analysis of Terry stop records and example model pipelines to predict arrest outcomes. The notebook performs data cleaning, exploratory analysis, model training, and evaluation with a focus on understanding predictors and potential disparities.

## Project Overview

- Dataset: Seattle Terry stop records (2015–2025 snapshot), ~65.9k records.
- Goal: Explore patterns in stops and build classifiers to predict `Arrest Flag` while highlighting limitations and fairness considerations.
- Output: Visualizations, model comparisons (ROC/PR, confusion matrices), and feature importance analyses.

## Objectives

- Clean and standardize raw Terry stop data (dates, categorical normalization, missing values).
- Conduct EDA to surface patterns by race, age group, call type, weapon presence, frisk, and officer attributes.
- Train multiple classifiers and compare performance, emphasizing how class imbalance affects results.
- Report key findings, limitations, and recommendations for further analysis or responsible deployment.

## Key Findings

- Class imbalance: Arrests are the minority (~11–12%), so accuracy is a misleading metric — use ROC/PR and threshold tuning.
- Strong predictors: Officer-related features (age / YOB) and operational context (`Initial Call Type`, `Weapon Type`) consistently show high importance in tree-based models.
- Frisk correlation: Frisked subjects show ~2x higher arrest rates compared to non-frisked subjects — indicative of a strong association that requires causal analysis.
- Call-type effects: Calls like shoplifting and assault have substantially higher arrest rates, making call context a legitimate and powerful predictor.
- Model performance: Gradient Boosting and Logistic Regression show the best ROC AUC (~0.80–0.83). Random Forest is competitive but shows the typical precision/recall trade-offs on the minority class.

## Requirements

- Python 3.8+
- Install the core dependencies into a virtual environment:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy ipython jupyter
```

Optional for headless execution:

```bash
pip install nbconvert
```

## Quick Start

1. Place `Terry_Stops_20251225.csv` in the repository root (next to `index.ipynb`).
2. Open the notebook in VS Code or Jupyter:

```bash
jupyter notebook index.ipynb
```

3. (Optional) Execute headlessly and save outputs:

```bash
jupyter nbconvert --to notebook --execute index.ipynb --output index_executed.ipynb
```

## What the Notebook Does

- Loads and cleans the Terry stop dataset.
- Explores arrest distributions and relationships with key features.
- Trains: Logistic Regression, Random Forest, Gradient Boosting, Decision Tree, Naive Bayes, AdaBoost, Extra Trees.
- Compares models (ROC AUC, PR curves, confusion matrices) and inspects feature importances.

## Notes & Caveats

- Imbalanced target: Use PR curves and tuned thresholds; consider class weighting or resampling for training.
- Observational limits: Correlation ≠ causation. Use causal methods before policy claims.
- Data quality: Inspect missingness and recording biases; replicate preprocessing in production.

## Next Steps

- Calibrate, tune thresholds, and run fairness audits.
- Package preprocessing and model pipeline for reproducible training/deployment.
- Add unit tests and a `requirements.txt` for environment reproducibility.


## Conclusion

- The notebook demonstrates that arrest outcomes can be predicted with reasonable discrimination, but operational use requires careful calibration, fairness auditing, and causal validation.
- Recommendations:
	- Calibrate probabilities and choose thresholds aligned with operational tolerances (precision vs recall).
	- Run fairness audits and consider reweighing or constrained optimization if disparities persist.
	- Perform causal inference (propensity scores, instrumental variables) before asserting policy recommendations.
	- Extract preprocessing and model pipelines for reproducible evaluation and monitoring.

## Contact

felix.kipkurui@student.moringaschool.com

---
Generated: January 2026
# Terry-Traffic-Stops