# Terry Traffic Stops — Data Analysis

This repository contains a Jupyter notebook that analyzes Terry stop (police stop) records and trains models to predict arrest outcomes.

## Contents

- `index.ipynb` — Main analysis notebook (data loading, cleaning, EDA, modeling, and visualizations).
- `Terry_Stops_20251225.csv` — Original dataset (not included in the repo by default; expected in repository root).

## Requirements

- Python 3.8+
- Recommended packages (install into a virtualenv):

```
pip install pandas numpy matplotlib seaborn scikit-learn scipy ipython jupyter
```

Optional for headless notebook execution:

```
pip install nbconvert
```

## Quick Start

1. Place the dataset file `Terry_Stops_20251225.csv` in the repository root (where `index.ipynb` lives).
2. Open the notebook in VS Code or Jupyter:

   - VS Code: Open `index.ipynb` in the editor and use the interactive cell toolbar to run cells.
   - Jupyter: Start a server and open the notebook:

```bash
jupyter notebook index.ipynb
```

3. (Optional) Execute the entire notebook headlessly and save outputs:

```bash
jupyter nbconvert --to notebook --execute index.ipynb --output index_executed.ipynb
```

## What the Notebook Does

- Loads and cleans the Terry stop dataset.
- Explores arrest distributions and relationships with features such as race, age groups, officer attributes, call types, weapon presence, and frisk flags.
- Trains several classifiers (Logistic Regression, Random Forest, Gradient Boosting, Decision Tree, Naive Bayes, AdaBoost, Extra Trees).
- Compares model performance (ROC AUC, precision/recall, confusion matrices) and inspects feature importances for tree-based models.

## Notes & Caveats

- The `Arrest Flag` is highly imbalanced (~11–12% arrests). Model evaluation and threshold selection should account for this.
- Correlation in observational data does not imply causation — treat fairness and policy conclusions cautiously.
- The notebook contains preprocessing steps (imputation, encoding, scaling) that should be reproduced exactly when packaging a model for production.

## Next Steps

- Calibrate model probabilities and tune thresholds for operational trade-offs.
- Add fairness audits and causal analyses before drawing policy conclusions.
- Extract preprocessing and model pipeline into standalone scripts for reproducible training and deployment.

## Contact

If you want changes or additions to the notebook (additional visualizations, model tuning, fairness checks), open an issue or message me.

---
Generated: January 2026
# Terry-Traffic-Stops