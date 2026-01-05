# Terry Traffic Stops — Data Analysis

This repository contains a Jupyter notebook analysis of Terry stop records and example model pipelines to predict arrest outcomes. The notebook performs data cleaning, exploratory analysis, model training, and evaluation with a focus on understanding predictors and potential disparities.

## Project Overview

This project analyzes **Terry Stop** records from Seattle Police Department (SPD), spanning 2015-2025. Terry Stops are brief investigative encounters where police temporarily detain individuals based on reasonable suspicion of criminal activity. This analysis uses machine learning and statistical methods to:

1. **Understand patterns** in arrest outcomes across different demographics and contexts
2. **Build predictive models** to identify factors most strongly associated with arrests
3. **Detect potential disparities** in how policing decisions vary by race, age, and other characteristics
4. **Inform policy discussions** on equity and accountability in law enforcement

### Dataset Characteristics
- **Total Records**: 65,884 police stops
- **Time Period**: 2015-2025
- **Target Variable**: Arrest Flag [Binary: Yes/No]
- **Class Distribution**: 88.5% non-arrests, 11.5% arrests 
- **Features**: Subject demographics, officer characteristics, weapon presence, call types, frisk decisions

---

## Objectives

**1. Predictive Modeling**
- Train multiple machine learning algorithms to predict arrest outcomes
- Evaluate models using appropriate metrics for imbalanced classification
- Identify the best-performing model for production use

**2. Feature Importance Analysis**
- Determine which factors most strongly influence arrest decisions
- Distinguish between objective factors (weapons, call type) and subjective/demographic factors (race, age)
- Understand if officer characteristics affect arrest propensity

**3. Statistical Investigation**
- Conduct hypothesis testing on racial and demographic disparities
- Quantify the relationship between frisk decisions and arrests
- Compare arrest rates across meaningful demographic groups

**4. Model Interpretability**
- Visualize model performance across multiple dimensions
- Understand prediction confidence and calibration
- Identify edge cases where models struggle

**5. Data Quality Assessment**
- Clean and standardize messy police data
- Handle missing values appropriately
- Engineer meaningful features from raw variables

---

## Analysis Workflow

```
1. DATA LOADING & CLEANING
	↓
2. EXPLORATORY DATA ANALYSIS (EDA)
	├─ Distribution of arrests
	├─ Racial/demographic disparities
	├─ Correlation analysis
	└─ Statistical hypothesis testing
	↓
3. FEATURE ENGINEERING
	├─ One-hot encoding of categorical variables
	├─ Feature scaling and preprocessing
	└─ Pipeline creation for reproducibility
	↓
4. MACHINE LEARNING MODELING
	├─ Train 7 different algorithms:
	│  ├─ Logistic Regression
	│  ├─ Random Forest
	│  ├─ Gradient Boosting
	│  ├─ Decision Tree
	│  ├─ Naive Bayes
	│  ├─ AdaBoost
	│  └─ Extra Trees
	↓
5. DETAILED PERFORMANCE ANALYSIS
	├─ ROC/PR curves and AUC scores
	├─ Confusion matrices
	├─ Feature importance rankings
	└─ Probability calibration
	↓
6. DEEPER EXPLORATIONS
	├─ Age group analysis
	├─ Weapon type effects
	├─ Call type patterns
	├─ Officer demographics
	└─ Frisk-arrest relationship
```

---

## Key Questions Being Answered

1. **Can we predict arrests?** How well do demographics and context predict arrest outcomes?
2. **What matters most?** Which factors have the strongest influence on arrest decisions?
3. **Are there disparities?** Do arrest rates differ significantly by race, age, or other protected characteristics?
4. **Is frisk a predictor or a cause?** Does frisk strongly correlate with arrests, and what does it indicate?
5. **Which model works best?** For different use cases (maximizing accuracy vs. catching arrests), which algorithm performs best?

---

## Important Caveats

 **Correlation ≠ Causation**: This analysis identifies patterns, not causal relationships. Observed disparities could result from:
- Different stop rates across neighborhoods
- Differences in actual crime patterns
- Officer assignment patterns
- Legitimate differences in case characteristics

 **Data Limitations**: Police data reflects enforcement decisions, not actual crime. Underrepresentation or overrepresentation in stops may reflect policing patterns rather than actual criminality.

 **Ethical Considerations**: Predictive models of arrests perpetuate historical patterns in the training data. A model that accurately predicts past arrests may simply be learning past biases.


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