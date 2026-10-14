# ML Algorithm Comparison: Classification, Boosting, Clustering and Regression

This repository demonstrates machine-learning workflows using three supplied datasets.

## Datasets
- `Social_Network_Ads.csv`: classification (`Purchased`) and K-Means clustering (`Age`, `EstimatedSalary`).
- `insurance_pre.csv`: regression to estimate `charges`.
- `Salary_Data.csv`: regression to estimate `Salary` from `YearsExperience`.

## Algorithms
- Classification: Logistic Regression, Decision Tree, Random Forest.
- Boosting: AdaBoost, Gradient Boosting.
- Clustering: K-Means (2 clusters).
- Regression: Linear Regression, Random Forest Regressor, Gradient Boosting Regressor.

## Evaluation
- Classification: accuracy, precision, recall, F1-score.
- Clustering: silhouette score and inertia.
- Regression: MAE, RMSE, R².

## Run locally
Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
python src/train_models.py
```

The script saves comparison metrics to `results/model_comparison.csv`, clustering assignments to
`results/customer_clusters.csv`, and the classification accuracy chart to
`results/classification_accuracy.png`.

## Important notes
- The test split is 20%, with `random_state=42` for repeatability.
- `User ID` is excluded from prediction because it is an identifier, not a meaningful predictive feature.
- Clustering does not use the `Purchased` label; it groups records using standardized `Age` and `EstimatedSalary`.
- Metrics depend on the split and dataset. Do not describe them as guaranteed performance on new data.
