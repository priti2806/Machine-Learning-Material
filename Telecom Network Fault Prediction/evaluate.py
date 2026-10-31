import pandas as pd
import joblib
import os

from sklearn.metrics import (
precision_score,
recall_score,
f1_score,
roc_auc_score,
accuracy_score,
)

# Load trained models and test data

models = joblib.load("trained_models.pkl")
X_test, y_test = joblib.load("test_data.pkl")
threshold_path = "decision_thresholds.pkl"
decision_thresholds = (
    joblib.load(threshold_path) if os.path.exists(threshold_path) else {}
)

# Evaluate models

results = []

for name, model in models.items():
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.decision_function(X_test)
    if name in decision_thresholds:
        predictions = (scores >= decision_thresholds[name]).astype(int)
    else:
        predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, scores)
    })


# Display results

results_df = pd.DataFrame(results)
print(results_df.round(3).to_string(index=False))

# Save evaluation results

results_df.to_csv("model_evaluation_results.csv", index=False)

print("\nResults saved to model_evaluation_results.csv")
