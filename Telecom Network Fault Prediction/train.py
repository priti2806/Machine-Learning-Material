import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, precision_recall_curve
)


def get_model_scores(model, features):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    return model.decision_function(features)

# 1. Load dataset from CSV
df = pd.read_csv("network_data.csv")

# 2. Feature engineering
df["congestion_score"] = (
    df["cpu_utilization"]
    * df["bandwidth_utilization"] / 100
)

df["high_latency"] = (df["latency_ms"] > 70).astype(int)
df["high_packet_loss"] = (df["packet_loss"] > 1.5).astype(int)
df["high_cpu_utilization"] = (df["cpu_utilization"] > 85).astype(int)
df["high_bandwidth_utilization"] = (df["bandwidth_utilization"] > 90).astype(int)
df["weak_signal"] = (df["signal_strength"] < -100).astype(int)

df["poor_network_quality"] = (
    (df["latency_ms"] > 70)
    & (df["packet_loss"] > 1.5)
).astype(int)

# 3. Split features and target
X = df.drop(columns=["network_fault"])
y = df["network_fault"]

# 4. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    random_state=43,
    stratify=y_train
)

# 5. Define models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42
    ),

}

# 6. Train baseline models
for name, model in models.items():
    model.fit(X_fit, y_fit)
    print(name, "trained successfully")


# 7. Faster cross-validation and hyperparameter tuning for SVM

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(
        random_state=42,
        cache_size=1000
    ))
])

param_grid = [
    {
        "model__C": [0.1, 1],
        "model__kernel": ["linear"],
        "model__class_weight": [None, "balanced"]
    }
]

grid = GridSearchCV(
    estimator=svm_pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    verbose=2,
    pre_dispatch="2*n_jobs",
    refit=True
)

grid.fit(X_fit, y_fit)

best_svm = grid.best_estimator_
models["SVM"] = best_svm

decision_thresholds = {}
for name, model in models.items():
    scores = get_model_scores(model, X_val)
    precision, recall, thresholds = precision_recall_curve(y_val, scores)
    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )
    decision_thresholds[name] = float(thresholds[f1_scores.argmax()])

for model in models.values():
    model.fit(X_train, y_train)

print("\nBest SVM parameters:")
print(grid.best_params_)

print("Best cross-validation F1:")
print(round(grid.best_score_, 3))


# 8. Evaluate tuned SVM on the untouched test set
y_score = best_svm.decision_function(X_test)
y_pred = (y_score >= decision_thresholds["SVM"]).astype(int)

print("\nTuned SVM Test Results")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("Precision:", round(
    precision_score(y_test, y_pred, zero_division=0), 3
))
print("Recall:", round(
    recall_score(y_test, y_pred, zero_division=0), 3
))
print("F1:", round(f1_score(y_test, y_pred, zero_division=0), 3))
print("ROC-AUC:", round(roc_auc_score(y_test, y_score), 3))
print("SVM F1 threshold:", round(decision_thresholds["SVM"], 3))


# 9. Save models, thresholds, and test data
joblib.dump(models, "trained_models.pkl")
joblib.dump(decision_thresholds, "decision_thresholds.pkl")
joblib.dump((X_test, y_test), "test_data.pkl")

print("\nTuned models and test data saved successfully!")