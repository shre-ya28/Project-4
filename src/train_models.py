"""
train_models.py
----------------
Trains and evaluates the full model suite requested in the project brief:
  - Logistic Regression (interpretability baseline)
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - XGBoost (optional -- used automatically if the `xgboost` package is
    installed, skipped gracefully otherwise)

For each model it computes Accuracy, Precision, Recall, F1-score and
ROC-AUC on a held-out stratified test set, selects the best model by
ROC-AUC, and saves all artifacts needed by the Streamlit app and notebook:

  models/best_model.pkl        -- the selected, fitted sklearn pipeline
  models/all_models.pkl        -- dict of every fitted pipeline (for comparison)
  models/metrics.json          -- evaluation metrics for every model
  models/feature_importance.json -- global feature importance for the best model
  models/roc_data.json         -- fpr/tpr points for each model's ROC curve
  models/test_predictions.csv  -- test-set probabilities + actuals (for the
                                   probability-distribution view in the app)

Run with:
    python src/train_models.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from data_processing import MODEL_FEATURES, build_model_table

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

RANDOM_STATE = 42
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "European_Bank.csv")


def get_model_suite():
    """Models that need scaled inputs are wrapped with a StandardScaler."""
    suite = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": Pipeline([
            ("clf", DecisionTreeClassifier(max_depth=6, min_samples_leaf=30,
                                            class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10,
                                            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("clf", GradientBoostingClassifier(n_estimators=250, max_depth=3, learning_rate=0.05,
                                                random_state=RANDOM_STATE)),
        ]),
    }
    if HAS_XGBOOST:
        suite["XGBoost"] = Pipeline([
            ("clf", XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                random_state=RANDOM_STATE,
            )),
        ])
    return suite


def evaluate(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "Accuracy": round(accuracy_score(y_test, pred), 4),
        "Precision": round(precision_score(y_test, pred), 4),
        "Recall": round(recall_score(y_test, pred), 4),
        "F1": round(f1_score(y_test, pred), 4),
        "ROC_AUC": round(roc_auc_score(y_test, proba), 4),
    }, proba, pred


def get_feature_importance(name, pipeline, X_test, y_test):
    """Native importance for tree models; |coefficient| for logistic
    regression; permutation importance as a universal fallback."""
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        result = permutation_importance(pipeline, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE)
        importances = result.importances_mean

    importances = np.array(importances, dtype=float)
    importances = importances / importances.sum()
    return dict(sorted(zip(MODEL_FEATURES, importances.tolist()), key=lambda x: x[1], reverse=True))


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    table = build_model_table(DATA_PATH)
    X = table[MODEL_FEATURES]
    y = table["Exited"]

    # Stratified train-test split preserving churn class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    suite = get_model_suite()
    metrics_all = {}
    roc_data = {}
    fitted_models = {}
    cv_scores = {}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, pipeline in suite.items():
        print(f"Training {name} ...")
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline

        metrics, proba, pred = evaluate(pipeline, X_test, y_test)
        metrics_all[name] = metrics

        cv = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="roc_auc")
        cv_scores[name] = {"mean_roc_auc": round(cv.mean(), 4), "std_roc_auc": round(cv.std(), 4)}

        fpr, tpr, _ = roc_curve(y_test, proba)
        # Downsample ROC points for compact JSON storage
        idx = np.linspace(0, len(fpr) - 1, min(100, len(fpr))).astype(int)
        roc_data[name] = {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist(), "auc": metrics["ROC_AUC"]}

        cm = confusion_matrix(y_test, pred).tolist()
        metrics_all[name]["ConfusionMatrix"] = cm

        print(f"  {name}: {metrics}")

    # Select best model by ROC-AUC
    best_name = max(metrics_all, key=lambda n: metrics_all[n]["ROC_AUC"])
    best_model = fitted_models[best_name]
    print(f"\nBest model by ROC-AUC: {best_name}")

    # Feature importance for the best model
    feat_importance = get_feature_importance(best_name, best_model, X_test, y_test)

    # Save artifacts
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(fitted_models, os.path.join(MODELS_DIR, "all_models.pkl"))

    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump({
            "best_model": best_name,
            "has_xgboost": HAS_XGBOOST,
            "metrics": metrics_all,
            "cv_scores": cv_scores,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "features": MODEL_FEATURES,
        }, f, indent=2)

    with open(os.path.join(MODELS_DIR, "feature_importance.json"), "w") as f:
        json.dump(feat_importance, f, indent=2)

    with open(os.path.join(MODELS_DIR, "roc_data.json"), "w") as f:
        json.dump(roc_data, f, indent=2)

    # Save test predictions for the probability-distribution dashboard view
    best_proba = best_model.predict_proba(X_test)[:, 1]
    test_predictions = pd.DataFrame({
        "ChurnProbability": best_proba,
        "ActualChurn": y_test.values,
    })
    test_predictions.to_csv(os.path.join(MODELS_DIR, "test_predictions.csv"), index=False)

    print("\nAll artifacts saved to models/")


if __name__ == "__main__":
    main()
