"""
model_utils.py
---------------
Shared utilities for loading the trained churn model and generating
predictions + explanations. Used by both the Streamlit app and the
model-development notebook so behavior never drifts between the two.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd

from data_processing import MODEL_FEATURES, single_customer_to_row

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_best_model():
    return joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))


def load_all_models():
    return joblib.load(os.path.join(MODELS_DIR, "all_models.pkl"))


def load_metrics():
    with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
        return json.load(f)


def load_feature_importance():
    with open(os.path.join(MODELS_DIR, "feature_importance.json")) as f:
        return json.load(f)


def load_roc_data():
    with open(os.path.join(MODELS_DIR, "roc_data.json")) as f:
        return json.load(f)


def load_test_predictions():
    return pd.read_csv(os.path.join(MODELS_DIR, "test_predictions.csv"))


def predict_proba_for_customer(model, customer: dict) -> float:
    """Predict churn probability (0-1) for a single raw customer dict."""
    row = single_customer_to_row(customer)[MODEL_FEATURES]
    return float(model.predict_proba(row)[0, 1])


def risk_band(probability: float) -> str:
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.6:
        return "Medium Risk"
    else:
        return "High Risk"


def local_sensitivity(model, customer: dict, feature: str, values) -> pd.DataFrame:
    """
    Simple, model-agnostic local explanation: hold every other feature at
    the customer's current value and sweep `feature` across `values`,
    recording how the predicted probability responds. This is the
    "individual conditional expectation" curve for one customer and powers
    the What-If Scenario Simulator.
    """
    rows = []
    for v in values:
        c = dict(customer)
        c[feature] = v
        prob = predict_proba_for_customer(model, c)
        rows.append({feature: v, "ChurnProbability": prob})
    return pd.DataFrame(rows)


def local_feature_contributions(model, customer: dict, baseline: dict, features=None) -> pd.DataFrame:
    """
    Poor-man's Shapley-style local attribution: for each feature, measure
    how much the predicted probability changes when that single feature is
    reset from the customer's value back to a population baseline (e.g.
    dataset mean/mode), holding everything else fixed. This approximates
    each feature's marginal contribution to THIS customer's score without
    requiring the optional `shap` dependency. If `shap` is installed, the
    training notebook additionally produces true SHAP values for global
    validation.
    """
    if features is None:
        features = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                    "HasCrCard", "IsActiveMember", "EstimatedSalary", "Geography", "Gender"]

    base_prob = predict_proba_for_customer(model, customer)
    rows = []
    for feat in features:
        perturbed = dict(customer)
        perturbed[feat] = baseline.get(feat, customer[feat])
        prob_without = predict_proba_for_customer(model, perturbed)
        rows.append({"Feature": feat, "Contribution": base_prob - prob_without})

    result = pd.DataFrame(rows).sort_values("Contribution", key=abs, ascending=False)
    return result
