"""
Streamlit Web Application
Predictive Modeling and Risk Scoring for Bank Customer Churn

Run with:
    streamlit run app/streamlit_app.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_processing import build_model_table, MODEL_FEATURES
from model_utils import (
    load_best_model,
    load_metrics,
    load_feature_importance,
    load_roc_data,
    load_test_predictions,
    predict_proba_for_customer,
    risk_band,
    local_sensitivity,
    local_feature_contributions,
)

st.set_page_config(page_title="European Bank | Churn Risk Scoring", page_icon="🎯", layout="wide")

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "European_Bank.csv")


@st.cache_resource
def get_model():
    return load_best_model()


@st.cache_data
def get_metrics():
    return load_metrics()


@st.cache_data
def get_feature_importance():
    return load_feature_importance()


@st.cache_data
def get_roc_data():
    return load_roc_data()


@st.cache_data
def get_test_predictions():
    return load_test_predictions()


@st.cache_data
def get_raw_data():
    return pd.read_csv(DATA_PATH)


model = get_model()
metrics = get_metrics()
feat_importance = get_feature_importance()
roc_data = get_roc_data()
test_preds = get_test_predictions()
raw_df = get_raw_data()

best_name = metrics["best_model"]

RISK_COLORS = {"Low Risk": "#4C9A2A", "Medium Risk": "#E8A33D", "High Risk": "#C0392B"}

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🎯 Predictive Churn Risk Scoring")
st.caption("European Central Bank — Customer Churn Intelligence System")

with st.expander("ℹ️ About this model", expanded=False):
    c1, c2, c3, c4, c5 = st.columns(5)
    m = metrics["metrics"][best_name]
    c1.metric("Model", best_name)
    c2.metric("ROC-AUC", m["ROC_AUC"])
    c3.metric("Accuracy", m["Accuracy"])
    c4.metric("Precision", m["Precision"])
    c5.metric("Recall", m["Recall"])
    st.caption(
        f"Trained on {metrics['n_train']:,} customers, evaluated on {metrics['n_test']:,} held-out "
        f"customers via a stratified 80/20 split. Predictions below use the {best_name} model, "
        f"selected as the best performer by ROC-AUC across all candidate models."
    )

tab1, tab2, tab3, tab4 = st.tabs(
    ["🧮 Risk Calculator", "📈 Probability Distribution", "🔍 Feature Importance", "🧪 What-If Simulator"]
)

# --------------------------------------------------------------------------
# TAB 1 — Customer churn risk calculator
# --------------------------------------------------------------------------
with tab1:
    st.subheader("Customer Churn Risk Calculator")
    st.caption("Enter a customer's profile to generate a live churn probability and risk score.")

    col1, col2, col3 = st.columns(3)
    with col1:
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"], key="calc_geo")
        gender = st.selectbox("Gender", ["Female", "Male"], key="calc_gender")
        age = st.slider("Age", 18, 92, 40, key="calc_age")
    with col2:
        credit_score = st.slider("Credit Score", 350, 850, 650, key="calc_cs")
        tenure = st.slider("Tenure (years with bank)", 0, 10, 5, key="calc_tenure")
        num_products = st.slider("Number of Products", 1, 4, 1, key="calc_prod")
    with col3:
        balance = st.number_input("Account Balance (€)", 0.0, 300000.0, 75000.0, step=1000.0, key="calc_bal")
        salary = st.number_input("Estimated Annual Salary (€)", 0.0, 300000.0, 100000.0, step=1000.0, key="calc_sal")
        has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"], key="calc_cc")
        is_active = st.selectbox("Active Member", ["Yes", "No"], key="calc_active")

    customer = {
        "CreditScore": credit_score, "Geography": geography, "Gender": gender, "Age": age,
        "Tenure": tenure, "Balance": balance, "NumOfProducts": num_products,
        "HasCrCard": 1 if has_cr_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active == "Yes" else 0,
        "EstimatedSalary": salary,
    }

    threshold = st.slider(
        "Classification threshold (probability at/above this = flagged as 'At Risk')",
        0.05, 0.95, 0.5, 0.05, key="calc_threshold",
    )

    if st.button("Calculate Churn Risk", type="primary"):
        prob = predict_proba_for_customer(model, customer)
        band = risk_band(prob)
        flagged = prob >= threshold

        c1, c2, c3 = st.columns(3)
        c1.metric("Churn Probability", f"{prob:.1%}")
        c2.metric("Risk Band", band)
        c3.metric("Flag @ Threshold", "⚠️ At Risk" if flagged else "✅ Retained")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": RISK_COLORS[band]},
                "steps": [
                    {"range": [0, 30], "color": "#E7F3E1"},
                    {"range": [30, 60], "color": "#FCEFDA"},
                    {"range": [60, 100], "color": "#F9DEDB"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "value": threshold * 100},
            },
            title={"text": "Churn Risk Score"},
        ))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Why this score? (local feature contribution)")
        baseline_customer = {
            "CreditScore": raw_df["CreditScore"].mean(), "Age": raw_df["Age"].mean(),
            "Tenure": raw_df["Tenure"].mean(), "Balance": raw_df["Balance"].mean(),
            "NumOfProducts": 1, "HasCrCard": 1, "IsActiveMember": 1,
            "EstimatedSalary": raw_df["EstimatedSalary"].mean(), "Geography": "France", "Gender": "Female",
        }
        contrib = local_feature_contributions(model, customer, baseline_customer)
        fig2 = px.bar(
            contrib, x="Contribution", y="Feature", orientation="h",
            color=contrib["Contribution"] > 0,
            color_discrete_map={True: "#C0392B", False: "#4C9A2A"},
            title="Feature contribution vs. an average, engaged customer",
        )
        fig2.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            "Red bars push this customer's risk **above** an average, engaged baseline customer; "
            "green bars pull it down. This is a local, model-agnostic approximation (not true SHAP)."
        )

# --------------------------------------------------------------------------
# TAB 2 — Probability distribution visualization
# --------------------------------------------------------------------------
with tab2:
    st.subheader("Predicted Churn Probability Distribution")
    st.caption(f"Distribution of {best_name} churn probabilities across the {len(test_preds):,} held-out test customers.")

    dist_threshold = st.slider("Threshold", 0.05, 0.95, 0.5, 0.05, key="dist_threshold")

    test_preds_display = test_preds.copy()
    test_preds_display["Actual"] = test_preds_display["ActualChurn"].map({0: "Retained", 1: "Churned"})
    test_preds_display["Predicted"] = np.where(test_preds_display["ChurnProbability"] >= dist_threshold, "Flagged At-Risk", "Flagged Retained")

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.histogram(
            test_preds_display, x="ChurnProbability", color="Actual", nbins=40, barmode="overlay",
            opacity=0.65, color_discrete_map={"Retained": "#4C72B0", "Churned": "#DD8452"},
            title="Churn Probability Distribution — Test Set",
        )
        fig.add_vline(x=dist_threshold, line_dash="dash", line_color="black",
                       annotation_text=f"Threshold = {dist_threshold}")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        tp = ((test_preds_display["ActualChurn"] == 1) & (test_preds_display["Predicted"] == "Flagged At-Risk")).sum()
        fp = ((test_preds_display["ActualChurn"] == 0) & (test_preds_display["Predicted"] == "Flagged At-Risk")).sum()
        fn = ((test_preds_display["ActualChurn"] == 1) & (test_preds_display["Predicted"] == "Flagged Retained")).sum()
        tn = ((test_preds_display["ActualChurn"] == 0) & (test_preds_display["Predicted"] == "Flagged Retained")).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        st.metric("True Positives (caught churners)", tp)
        st.metric("False Positives (false alarms)", fp)
        st.metric("False Negatives (missed churners)", fn)
        st.metric("Precision @ threshold", f"{precision:.1%}")
        st.metric("Recall @ threshold", f"{recall:.1%}")
        st.metric("F1 @ threshold", f"{f1:.1%}")

    st.markdown("#### ROC Curves — All Trained Models")
    fig = go.Figure()
    for name, d in roc_data.items():
        fig.add_trace(go.Scatter(x=d["fpr"], y=d["tpr"], mode="lines", name=f"{name} (AUC={d['auc']:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="grey"), name="Random"))
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### All-Model Metric Comparison")
    metrics_df = pd.DataFrame(metrics["metrics"]).T[["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]].astype(float)
    st.dataframe(metrics_df.style.highlight_max(axis=0, color="lightgreen"), use_container_width=True)

# --------------------------------------------------------------------------
# TAB 3 — Feature importance dashboard
# --------------------------------------------------------------------------
with tab3:
    st.subheader("Feature Importance Dashboard")
    st.caption(f"Global feature importance for the {best_name} model.")

    fi_df = pd.DataFrame(list(feat_importance.items()), columns=["Feature", "Importance"]).sort_values("Importance")
    fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                 title=f"Global Feature Importance — {best_name}", color="Importance",
                 color_continuous_scale="Viridis")
    fig.update_layout(height=500, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Top Churn Drivers")
    top5 = fi_df.sort_values("Importance", ascending=False).head(5)
    cols = st.columns(5)
    for i, (_, row) in enumerate(top5.iterrows()):
        cols[i].metric(row["Feature"], f"{row['Importance']:.1%}")

    st.markdown("#### Partial Dependence — Sensitivity by Feature")
    pdp_feature = st.selectbox(
        "Choose a feature to see how churn probability changes as it varies "
        "(all other features held at a representative average customer)",
        ["Age", "NumOfProducts", "Balance", "Tenure", "CreditScore", "EstimatedSalary"],
    )
    avg_customer = {
        "CreditScore": raw_df["CreditScore"].mean(), "Geography": "France", "Gender": "Female",
        "Age": raw_df["Age"].mean(), "Tenure": raw_df["Tenure"].mean(), "Balance": raw_df["Balance"].mean(),
        "NumOfProducts": 1, "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": raw_df["EstimatedSalary"].mean(),
    }
    if pdp_feature in ["NumOfProducts", "Tenure"]:
        sweep = list(range(int(raw_df[pdp_feature].min()), int(raw_df[pdp_feature].max()) + 1))
    else:
        sweep = np.linspace(raw_df[pdp_feature].min(), raw_df[pdp_feature].max(), 30)

    sens_df = local_sensitivity(model, avg_customer, pdp_feature, sweep)
    fig = px.line(sens_df, x=pdp_feature, y="ChurnProbability", markers=True,
                  title=f"Churn Probability vs. {pdp_feature} (average customer, all else held constant)")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# TAB 4 — What-if scenario simulator
# --------------------------------------------------------------------------
with tab4:
    st.subheader("What-If Scenario Simulator")
    st.caption("Pick a starting customer profile, then adjust engagement and product values to see how churn risk responds in real time.")

    preset = st.radio(
        "Starting profile", ["Typical loyal customer", "At-risk customer", "Custom (edit below)"],
        horizontal=True,
    )

    if preset == "Typical loyal customer":
        base = {"CreditScore": 700, "Geography": "France", "Gender": "Female", "Age": 35,
                "Tenure": 6, "Balance": 60000.0, "NumOfProducts": 1, "HasCrCard": 1,
                "IsActiveMember": 1, "EstimatedSalary": 90000.0}
    elif preset == "At-risk customer":
        base = {"CreditScore": 600, "Geography": "Germany", "Gender": "Female", "Age": 50,
                "Tenure": 2, "Balance": 130000.0, "NumOfProducts": 3, "HasCrCard": 1,
                "IsActiveMember": 0, "EstimatedSalary": 70000.0}
    else:
        base = {"CreditScore": 650, "Geography": "France", "Gender": "Female", "Age": 40,
                "Tenure": 5, "Balance": 75000.0, "NumOfProducts": 1, "HasCrCard": 1,
                "IsActiveMember": 1, "EstimatedSalary": 100000.0}

    baseline_prob = predict_proba_for_customer(model, base)

    st.markdown("#### Adjust engagement & product values")
    col1, col2 = st.columns(2)
    with col1:
        sim_products = st.slider("Number of Products", 1, 4, base["NumOfProducts"], key="sim_prod")
        sim_active = st.selectbox("Active Member", ["Yes", "No"], index=0 if base["IsActiveMember"] == 1 else 1, key="sim_active")
    with col2:
        sim_tenure = st.slider("Tenure (years)", 0, 10, base["Tenure"], key="sim_tenure")
        sim_balance = st.number_input("Balance (€)", 0.0, 300000.0, float(base["Balance"]), step=5000.0, key="sim_balance")

    scenario = dict(base)
    scenario["NumOfProducts"] = sim_products
    scenario["IsActiveMember"] = 1 if sim_active == "Yes" else 0
    scenario["Tenure"] = sim_tenure
    scenario["Balance"] = sim_balance

    scenario_prob = predict_proba_for_customer(model, scenario)
    delta = scenario_prob - baseline_prob

    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline Churn Probability", f"{baseline_prob:.1%}")
    c2.metric("Scenario Churn Probability", f"{scenario_prob:.1%}", delta=f"{delta:+.1%}", delta_color="inverse")
    c3.metric("Risk Band (scenario)", risk_band(scenario_prob))

    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Baseline", "Scenario"], y=[baseline_prob * 100, scenario_prob * 100],
                          marker_color=[RISK_COLORS[risk_band(baseline_prob)], RISK_COLORS[risk_band(scenario_prob)]],
                          text=[f"{baseline_prob:.1%}", f"{scenario_prob:.1%}"], textposition="outside"))
    fig.update_layout(title="Baseline vs. Scenario Churn Probability", yaxis_title="Churn Probability (%)", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Product-count sensitivity for this scenario")
    sweep_df = local_sensitivity(model, scenario, "NumOfProducts", [1, 2, 3, 4])
    fig = px.bar(sweep_df, x="NumOfProducts", y="ChurnProbability", text="ChurnProbability",
                 title="How churn risk moves if this customer's product count changes")
    fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Predictive Modeling and Risk Scoring for Bank Customer Churn · Unified Mentor Project")
