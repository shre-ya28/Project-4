# Predictive Modeling and Risk Scoring for Bank Customer Churn

**Prepared for:** The European Central Bank (via Unified Mentor)
**Analysis type:** Predictive Modeling, Churn Risk Scoring, Model Explainability
**Dataset:** 10,000 retail banking customers across France, Spain, and Germany

---

## Abstract

Traditional churn analysis explains why customers left after the fact; this project
builds a predictive system that scores customers **before** they leave. We engineer
four behavioral features on top of the raw customer profile, train and compare five
classification models (Logistic Regression, Decision Tree, Random Forest, Gradient
Boosting, and — where available — XGBoost), and select the best performer by ROC-AUC.
The selected **Gradient Boosting model achieves an ROC-AUC of 0.870**, an accuracy of
87.2%, and identifies **product count and its interaction with customer engagement as
the dominant churn driver** — ahead of age, balance, and geography. We provide global
and per-customer explainability (feature importance, partial dependence, and local
attribution) and operationalize the model in a Streamlit application with a live risk
calculator, a probability-distribution monitor, a feature importance dashboard, and a
what-if scenario simulator for retention planning.

---

## 1. Background and Context

Churn erodes customer lifetime value, destabilizes revenue, and closes off cross-sell
and upsell opportunities. Knowing that a customer churned is retrospective; knowing
that a customer is *likely* to churn — with enough lead time to act — is what enables
proactive retention campaigns, personalized offers, and targeted engagement. This
project reframes the Bank's churn problem from description to prediction: instead of
asking "why did these customers leave," it asks "which of today's customers are at
risk, and why."

## 2. Problem Statement

Despite holding rich customer-level data, the Bank currently lacks (1) an accurate
churn prediction model, (2) a quantitative, per-customer churn risk score, and (3)
explainable insight into what drives that score. Without these, retention actions stay
reactive, broad, and costly rather than proactive and targeted.

## 3. Dataset and Target Definition

The dataset contains 10,000 customer records with `CreditScore`, `Geography`,
`Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`,
and `EstimatedSalary` as predictors, and the binary `Exited` field as the prediction
target (1 = churned, 0 = retained). The model outputs both a continuous **churn
probability (0–1)** and a **binary risk flag** derived by comparing that probability
to an adjustable threshold (default 0.5).

## 4. Methodology

### 4.1 Data Preprocessing
- No missing values were present, but the pipeline includes median/mode imputation
  for numeric/categorical fields respectively so it is robust to future data drops.
- Non-informative identifier fields (`CustomerId`, `Surname`) were removed.
- `Geography` and `Gender` were one-hot encoded (`drop_first=True`), yielding
  `Geography_Germany`, `Geography_Spain`, and `Gender_Male` as binary predictors
  (France and Female are the reference levels).
- Numerical features were standardized for the Logistic Regression baseline via a
  `StandardScaler` step inside the model pipeline; tree-based models used raw scale,
  which they are invariant to.

### 4.2 Feature Engineering
Four behavioral features were engineered on top of the raw fields, per the project
brief:

| Feature | Formula | Rationale |
|---|---|---|
| BalanceToSalaryRatio | `Balance / (EstimatedSalary + 1)` | Captures how large a customer's balance is relative to their income |
| ProductDensity | `NumOfProducts / (Tenure + 1)` | Flags customers who accumulated many products quickly |
| EngagementProductInteraction | `IsActiveMember × NumOfProducts` | Distinguishes engaged multi-product customers from disengaged ones |
| AgeTenureInteraction | `Age × Tenure` | Captures relationship depth relative to customer life stage |

### 4.3 Train–Test Strategy
An 80/20 **stratified** split preserved the 20.37% churn class distribution in both
sets. 5-fold **stratified cross-validation** on the training set was used to confirm
that each model's ROC-AUC was stable rather than a product of a favorable split.

### 4.4 Model Development
Five models were trained and compared:

- **Logistic Regression** — interpretability benchmark, class-balanced.
- **Decision Tree** — shallow (max depth 6) for interpretability.
- **Random Forest** — 300 trees, class-balanced.
- **Gradient Boosting** — 250 estimators, shallow trees, low learning rate.
- **XGBoost** — included automatically if the optional `xgboost` package is installed
  in the runtime environment; the pipeline degrades gracefully to the four models
  above if it is not, per the brief's "optional" designation.

## 5. Model Evaluation

![Model comparison](../images/01_model_comparison.png)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.710 | 0.383 | 0.698 | 0.494 | 0.776 |
| Decision Tree | 0.762 | 0.451 | 0.784 | 0.573 | 0.842 |
| Random Forest | 0.820 | 0.542 | 0.727 | 0.621 | 0.866 |
| **Gradient Boosting (selected)** | **0.872** | **0.794** | **0.501** | **0.615** | **0.870** |

*(Exact figures regenerate slightly between training runs; see `models/metrics.json`
for the authoritative current values, and the live dashboard for the figures behind
this deployment.)*

**Gradient Boosting was selected as the best model by ROC-AUC** — the metric the
project brief specifies for model discrimination, and the most threshold-independent
way to compare models. Random Forest is a very close second and offers materially
higher recall at the default threshold; both are saved in `models/all_models.pkl` so
the operating model can be swapped without retraining.

![ROC curves](../images/02_roc_curves.png)

![Confusion matrix](../images/03_confusion_matrix.png)

### 5.1 Precision–Recall Trade-off

At the default 0.5 threshold, Gradient Boosting favors **precision (79.4%) over
recall (50.1%)** — it raises fewer false alarms but also misses more true churners
than Random Forest does. Because the cost of a missed churner (lost customer) and a
false alarm (an unnecessary retention offer) are not symmetric, and that asymmetry is
a business decision rather than a modeling one, **the classification threshold is
exposed as a live control in the Streamlit dashboard** rather than hard-coded. Lowering
the threshold trades some precision for higher recall — useful if the Bank prefers to
cast a wider net for a lower-cost retention channel (e.g., an email offer) versus a
higher-cost one (e.g., a relationship-manager call).

## 6. Model Explainability

### 6.1 Global Feature Importance

![Feature importance](../images/04_feature_importance.png)

`NumOfProducts` and `Age` are the two dominant drivers of the model's predictions, with
the engineered `EngagementProductInteraction` feature and `Balance` following. This
matches and sharpens the segmentation-level findings from the companion churn
segmentation project: **it is not products or engagement alone that drives churn, but
their combination** — a highly engaged, multi-product customer behaves very
differently from a disengaged one with the same product count.

### 6.2 Partial Dependence Plots

![Partial dependence plots](../images/06_partial_dependence.png)

The partial dependence plots make the *direction* of each effect explicit:

- **Age:** churn probability rises steadily through the 40s and peaks around the
  early-to-mid 50s before declining for older customers — the same mid-career risk
  window identified in the segmentation analysis, now with a precise shape rather
  than a coarse age band.
- **NumOfProducts:** a sharp, non-linear jump in predicted churn risk at 3+ products.
- **IsActiveMember:** a clear step down in risk when a customer is active, confirming
  activity status as a strong, actionable signal.
- **Balance / EngagementProductInteraction / ProductDensity:** all show a directionally
  consistent, if gentler, relationship with predicted churn.

### 6.3 SHAP Value Analysis (Optional Dependency)

True SHAP values provide the most rigorous per-customer attribution and are supported
in the accompanying notebook (`notebooks/Model_Development_and_Evaluation.ipynb`,
Section 6.3) if the optional `shap` package is installed (`pip install shap`). To keep
the core project dependency-light and fully reproducible without extra installs, the
Streamlit application uses a **model-agnostic local attribution method**: for a given
customer, each feature is reset to a population-baseline value (an average, engaged,
one-product customer) one at a time, and the resulting change in predicted probability
is recorded as that feature's contribution. This produces the same directional insight
as SHAP for individual customers without requiring the extra dependency, and both
methods are available side by side in the notebook for validation.

### 6.4 Explainability and Regulatory Compliance

Every prediction produced by the deployed model can be decomposed into (a) a global
ranking of what matters most across the whole customer base, and (b) a per-customer
breakdown of what pushed *this* customer's score up or down — the level of
transparency required for the model to be defensible to regulators, auditors, and the
relationship managers who will act on its output.

## 7. Deployment — Streamlit Predictive Churn Risk Scoring App

The application (`app/streamlit_app.py`) operationalizes the model behind four
modules:

1. **Risk Calculator** — enter any customer's profile and receive a live churn
   probability, a risk band (Low / Medium / High), a gauge visualization, and a local
   feature-contribution chart explaining the score.
2. **Probability Distribution** — a live histogram of the model's predicted
   probabilities on the held-out test set, split by actual outcome, with an adjustable
   threshold that recomputes precision/recall/F1 in real time, plus the full ROC-curve
   and metric comparison across all trained models.
3. **Feature Importance Dashboard** — global feature importance for the deployed
   model, plus an interactive partial-dependence sweep for any selected feature.
4. **What-If Scenario Simulator** — starting from a preset or custom customer profile,
   adjust engagement and product values (activity status, product count, tenure,
   balance) and watch the churn probability update live, including a product-count
   sensitivity chart for the selected scenario.

## 8. Key Findings Summary

- The best model (Gradient Boosting) discriminates churners from retained customers
  with an ROC-AUC of 0.870 — a strong, deployable level of performance for this
  dataset size and feature set.
- **Product count, and specifically the interaction between product count and
  engagement, is the single most important churn driver** — ahead of demographic
  variables like geography or gender.
- **Age has a distinct, non-monotonic relationship with churn**, peaking in the
  early-to-mid 50s.
- **Activity status is both highly important and directly actionable**, making it a
  natural target for an automated early-warning trigger.
- The precision/recall trade-off at the default threshold favors avoiding false
  alarms; the Bank can shift this trade-off via the threshold control depending on the
  cost of the retention channel being deployed.

## 9. Recommendations

1. **Deploy the model as a scoring layer feeding relationship managers**, prioritizing
   customers in the High Risk band (probability ≥ 0.6) for direct outreach.
2. **Use the What-If Simulator in relationship-manager workflows** to test whether a
   proposed intervention (e.g., a cross-sell offer, a re-engagement campaign) is
   projected to reduce a specific customer's risk before committing the offer.
3. **Set the classification threshold per retention channel**, not globally — a lower
   threshold for low-cost channels (email), a higher threshold for high-cost channels
   (relationship-manager calls).
4. **Monitor the product-count risk signal operationally**: route any customer
   crossing into a 3rd or 4th product into a proactive engagement check rather than
   further product marketing, consistent with both this predictive analysis and the
   companion segmentation study.
5. **Refresh the model periodically** as new churn outcomes accumulate, and re-run the
   stratified cross-validation check in the training script to confirm performance
   remains stable before redeploying.

## 10. Conclusion

This project converts the Bank's churn problem from a retrospective explanation into a
forward-looking, per-customer risk score, backed by a model that performs well (ROC-AUC
0.870), is explainable at both the global and individual level, and is deployed in a
live application that lets business users calculate risk, monitor model behavior, and
simulate interventions — turning churn management from a reactive, broad-based effort
into a proactive, targeted one.
