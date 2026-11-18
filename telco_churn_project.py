"""
Introduction to Data Science - Fall 2026 - Individual Project
Author  : Albion Hoxha
Dataset : Telco Customer Churn (WA_Fn-UseC_-Telco-Customer-Churn.csv)
Goal    : Predict whether a customer will churn (leave the company)
Models  : Logistic Regression (Logit), Random Forest, Soft-Voting Ensemble

How to run:
    python telco_churn_project.py [path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv]
Outputs (figures + metrics.json) are written to the ./outputs folder.
"""

# =============================================================================
# 0. IMPORTS AND SETTINGS
# =============================================================================
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # save figures without opening windows
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, accuracy_score)

RANDOM_STATE = 42                           # fixed seed -> reproducible results
DATA_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("WA_Fn-UseC_-Telco-Customer-Churn.csv")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
results = {}                                # everything we report is collected here

plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})
C_STAY, C_CHURN = "#2E6F8E", "#D1495B"      # colors used consistently in all plots


# =============================================================================
# 1. LOAD AND DESCRIBE THE RAW DATA
# =============================================================================
df_raw = pd.read_csv(DATA_PATH)
print("Raw shape:", df_raw.shape)
print(df_raw.dtypes)

results["n_rows"], results["n_cols"] = df_raw.shape
results["churn_rate"] = round((df_raw["Churn"] == "Yes").mean() * 100, 2)

# TotalCharges is stored as text; blank strings (" ") hide missing values.
blank_tc = (df_raw["TotalCharges"].str.strip() == "").sum()
results["blank_totalcharges"] = int(blank_tc)
print(f"Blank TotalCharges values: {blank_tc}")

# Descriptive statistics for the numeric fields
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
tmp = df_raw.copy()
tmp["TotalCharges"] = pd.to_numeric(tmp["TotalCharges"], errors="coerce")
desc = tmp[num_cols].describe().T.round(2)
print(desc)

# Outlier check with the IQR rule (value < Q1-1.5*IQR or > Q3+1.5*IQR)
def iqr_outliers(s):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())

outliers = {c: iqr_outliers(tmp[c].dropna()) for c in num_cols}
# Same rule inside the churned group only (these are the circles in the boxplots)
churned = tmp[tmp["Churn"] == "Yes"]
results["outliers_churned"] = {c: iqr_outliers(churned[c].dropna()) for c in num_cols}
print("IQR outliers among churners:", results["outliers_churned"])
desc["skew"] = tmp[num_cols].skew().round(2)
desc["iqr_outliers"] = pd.Series(outliers)
results["describe_numeric"] = desc.reset_index().rename(columns={"index": "field"}).to_dict("records")
print("IQR outliers per numeric field:", outliers)

# Field overview: type, number of distinct values, most common value and its share
field_info = pd.DataFrame({"dtype": df_raw.dtypes.astype(str),
                           "n_unique": df_raw.nunique(),
                           "top": tmp.apply(lambda s: str(s.mode().iloc[0])),
                           "top_share": tmp.apply(lambda s: s.value_counts(normalize=True).iloc[0] * 100).round(1)})
results["fields"] = field_info.reset_index().rename(columns={"index": "field"}).to_dict("records")

# Churn rate by contract and payment method (justifies the M2M_ECheck variable)
cat_summary = {}
for c in ["Contract", "PaymentMethod"]:
    cat_summary[c] = (df_raw.groupby(c)["Churn"]
                      .apply(lambda s: round((s == "Yes").mean() * 100, 1)).to_dict())
results["churn_by_category"] = {k: {str(a): b for a, b in v.items()} for k, v in cat_summary.items()}

# Figure 1: boxplots of numeric fields split by churn (shows outliers + separation)
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
for ax, c in zip(axes, num_cols):
    data = [tmp.loc[tmp["Churn"] == "No", c].dropna(), tmp.loc[tmp["Churn"] == "Yes", c].dropna()]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops={"color": "black"})
    for patch, col in zip(bp["boxes"], [C_STAY, C_CHURN]):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    ax.set_xticks([1, 2], ["Stayed", "Churned"])
    ax.set_title(c)
fig.tight_layout()
fig.savefig(OUT / "fig1_boxplots.png"); plt.close(fig)


# =============================================================================
# 2. DATA CLEANING AND VARIABLE GENERATION
# =============================================================================
df = df_raw.copy()

# 2.1 Drop the ID column - it is a unique identifier with no predictive meaning
df = df.drop(columns=["customerID"])

# 2.2 Convert TotalCharges to numeric. The 11 blanks all have tenure == 0
#     (brand-new customers who were never billed), so the correct value is 0.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
assert (df.loc[df["TotalCharges"].isna(), "tenure"] == 0).all()
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# 2.3 Target variable: Churn Yes/No -> 1/0
df["Churn"] = (df["Churn"] == "Yes").astype(int)

# 2.4 Simplify "No internet service" / "No phone service" to "No".
#     The information is already contained in InternetService / PhoneService,
#     so keeping it would create perfectly collinear dummies.
for c in ["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
          "TechSupport", "StreamingTV", "StreamingMovies"]:
    df[c] = df[c].replace({"No internet service": "No", "No phone service": "No"})

# 2.5 New variables (feature engineering)
addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
          "TechSupport", "StreamingTV", "StreamingMovies"]
# Number of add-on services: more services -> customer is more "locked in"
df["NumAddOns"] = (df[addons] == "Yes").sum(axis=1)
# Tenure groups: churn risk is highest in the first year
df["TenureGroup"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72],
                           labels=["0-12m", "13-24m", "25-48m", "49-72m"])
# Average monthly spend over the whole relationship (smooths price changes)
df["AvgMonthlySpend"] = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"],
                                 df["MonthlyCharges"])
# Flag for the riskiest combination seen in EDA: month-to-month + electronic check
df["M2M_ECheck"] = ((df["Contract"] == "Month-to-month") &
                    (df["PaymentMethod"] == "Electronic check")).astype(int)
# Log of TotalCharges: the raw field is strongly right-skewed
df["LogTotalCharges"] = np.log1p(df["TotalCharges"])

# 2.6 Drop TotalCharges itself: it is almost tenure x MonthlyCharges
#     (corr > 0.8), so it adds multicollinearity to the Logit model.
corr_tc = df[["TotalCharges", "tenure"]].corr().iloc[0, 1]
results["corr_totalcharges_tenure"] = round(corr_tc, 3)
df = df.drop(columns=["TotalCharges"])

# 2.7 One-hot encode categorical fields (drop_first avoids the dummy trap)
cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]  # works in pandas 2 and 3
X = pd.get_dummies(df.drop(columns=["Churn"]), columns=cat_cols, drop_first=True, dtype=int)
y = df["Churn"]
results["n_features"] = X.shape[1]
results["new_features"] = ["NumAddOns", "TenureGroup", "AvgMonthlySpend",
                           "M2M_ECheck", "LogTotalCharges"]
print("Model matrix:", X.shape)

# Churn rate by tenure group (supports the TenureGroup variable in the report)
results["churn_by_tenure_group"] = (df.groupby("TenureGroup", observed=True)["Churn"]
                                    .mean().mul(100).round(1).astype(float).to_dict())


# =============================================================================
# 3. TRAIN-TEST SPLIT
# =============================================================================
# The data is a single cross-section (no dates), so a time-based split is not
# possible. We use a RANDOM split, STRATIFIED on Churn, so that both samples
# keep the same 26.5% churn share (classes are imbalanced). 80% / 20%.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
results["split"] = {"train": len(X_train), "test": len(X_test),
                    "train_churn": round(float(y_train.mean()) * 100, 2),
                    "test_churn": round(float(y_test.mean()) * 100, 2)}
print("Train/test:", results["split"])


# =============================================================================
# 4. MODELS
# =============================================================================
# 4.1 Logit: interpretable baseline. Features are standardized so coefficients
#     are comparable. class_weight="balanced" up-weights the minority class.
logit = Pipeline([("scale", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                             random_state=RANDOM_STATE))])

# 4.2 Random Forest: captures non-linearities and interactions automatically.
#     Depth and leaf size are limited to reduce overfitting.
rf = RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=10,
                            class_weight="balanced", n_jobs=-1,
                            random_state=RANDOM_STATE)

# 4.3 Ensemble: soft voting averages the predicted probabilities of both models
ens = VotingClassifier([("logit", logit), ("rf", rf)], voting="soft")

models = {"Logit": logit, "Random Forest": rf, "Ensemble": ens}


# =============================================================================
# 5. EVALUATION ON THE TEST SAMPLE
# =============================================================================
results["test"] = {}
probas = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    p = m.predict_proba(X_test)[:, 1]
    pred = (p >= 0.5).astype(int)
    probas[name] = p
    cm = confusion_matrix(y_test, pred)
    rep = classification_report(y_test, pred, target_names=["Stay", "Churn"],
                                output_dict=True)
    print(f"\n===== {name} =====")
    print(classification_report(y_test, pred, target_names=["Stay", "Churn"]))
    print("Confusion matrix [rows=actual, cols=predicted]:\n", cm)
    results["test"][name] = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, p), 4),
        "confusion_matrix": cm.tolist(),
        "report": {k: {kk: round(vv, 2) for kk, vv in v.items()}
                   for k, v in rep.items() if isinstance(v, dict)}}

# Naive benchmark: always predict "Stay"
results["baseline_accuracy"] = round(1 - y_test.mean(), 4)

# Figure 2: confusion matrices of the three models side by side
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
for ax, (name, r) in zip(axes, results["test"].items()):
    cm = np.array(r["confusion_matrix"])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1], ["Stay", "Churn"]); ax.set_yticks([0, 1], ["Stay", "Churn"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(name)
fig.tight_layout()
fig.savefig(OUT / "fig2_confusion_matrices.png"); plt.close(fig)


# =============================================================================
# 6. BIAS / ROBUSTNESS CHECKS
# =============================================================================
# Best model = highest ROC-AUC on the test sample
best = max(results["test"], key=lambda k: results["test"][k]["roc_auc"])
results["best_model"] = best

# 6.1 Does performance differ across sub-groups? (fairness / generalization)
test_df = X_test.copy()
test_df["y"], test_df["p"] = y_test.values, probas[best]
groups = {"Female": X_test["gender_Male"] == 0, "Male": X_test["gender_Male"] == 1,
          "Senior": X_test["SeniorCitizen"] == 1, "Non-senior": X_test["SeniorCitizen"] == 0}
results["subgroup_auc"] = {}
for g, mask in groups.items():
    sub = test_df[mask.values]
    results["subgroup_auc"][g] = {"n": int(len(sub)),
                                  "auc": round(roc_auc_score(sub["y"], sub["p"]), 3),
                                  "churn_rate": round(sub["y"].mean() * 100, 1)}

# 6.2 Overfitting check: compare train vs test AUC of the best model
train_auc = roc_auc_score(y_train, models[best].predict_proba(X_train)[:, 1])
results["overfit_check"] = {"train_auc": round(train_auc, 4),
                            "test_auc": results["test"][best]["roc_auc"]}

# =============================================================================
# 7. SAVE RESULTS
# =============================================================================
with open(OUT / "metrics.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nBest model: {best}. Results and figures saved in '{OUT}/'.")
