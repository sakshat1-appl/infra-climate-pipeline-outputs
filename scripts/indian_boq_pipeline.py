import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import json
import os
import zipfile

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Paths
ZIP_PATH = r'C:\Users\HP\AppData\Local\Temp\opencode\kaggle_boq.zip'
OUTDIR = r'C:\Users\HP\AppData\Local\Temp\opencode\output_indian_boq'
os.makedirs(OUTDIR, exist_ok=True)

print("=" * 70)
print("INDIAN CONSTRUCTION COST PIPELINE")
print("Dataset: kaiwalyaatulraut/construction-and-electrical-boq-dataset")
print("Source: Kaggle (CC0 public domain) - synthetic Indian BOQ data")
print("=" * 70)

# ------------------------------------------------------------------
# 1. LOAD
# ------------------------------------------------------------------
print("\n[1/5] Loading dataset ...")
with zipfile.ZipFile(ZIP_PATH) as z:
    projects = pd.read_csv(z.open('boq_projects.csv'))
    items = pd.read_csv(z.open('boq_line_items.csv'))

print("  Projects:", projects.shape[0], "rows x", projects.shape[1], "cols")
print("  Line items:", items.shape[0], "rows x", items.shape[1], "cols")
print("  Project columns:", projects.columns.tolist())

# ------------------------------------------------------------------
# 2. CLEAN & FEATURE ENGINEER
# ------------------------------------------------------------------
print("\n[2/5] Cleaning & feature engineering ...")

df = projects.copy()

# Parse dates
df['project_start_date'] = pd.to_datetime(df['project_start_date'], errors='coerce')
df['start_year'] = df['project_start_date'].dt.year
df['start_month'] = df['project_start_date'].dt.month
df['start_quarter'] = df['project_start_date'].dt.quarter

# Cost components (for feature engineering ONLY - NOT as target predictors directly)
df['civil_share'] = df['total_civil_cost_inr'] / df['total_project_cost_inr']
df['electrical_share'] = df['total_electrical_cost_inr'] / df['total_project_cost_inr']
df['plumbing_share'] = df['total_plumbing_cost_inr'] / df['total_project_cost_inr']

# Item-level aggregates from line items
item_agg = items.groupby('project_id').agg(
    n_items=('item_id', 'count'),
    total_line_amount=('amount_inr', 'sum'),
    avg_rate=('rate_per_unit_inr', 'mean'),
    max_rate=('rate_per_unit_inr', 'max'),
    n_categories=('category', 'nunique'),
    n_subcategories=('sub_category', 'nunique'),
).reset_index()

# Verify line item sum matches project total
item_agg['line_sum_vs_total_ratio'] = item_agg['total_line_amount'] / df.set_index('project_id').loc[item_agg['project_id'], 'total_project_cost_inr'].values
print("  Line-item sum vs project total ratio: mean=%.4f  median=%.4f" % (
    item_agg['line_sum_vs_total_ratio'].mean(), item_agg['line_sum_vs_total_ratio'].median()
))

df = df.merge(item_agg, on='project_id', how='left')

# Cost per sqft (derived, but contains target info - use carefully or exclude)
# We'll EXCLUDE cost_per_sqft_inr from predictors to avoid target leakage
# because it's literally total_project_cost_inr / built_up_area_sqft

# Target
df["target_cost_inr"] = pd.to_numeric(df["total_project_cost_inr"], errors="coerce")
df = df.dropna(subset=["target_cost_inr"]).copy()
print("  Rows with valid target:", len(df))

# ------------------------------------------------------------------
# 3. TIME SERIES (ARIMA/ARMA) - aggregate by month
# ------------------------------------------------------------------
print("\n[3/5] Time-series analysis (ARIMA/ARMA on monthly total project cost) ...")

ts = (
    df.dropna(subset=['project_start_date'])
    .groupby(df['project_start_date'].dt.to_period('Q'))['target_cost_inr']
    .sum()
    .sort_index()
    .astype(float)
)
ts.index = ts.index.to_timestamp()
print("  Quarterly observations:", len(ts))
print("  Date range:", ts.index.min(), "to", ts.index.max())
print("  Series:\n", ts.round(0).to_string())

ts_summary = {
    "n_quarters": int(len(ts)),
    "date_min": str(ts.index.min()) if len(ts) > 0 else None,
    "date_max": str(ts.index.max()) if len(ts) > 0 else None,
    "adf_p_level": None,
    "adf_p_diff": None,
}

arima_result = None
if len(ts.dropna()) >= 8:
    adf = adfuller(ts.dropna())
    ts_summary["adf_p_level"] = float(adf[1])
    ts_diff = ts.diff().dropna()
    if len(ts_diff) >= 8:
        adf_d = adfuller(ts_diff)
        ts_summary["adf_p_diff"] = float(adf_d[1])
    print("  ADF p-value (levels): %.4f" % ts_summary["adf_p_level"])
    print("  ADF p-value (1st diff): %.4f" % ts_summary["adf_p_diff"])

    best = None
    for p in range(0, 3):
        for d in range(0, 2):
            for q in range(0, 3):
                try:
                    m = ARIMA(ts, order=(p, d, q)).fit()
                    if best is None or m.aic < best[0]:
                        best = (m.aic, (p, d, q), m)
                except Exception:
                    continue

    if best is None:
        print("  ARIMA: no convergent model found.")
        arima_result = {"skipped": True, "reason": "no convergent model"}
    else:
        aic, order, model = best
        print("  Best ARIMA order: %s, AIC: %.2f" % (order, aic))
        print("  Params:", model.params.round(4).to_dict())
        fc = model.get_forecast(steps=4)
        fc_mean = fc.predicted_mean
        fc_ci = fc.conf_int()
        print("  4-quarter forecast (total cost INR):")
        for i in range(4):
            print("    Q+%d: %.0f  (CI: %.0f - %.0f)" % (
                i + 1, fc_mean.iloc[i], fc_ci.iloc[i, 0], fc_ci.iloc[i, 1]))
        arima_result = {
            "order": order,
            "aic": float(aic),
            "params": model.params.round(6).to_dict(),
            "forecast": fc_mean.round(0).tolist(),
            "forecast_ci_lower": fc_ci.iloc[:, 0].round(0).tolist(),
            "forecast_ci_upper": fc_ci.iloc[:, 1].round(0).tolist(),
        }
else:
    print("  SKIPPED: only %d quarterly observations (need >= 8 for ARIMA)." % len(ts.dropna()))
    arima_result = {"skipped": True, "reason": "insufficient observations (%d quarters)" % len(ts.dropna())}

# ------------------------------------------------------------------
# 4. REGRESSION - predict total_project_cost_inr
# ------------------------------------------------------------------
print("\n[4/5] Regression: total_project_cost_inr ~ project features ...")

# Feature selection - AVOID target leakage:
# EXCLUDE: total_civil_cost_inr, total_electrical_cost_inr, total_plumbing_cost_inr
#          cost_per_sqft_inr (derived from target / area)
#          total_line_amount (derived from items that sum to target)
feature_cols = [
    "built_up_area_sqft", "num_floors", "estimated_duration_months",
    "start_year", "start_month", "start_quarter",
    "building_type", "city", "state",
    "n_items", "avg_rate", "max_rate", "n_categories", "n_subcategories",
]

reg_df = df[feature_cols + ["target_cost_inr"]].dropna().copy()
print("  Modeling rows:", len(reg_df))
print("  Features:", len(feature_cols))
for c in feature_cols:
    print("    ", c, "-", reg_df[c].dtype)

num_features = [c for c in feature_cols if reg_df[c].dtype in [np.float64, np.int64]]
cat_features = [c for c in feature_cols if c not in num_features]

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ]), num_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
        ]), cat_features),
    ],
    remainder="drop",
)

X = reg_df[feature_cols]
y = reg_df["target_cost_inr"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipe = Pipeline([
    ("prep", preprocess),
    ("reg", Ridge(alpha=1.0)),
])
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print("  Train n: %d  Test n: %d" % (len(y_train), len(y_test)))
print("  R^2:  %.4f" % r2)
print("  RMSE: %.0f INR" % rmse)
print("  MAE:  %.0f INR" % mae)
print("  MAPE: %.2f%%" % mape)

ridge = pipe.named_steps["reg"]
prep = pipe.named_steps["prep"]
try:
    feature_names = prep.get_feature_names_out()
except Exception:
    feature_names = [f"f{i}" for i in range(ridge.coef_.shape[0])]

coef_df = pd.DataFrame({"feature": feature_names, "coef": ridge.coef_})
coef_df = coef_df.reindex(coef_df["coef"].abs().sort_values(ascending=False).index)
print("  Top coefficients by |value|:")
print(coef_df.head(15).to_string(index=False))

residuals = y_test - y_pred
resid_std = residuals.std()
resid_mean = residuals.mean()
print("  Residual mean: %.0f  Residual std: %.0f" % (resid_mean, resid_std))

reg_summary = {
    "n_train": int(len(y_train)),
    "n_test": int(len(y_test)),
    "r2": float(r2),
    "rmse": float(rmse),
    "mae": float(mae),
    "mape": float(mape),
    "resid_mean": float(resid_mean),
    "resid_std": float(resid_std),
    "n_features": int(len(feature_names)),
    "top_features": coef_df.head(15).to_dict("records"),
}

# ------------------------------------------------------------------
# 5. MONTE CARLO
# ------------------------------------------------------------------
print("\n[5/5] Monte Carlo simulation of construction cost outcomes ...")

n_sim = 10000
rng = np.random.default_rng(42)

# Sample feature profiles from the data
sample_idx = rng.choice(len(reg_df), size=n_sim, replace=True)
sample_X = X.iloc[sample_idx].reset_index(drop=True)

# Predicted cost from regression
sim_pred = pipe.predict(sample_X)

# Add residual noise
sim_noise = rng.normal(loc=resid_mean, scale=resid_std, size=n_sim)
sim_actual = sim_pred + sim_noise
sim_actual = np.clip(sim_actual, 0, None)

# Also compute cost per sqft for context
sample_area = reg_df["built_up_area_sqft"].iloc[sample_idx].values
sim_cost_per_sqft = sim_actual / sample_area

def summarize(arr, label):
    return {
        "label": label,
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }

mc = summarize(sim_actual, "monte_carlo_total_cost")
mc_psf = summarize(sim_cost_per_sqft, "monte_carlo_cost_per_sqft")

print("  Simulations: %d projects" % n_sim)
print("\n  TOTAL PROJECT COST (INR):")
for k, v in mc.items():
    if k == "label":
        continue
    print("    %-18s %s" % (k, f"{v:,.0f}" if isinstance(v, float) else v))
print("\n  COST PER SQFT (INR):")
for k, v in mc_psf.items():
    if k == "label":
        continue
    print("    %-18s %s" % (k, f"{v:,.0f}" if isinstance(v, float) else v))

mc_summary = {
    "n_sim": n_sim,
    "total_cost_inr": mc,
    "cost_per_sqft_inr": mc_psf,
}

# ------------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------------
results = {
    "dataset": "kaiwalyaatulraut/construction-and-electrical-boq-dataset",
    "source": "Kaggle (CC0 public domain)",
    "note": "SYNTHETIC data - programmatically generated Indian BOQ rates, not real projects",
    "n_projects_loaded": int(projects.shape[0]),
    "n_line_items": int(items.shape[0]),
    "n_modeling_rows": int(len(reg_df)),
    "feature_cols": feature_cols,
    "target": "total_project_cost_inr",
    "timeseries": ts_summary,
    "arima": arima_result,
    "regression": reg_summary,
    "monte_carlo": mc_summary,
}

out_path = os.path.join(OUTDIR, "indian_boq_pipeline_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

sim_df = pd.DataFrame({
    "sampled_project_id": reg_df["project_id"].iloc[sample_idx].values if "project_id" in reg_df.columns else sample_idx,
    "predicted_total_cost_inr": sim_pred,
    "simulated_total_cost_inr": sim_actual,
    "built_up_area_sqft": sample_area,
    "simulated_cost_per_sqft_inr": sim_cost_per_sqft,
})
sim_path = os.path.join(OUTDIR, "monte_carlo_draws.csv")
sim_df.to_csv(sim_path, index=False)

# Also save cleaned modeling dataset
model_out = os.path.join(OUTDIR, "modeling_dataset.csv")
reg_df.to_csv(model_out, index=False)

print("\n" + "=" * 70)
print("RESULTS SAVED")
print("  JSON: %s" % out_path)
print("  CSV draws: %s" % sim_path)
print("  Modeling dataset: %s" % model_out)
print("=" * 70)
