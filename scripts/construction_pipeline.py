import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import io
import json
import os
import sys
from datetime import datetime

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

URL = "http://training.theodi.org/resources/dataset2.csv"
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTDIR, exist_ok=True)

print("=" * 70)
print("CONSTRUCTION PROJECT OUTCOME PIPELINE")
print("External source:", URL)
print("=" * 70)

# ------------------------------------------------------------------
# 1. LOAD
# ------------------------------------------------------------------
print("\n[1/5] Loading dataset ...")
r = requests.get(URL, timeout=60)
r.raise_for_status()
df = pd.read_csv(io.StringIO(r.text))
print("  Rows:", len(df), " Cols:", len(df.columns))
print("  Columns:", df.columns.tolist())

# ------------------------------------------------------------------
# 2. CLEAN
# ------------------------------------------------------------------
print("\n[2/5] Cleaning ...")

numeric_cols = [
    "Lifecycle Cost",
    "Schedule Variance (in days)",
    "Schedule Variance (%)",
    "Cost Variance ($ M)",
    "Cost Variance (%)",
    "Planned Cost ($ M)",
    "Projected/Actual Cost ($ M)",
]
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

date_cols = ["Start Date", "Completion Date (B1)", "Planned Project Completion Date (B2)",
             "Projected/Actual Project Completion Date (B2)", "Updated Date"]
for c in date_cols:
    df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)

# Derive clean target and key features
df["actual_cost_m"] = df["Projected/Actual Cost ($ M)"]
df["planned_cost_m"] = df["Planned Cost ($ M)"]
df["cost_var_pct"] = df["Cost Variance (%)"]
df["sched_var_days"] = df["Schedule Variance (in days)"]
df["updated_q"] = df["Updated Date"].dt.to_period("Q")

# Drop rows missing the core fields
core = ["actual_cost_m", "planned_cost_m", "cost_var_pct", "Updated Date"]
df_clean = df.dropna(subset=core).copy()
print("  Rows after dropping missing core fields:", len(df_clean))

# Remove extreme outliers (cost variance > 5000% or actual/planned ratio absurd)
df_clean["cost_ratio"] = df_clean["actual_cost_m"] / df_clean["planned_cost_m"]
df_clean = df_clean[(df_clean["cost_ratio"] > 0.01) & (df_clean["cost_ratio"] < 100)].copy()
print("  Rows after outlier filter:", len(df_clean))

# ------------------------------------------------------------------
# 3. TIME SERIES: ARIMA / ARMA on quarterly actual cost
# ------------------------------------------------------------------
print("\n[3/5] Time-series analysis (ARIMA/ARMA on quarterly actual cost) ...")

ts = (
    df_clean.groupby("updated_q")["actual_cost_m"]
    .sum()
    .sort_index()
    .astype(float)
)
ts.index = ts.index.to_timestamp()
ts = ts.asfreq("QE")  # ensure quarterly frequency (pandas 3.0 renamed 'Q' -> 'QE')
print("  Quarterly observations:", len(ts))
print("  Date range:", ts.index.min(), "to", ts.index.max())
print("  Series head:\n", ts.head(4).to_string())

# Stationarity check (only if enough observations)
ts_summary = {
    "n_quarters": int(len(ts)),
    "date_min": str(ts.index.min()) if len(ts) > 0 else None,
    "date_max": str(ts.index.max()) if len(ts) > 0 else None,
    "adf_p_level": None,
    "adf_p_diff": None,
}

if len(ts.dropna()) >= 8:
    adf = adfuller(ts.dropna())
    ts_summary["adf_p_level"] = float(adf[1])
    ts_diff = ts.diff().dropna()
    if len(ts_diff) >= 8:
        adf_d = adfuller(ts_diff)
        ts_summary["adf_p_diff"] = float(adf_d[1])
    print("  ADF p-value (levels): %.4f" % ts_summary["adf_p_level"])
    print("  ADF p-value (1st diff): %s" % (
        "%.4f" % ts_summary["adf_p_diff"] if ts_summary["adf_p_diff"] else "n/a (too few diffs)"))

    # Fit ARIMA (order selected by AIC over a small grid)
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
        arima_result = None
    else:
        aic, order, model = best
        print("  Best ARIMA order: %s, AIC: %.2f" % (order, aic))
        print("  Params:", model.params.round(4).to_dict())
        # Forecast next 4 quarters
        fc = model.get_forecast(steps=4)
        fc_mean = fc.predicted_mean
        fc_ci = fc.conf_int()
        print("  4-quarter forecast (actual cost $M):")
        for i in range(4):
            print("    Q+%d: %.2f  (CI: %.2f - %.2f)" % (
                i + 1, fc_mean.iloc[i], fc_ci.iloc[i, 0], fc_ci.iloc[i, 1]))
        arima_result = {
            "order": order,
            "aic": float(aic),
            "params": model.params.round(6).to_dict(),
            "forecast": fc_mean.round(4).tolist(),
            "forecast_ci_lower": fc_ci.iloc[:, 0].round(4).tolist(),
            "forecast_ci_upper": fc_ci.iloc[:, 1].round(4).tolist(),
        }
else:
    print("  SKIPPED: only %d quarterly observations (need >= 8 for ARIMA)." % len(ts.dropna()))
    print("  Date range too short (%s to %s) for a meaningful ARIMA/ARMA model." % (
        ts_summary["date_min"], ts_summary["date_max"]))
    arima_result = {
        "skipped": True,
        "reason": "insufficient observations (%d quarters)" % len(ts.dropna()),
    }

# ------------------------------------------------------------------
# 4. REGRESSION: predict actual cost from planned cost + features
# ------------------------------------------------------------------
print("\n[4/5] Regression: actual_cost_m ~ planned_cost_m + cost_var_pct + sched_var_days + agency ...")

feature_cols = ["planned_cost_m", "cost_var_pct", "sched_var_days", "Agency Name", "Investment Title"]
reg_df = df_clean[feature_cols + ["actual_cost_m"]].dropna().copy()

# Encode categorical
cat_features = ["Agency Name", "Investment Title"]
num_features = ["planned_cost_m", "cost_var_pct", "sched_var_days"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), cat_features),
    ],
    remainder="drop",
)

X = reg_df[feature_cols]
y = reg_df["actual_cost_m"]

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
print("  Train n: %d  Test n: %d" % (len(y_train), len(y_test)))
print("  R^2:  %.4f" % r2)
print("  RMSE: %.4f" % rmse)
print("  MAE:  %.4f" % mae)

# Coefficient interpretation for numeric features
ridge = pipe.named_steps["reg"]
prep = pipe.named_steps["prep"]
# Get feature names after transform
try:
    feature_names = prep.get_feature_names_out()
except Exception:
    feature_names = [f"f{i}" for i in range(ridge.coef_.shape[0])]

coef_df = pd.DataFrame({"feature": feature_names, "coef": ridge.coef_})
coef_df = coef_df.reindex(coef_df["coef"].abs().sort_values(ascending=False).index)
print("  Top coefficients by |value|:")
print(coef_df.head(12).to_string(index=False))

# Residuals for Monte Carlo
residuals = y_test - y_pred
resid_std = residuals.std()
resid_mean = residuals.mean()
print("  Residual mean: %.4f  Residual std: %.4f" % (resid_mean, resid_std))

reg_summary = {
    "n_train": int(len(y_train)),
    "n_test": int(len(y_test)),
    "r2": float(r2),
    "rmse": float(rmse),
    "mae": float(mae),
    "resid_mean": float(resid_mean),
    "resid_std": float(resid_std),
    "n_features": int(len(feature_names)),
    "top_features": coef_df.head(10).to_dict("records"),
}

# ------------------------------------------------------------------
# 5. MONTE CARLO: simulate future project cost outcomes
# ------------------------------------------------------------------
print("\n[5/5] Monte Carlo simulation of construction cost outcomes ...")

n_sim = 10000
rng = np.random.default_rng(42)

# Use a representative set of planned costs from the data
planned_costs = reg_df["planned_cost_m"].values
sample_planned = rng.choice(planned_costs, size=n_sim, replace=True)

# Cost variance % distribution from data (empirical)
cost_var_pct_vals = reg_df["cost_var_pct"].dropna().values

# Simulation 1: Empirical bootstrap of cost variance %
sim_cost_var_emp = rng.choice(cost_var_pct_vals, size=n_sim, replace=True)
sim_actual_emp = sample_planned * (1 + sim_cost_var_emp / 100.0)

# Simulation 2: Regression-based with normal residual noise
# predicted = planned * (1 + expected_var_pct/100) + noise
# Use mean cost_var_pct as baseline expectation
mean_var_pct = reg_df["cost_var_pct"].mean()
sim_pred_base = sample_planned * (1 + mean_var_pct / 100.0)
sim_noise = rng.normal(loc=resid_mean, scale=resid_std, size=n_sim)
sim_actual_reg = sim_pred_base + sim_noise

# Clip negatives
sim_actual_emp = np.clip(sim_actual_emp, 0, None)
sim_actual_reg = np.clip(sim_actual_reg, 0, None)

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
        "prob_overrun": float(np.mean(arr > sample_planned)),
        "mean_overrun_pct": float(np.mean((arr - sample_planned) / sample_planned * 100)),
    }

mc_emp = summarize(sim_actual_emp, "empirical-bootstrap")
mc_reg = summarize(sim_actual_reg, "regression-residual")

print("  Simulations: %d projects" % n_sim)
print("\n  EMPIRICAL BOOTSTRAP (sampled historical cost variance %):")
for k, v in mc_emp.items():
    if k == "label":
        continue
    print("    %-18s %s" % (k, v))
print("\n  REGRESSION + RESIDUAL NOISE:")
for k, v in mc_reg.items():
    if k == "label":
        continue
    print("    %-18s %s" % (k, v))

# Value-at-Risk style: cost overrun at 95% confidence
overrun_pct_emp = (sim_actual_emp - sample_planned) / sample_planned * 100
var95_overrun = np.percentile(overrun_pct_emp, 95)
print("\n  VaR-style: 95th percentile cost overrun = %.2f%%" % var95_overrun)

mc_summary = {
    "n_sim": n_sim,
    "empirical": mc_emp,
    "regression": mc_reg,
    "var95_overrun_pct": float(var95_overrun),
}

# ------------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------------
results = {
    "source": URL,
    "n_rows_loaded": int(len(df)),
    "n_rows_clean": int(len(df_clean)),
    "timeseries": ts_summary,
    "arima": arima_result,
    "regression": reg_summary,
    "monte_carlo": mc_summary,
}

out_path = os.path.join(OUTDIR, "pipeline_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

# Also save the simulated draws for inspection
sim_df = pd.DataFrame({
    "planned_cost_m": sample_planned,
    "actual_cost_empirical": sim_actual_emp,
    "actual_cost_regression": sim_actual_reg,
    "overrun_pct_empirical": overrun_pct_emp,
})
sim_path = os.path.join(OUTDIR, "monte_carlo_draws.csv")
sim_df.to_csv(sim_path, index=False)

print("\n" + "=" * 70)
print("RESULTS SAVED")
print("  JSON: %s" % out_path)
print("  CSV:  %s" % sim_path)
print("=" * 70)
