import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import json
import os

from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

REPO_ROOT = r"D:\INFRA_ TEST\repo"
ASSET_MATRIX = os.path.join(REPO_ROOT, "integration", "data_output", "feature_matrix_asset.csv")
MONTH_MATRIX = os.path.join(REPO_ROOT, "integration", "data_output", "feature_matrix_asset_month.csv")
COVERAGE_REPORT = os.path.join(REPO_ROOT, "integration", "data_output", "coverage_report.json")
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_integration")
os.makedirs(OUTDIR, exist_ok=True)

print("=" * 70)
print("CONSTRUCTION PROJECT OUTCOME PIPELINE — INTEGRATION FEATURES")
print("=" * 70)
print("Repo root:", REPO_ROOT)
print()

# ------------------------------------------------------------------
# 1. LOAD INTEGRATION FEATURES (asset level)
# ------------------------------------------------------------------
print("[1/5] Loading integration asset feature matrix ...")
asset_df = pd.read_csv(ASSET_MATRIX)
print("  Shape:", asset_df.shape)
print("  Assets:", asset_df["asset_id"].nunique())

# Highway proximity features
hwy_cols = [
    "rd_dist_nh_m", "rd_hwy_len_10km_m", "rd_nh_len_10km_m",
    "rd_node_count_10km", "rd_node_degree_mean_10km", "rd_on_corridor_flag",
]
# Maritime proximity features
mar_cols = [
    "mt_coast_dist_m", "mt_coastal_flag", "mt_dist_major_port_m",
    "mt_in_port_polygon", "mt_ports_within_100km", "mt_pw_dist_m",
    "mt_pw_vessel_count_total_yr", "mt_pw_vessel_count_container_yr",
    "mt_pw_share_import_pct", "mt_pw_share_export_pct",
]
# Weather coverage features
wx_cols = ["wx_up_covered", "wx_up_distance_km", "wx_point_covered"]

for c in hwy_cols + mar_cols + wx_cols:
    if c in asset_df.columns:
        asset_df[c] = pd.to_numeric(asset_df[c], errors="coerce")

print("  Highway cols present:", [c for c in hwy_cols if c in asset_df.columns])
print("  Maritime cols present:", [c for c in mar_cols if c in asset_df.columns])
print("  Weather cols present:", [c for c in wx_cols if c in asset_df.columns])

# ------------------------------------------------------------------
# 2. AGGREGATE MONTHLY RAINFALL TO ASSET LEVEL
# ------------------------------------------------------------------
print("\n[2/5] Aggregating monthly rainfall features to asset level ...")
month_df = pd.read_csv(MONTH_MATRIX)
print("  Month matrix shape:", month_df.shape)

rain_cols = [
    "pan_wx_precip_month_mm", "pan_wx_rain_days_month",
    "pan_wx_monsoon_JJAS_mm", "pan_wx_anomaly_z",
    "up_wx_precip_month_mm", "up_wx_rain_days_month",
    "up_wx_monsoon_JJAS_mm", "up_wx_anomaly_z",
]
for c in rain_cols:
    if c in month_df.columns:
        month_df[c] = pd.to_numeric(month_df[c], errors="coerce")

agg = month_df.groupby("asset_id")[rain_cols].agg(["mean", "std", "max"])
agg.columns = ["_".join(col).strip() for col in agg.columns.values]
agg = agg.reset_index()
print("  Aggregated rainfall features:", agg.shape[1] - 1, "columns per asset")

# ------------------------------------------------------------------
# 3. MERGE INTO ONE FEATURE TABLE
# ------------------------------------------------------------------
print("\n[3/5] Merging asset + rainfall features ...")
merged = asset_df.merge(agg, on="asset_id", how="left")
print("  Merged shape:", merged.shape)

# Identify feature columns for modeling (exclude identifiers/provenance/null_reason)
exclude = {
    "asset_id", "sector", "name", "coord_role", "verification_status",
    "verified_lat", "verified_lon", "rd_corridor_name",
    "rd_prov_source_name", "rd_prov_source_url", "rd_prov_licence",
    "rd_prov_retrieval_date", "rd_prov_is_observed", "rd_prov_is_imputed",
    "mt_prov_source_name", "mt_prov_source_url", "mt_prov_licence",
    "mt_prov_retrieval_date", "mt_prov_is_observed", "mt_prov_is_imputed",
    "cov_up_district_panel_status", "cov_nasa_power_point_status",
    "cov_osm_highways_status", "cov_portwatch_ports_status",
    "cov_hdx_sea_ports_status", "cov_natural_earth_coastline_status",
    "cov_up_district_panel_reason", "cov_nasa_power_point_reason",
    "cov_osm_highways_reason", "cov_portwatch_ports_reason",
    "cov_hdx_sea_ports_reason", "cov_natural_earth_coastline_reason",
    "null_reason",
}

feature_cols = [c for c in merged.columns if c not in exclude]
print("  Modeling features:", len(feature_cols))
for c in feature_cols:
    print("    ", c)

# ------------------------------------------------------------------
# 4. TARGET VARIABLE (SYNTHETIC / DEMONSTRATIVE)
# ------------------------------------------------------------------
print("\n[4/5] Target variable ...")
print("  NOTE: The integration assets have NO verified cost target.")
print("        AUDIT.md confirms all cost columns in the Master file are")
print("        formula-derived / quarantined and not reliable ground truth.")
print("        A synthetic target is built below purely to demonstrate the")
print("        methodology — it is NOT a real cost prediction.")

rng = np.random.default_rng(42)

# Synthetic cost deviation driven by real features + noise
# Logic: farther from highway/port + higher rainfall anomaly -> higher deviation
merged["synthetic_cost_dev_pct"] = 0.0

if "rd_dist_nh_m" in merged.columns:
    merged["synthetic_cost_dev_pct"] += (
        np.log1p(merged["rd_dist_nh_m"].fillna(0)) / 20.0
    )
if "mt_dist_major_port_m" in merged.columns:
    merged["synthetic_cost_dev_pct"] += (
        np.log1p(merged["mt_dist_major_port_m"].fillna(0)) / 50.0
    )
if "pan_wx_anomaly_z_mean" in merged.columns:
    merged["synthetic_cost_dev_pct"] += (
        merged["pan_wx_anomaly_z_mean"].fillna(0).abs() * 2.0
    )
if "up_wx_anomaly_z_mean" in merged.columns:
    merged["synthetic_cost_dev_pct"] += (
        merged["up_wx_anomaly_z_mean"].fillna(0).abs() * 2.0
    )

# Add noise
merged["synthetic_cost_dev_pct"] += rng.normal(0, 5.0, size=len(merged))
merged["synthetic_cost_dev_pct"] = merged["synthetic_cost_dev_pct"].clip(-50, 200)

model_df = merged.dropna(subset=["synthetic_cost_dev_pct"]).copy()
print("  Rows with target:", len(model_df))
print("  Target stats: mean=%.2f  std=%.2f  min=%.2f  max=%.2f" % (
    model_df["synthetic_cost_dev_pct"].mean(),
    model_df["synthetic_cost_dev_pct"].std(),
    model_df["synthetic_cost_dev_pct"].min(),
    model_df["synthetic_cost_dev_pct"].max(),
))

# ------------------------------------------------------------------
# 5. TIME-SERIES (ARIMA/ARMA) — SKIPPED (no time series per asset)
# ------------------------------------------------------------------
print("\n[5/5] Time-series analysis (ARIMA/ARMA) ...")
print("  SKIPPED: no per-asset time series of cost outcomes available.")
print("           The month matrix has rainfall time series but no cost target.")
arima_result = {"skipped": True, "reason": "no cost time series per asset"}

# ------------------------------------------------------------------
# 6. REGRESSION
# ------------------------------------------------------------------
print("\n[6/5] Regression: synthetic_cost_dev_pct ~ integration features ...")

X = model_df[feature_cols].copy()
y = model_df["synthetic_cost_dev_pct"].copy()

# Fill NaNs for modeling
num_features = [c for c in feature_cols if X[c].dtype in [np.float64, np.int64]]
cat_features = [c for c in feature_cols if c not in num_features]

preprocess = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
        ]), cat_features),
    ],
    remainder="drop",
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
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

reg_summary = {
    "n_train": int(len(y_train)),
    "n_test": int(len(y_test)),
    "r2": float(r2),
    "rmse": float(rmse),
    "mae": float(mae),
    "n_features": int(len(feature_names)),
    "top_features": coef_df.head(15).to_dict("records"),
}

# ------------------------------------------------------------------
# 7. MONTE CARLO
# ------------------------------------------------------------------
print("\n[7/5] Monte Carlo simulation of synthetic cost outcomes ...")

n_sim = 10000
rng2 = np.random.default_rng(123)

# Sample feature profiles from the data
sample_idx = rng2.choice(len(model_df), size=n_sim, replace=True)
sample_X = X.iloc[sample_idx].reset_index(drop=True)

# Predicted cost deviation from regression
sim_pred = pipe.predict(sample_X)

# Add residual noise
resid_std = np.std(y_test - pipe.predict(X_test))
sim_noise = rng2.normal(0, resid_std, size=n_sim)
sim_actual = sim_pred + sim_noise
sim_actual = np.clip(sim_actual, -50, 200)

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

mc = summarize(sim_actual, "monte_carlo")
print("  Simulations: %d projects" % n_sim)
for k, v in mc.items():
    if k == "label":
        continue
    print("    %-18s %s" % (k, v))

mc_summary = {
    "n_sim": n_sim,
    "monte_carlo": mc,
}

# ------------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------------
results = {
    "repo_root": REPO_ROOT,
    "n_assets": int(asset_df.shape[0]),
    "n_month_rows": int(month_df.shape[0]),
    "feature_cols": feature_cols,
    "target": "synthetic_cost_dev_pct (demonstrative only — not real ground truth)",
    "timeseries": {"skipped": True, "reason": "no cost time series per asset"},
    "arima": arima_result,
    "regression": reg_summary,
    "monte_carlo": mc_summary,
}

out_path = os.path.join(OUTDIR, "integration_pipeline_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

merged_out = os.path.join(OUTDIR, "merged_features.csv")
merged.to_csv(merged_out, index=False)

print("\n" + "=" * 70)
print("RESULTS SAVED")
print("  JSON: %s" % out_path)
print("  CSV:  %s" % merged_out)
print("=" * 70)
