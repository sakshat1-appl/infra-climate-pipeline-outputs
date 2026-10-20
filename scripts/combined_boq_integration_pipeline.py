"""
Combined pipeline: integration-layer exposure features + Indian BOQ cost data.

Steps
  1. Load BOQ projects (Kaggle, synthetic Indian BOQ) and the repo integration features.
  2. Assign each integration asset to a state (nearest state centroid) and aggregate
     exposure features to state level.
  3. Merge exposure features onto BOQ projects by state.
  4. Regression  : log(total_project_cost_inr) ~ BOQ features + exposure features
  5. ARMA / ARIMA: quarterly programme cost, order selected by AIC
  6. Monte Carlo : project-level cost draws + forward quarterly forecast paths

All outputs are written to output_dir. The repository is never modified.
"""
import json
import os
import warnings
import zipfile

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BOQ_ZIP = r"D:\consolidated_pipeline\inputs\kaggle_boq.zip"
ASSET_CSV = r"D:\INFRA_ TEST\repo\integration\data_output\feature_matrix_asset.csv"
MONTH_CSV = r"D:\INFRA_ TEST\repo\integration\data_output\feature_matrix_asset_month.csv"
OUTDIR = r"D:\consolidated_pipeline\outputs_combined"

SEED = 42
N_MC = 10_000
HORIZON = 8
os.makedirs(OUTDIR, exist_ok=True)

STATE_CENTROIDS = {
    "Maharashtra": (19.75, 75.71),
    "Gujarat": (22.26, 71.19),
    "Tamil Nadu": (11.13, 78.66),
    "Telangana": (18.11, 79.02),
    "Uttar Pradesh": (26.85, 80.91),
    "Karnataka": (15.32, 75.71),
    "West Bengal": (22.99, 87.85),
    "Rajasthan": (27.02, 74.22),
    "Delhi": (28.70, 77.10),
    "Madhya Pradesh": (23.47, 78.66),
}


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def assign_states(assets):
    names = list(STATE_CENTROIDS)
    clat = np.array([STATE_CENTROIDS[s][0] for s in names])
    clon = np.array([STATE_CENTROIDS[s][1] for s in names])
    out = []
    for lat, lon in zip(assets["verified_lat"], assets["verified_lon"]):
        if pd.isna(lat) or pd.isna(lon):
            out.append(None)
            continue
        d = haversine(lat, lon, clat, clon)
        out.append(names[int(np.argmin(d))])
    return pd.Series(out, index=assets.index, name="state")


def build_state_exposure():
    assets = pd.read_csv(ASSET_CSV)
    months = pd.read_csv(MONTH_CSV)
    assets["state"] = assign_states(assets)

    rain = months.groupby("asset_id").agg(
        wx_anomaly_index_mean=("pan_wx_anomaly_z", "mean"),
        wx_anomaly_index_sd=("pan_wx_anomaly_z", "std"),
        wx_precip_month_mean=("pan_wx_precip_month_mm", "mean"),
        wx_monsoon_mean=("pan_wx_monsoon_JJAS_mm", "mean"),
        wx_rain_days_mean=("pan_wx_rain_days_month", "mean"),
    )
    assets = assets.merge(rain, left_on="asset_id", right_index=True, how="left")

    assets["log_dist_nh"] = np.log1p(assets["rd_dist_nh_m"])
    assets["log_dist_port"] = np.log1p(assets["mt_dist_major_port_m"])
    assets["in_port_polygon"] = assets["mt_in_port_polygon"].fillna(0).astype(float)
    assets["coastal"] = assets["mt_coastal_flag"].fillna(0).astype(float)
    assets["on_corridor"] = assets["rd_on_corridor_flag"].fillna(0).astype(float)

    agg = assets.dropna(subset=["state"]).groupby("state").agg(
        exp_n_assets=("asset_id", "count"),
        exp_on_corridor_share=("on_corridor", "mean"),
        exp_log_dist_nh=("log_dist_nh", "mean"),
        exp_log_dist_port=("log_dist_port", "mean"),
        exp_in_port_polygon_share=("in_port_polygon", "mean"),
        exp_coastal_share=("coastal", "mean"),
        exp_anomaly_index=("wx_anomaly_index_mean", "mean"),
        exp_anomaly_index_sd=("wx_anomaly_index_mean", "std"),
        exp_precip_month=("wx_precip_month_mean", "mean"),
        exp_monsoon=("wx_monsoon_mean", "mean"),
        exp_rain_days=("wx_rain_days_mean", "mean"),
    )
    return agg, assets


def load_boq():
    z = zipfile.ZipFile(BOQ_ZIP)
    proj = pd.read_csv(z.open("boq_projects.csv"))
    items = pd.read_csv(z.open("boq_line_items.csv"))

    proj = proj.dropna(subset=["total_project_cost_inr", "project_start_date"]).copy()
    proj["project_start_date"] = pd.to_datetime(proj["project_start_date"], errors="coerce")
    proj = proj.dropna(subset=["project_start_date"])

    it = items.groupby("project_id").agg(
        n_items=("item_id", "count") if "item_id" in items.columns else ("project_id", "size"),
        avg_rate=("rate_inr", "mean") if "rate_inr" in items.columns else ("project_id", "size"),
        max_rate=("rate_inr", "max") if "rate_inr" in items.columns else ("project_id", "size"),
        n_categories=("category", "nunique") if "category" in items.columns else ("project_id", "size"),
        n_subcategories=("subcategory", "nunique") if "subcategory" in items.columns else ("project_id", "size"),
    )
    proj = proj.merge(it, left_on="project_id", right_index=True, how="left")
    proj["start_year"] = proj["project_start_date"].dt.year
    proj["start_month"] = proj["project_start_date"].dt.month
    proj["start_quarter"] = proj["project_start_date"].dt.quarter
    proj["log_area"] = np.log1p(proj["built_up_area_sqft"])
    proj["log_cost"] = np.log1p(proj["total_project_cost_inr"])
    return proj


def main():
    print("=" * 72)
    print("COMBINED PIPELINE: integration exposure + Indian BOQ cost")
    print("=" * 72)

    print("\n[1/6] Loading data ...")
    proj = load_boq()
    state_exp, assets = build_state_exposure()
    print(f"  BOQ projects with target: {len(proj)}")
    print(f"  Integration assets assigned to a state: {assets['state'].notna().sum()}/{len(assets)}")
    print(f"  States in BOQ: {proj['state'].nunique()}")
    print(f"  States with exposure features: {len(state_exp)}")

    nat = state_exp.mean(numeric_only=True)
    merged = proj.merge(state_exp, left_on="state", right_index=True, how="left")
    exp_cols = [c for c in state_exp.columns]
    merged["exposure_imputed"] = merged[exp_cols].isna().all(axis=1).astype(int)
    for c in exp_cols:
        merged[c] = merged[c].fillna(nat[c])
    print(f"  Projects without direct state exposure (national fallback): {int(merged['exposure_imputed'].sum())}")

    state_exp.reset_index().to_csv(os.path.join(OUTDIR, "state_exposure.csv"), index=False)

    print("\n[2/6] Regression: log(total_project_cost_inr) ~ BOQ + exposure ...")
    num_feats = [
        "log_area", "num_floors", "estimated_duration_months",
        "start_year", "start_month", "start_quarter",
        "n_items", "avg_rate", "max_rate", "n_categories", "n_subcategories",
    ] + exp_cols + ["exposure_imputed"]
    cat_feats = ["building_type", "city", "state"]
    feats = num_feats + cat_feats
    model_df = merged.dropna(subset=["log_cost"]).copy()

    Xtr, Xte, ytr, yte = train_test_split(
        model_df[feats], model_df["log_cost"], test_size=0.25, random_state=SEED
    )
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_feats),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=5))]), cat_feats),
    ])
    reg = Pipeline([("pre", pre), ("m", Ridge(alpha=1.0))])
    reg.fit(Xtr, ytr)
    pred_log = reg.predict(Xte)
    pred = np.expm1(pred_log)
    actual = np.expm1(yte)
    resid = ytr - reg.predict(Xtr)
    resid_sd = float(np.std(resid, ddof=1))

    metrics = {
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "n_features_encoded": int(len(reg.named_steps["pre"].get_feature_names_out())),
        "r2_log": float(r2_score(yte, pred_log)),
        "r2_cost": float(r2_score(actual, pred)),
        "rmse_inr": float(np.sqrt(mean_squared_error(actual, pred))),
        "mae_inr": float(mean_absolute_error(actual, pred)),
        "mape_pct": float(np.mean(np.abs((actual - pred) / actual)) * 100),
        "resid_sd_log": resid_sd,
    }
    print(f"  Train/Test: {metrics['n_train']}/{metrics['n_test']}  features encoded: {metrics['n_features_encoded']}")
    print(f"  R^2 (log) {metrics['r2_log']:.4f} | R^2 (INR) {metrics['r2_cost']:.4f}")
    print(f"  RMSE {metrics['rmse_inr']:,.0f} INR | MAE {metrics['mae_inr']:,.0f} INR | MAPE {metrics['mape_pct']:.2f}%")

    names = reg.named_steps["pre"].get_feature_names_out()
    coefs = pd.Series(reg.named_steps["m"].coef_, index=names)
    top = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(15)
    print("  Top standardized coefficients:")
    for k, v in top.items():
        print(f"    {k:48s} {v:+.4f}")

    print("\n[3/6] Quarterly series (programme cost) ...")
    q = (proj.set_index("project_start_date")["total_project_cost_inr"]
         .resample("QS").sum())
    ts = q[q > 0].copy()
    print(f"  Quarters: {len(ts)}  ({ts.index.min().date()} -> {ts.index.max().date()})")
    adf_lvl = float(adfuller(ts.dropna(), autolag="AIC")[1])
    adf_diff = float(adfuller(ts.diff().dropna(), autolag="AIC")[1])
    print(f"  ADF p (levels) {adf_lvl:.4g} | ADF p (diff) {adf_diff:.4g}")

    print("\n[4/6] ARMA / ARIMA order search ...")
    fits = {}
    for d in (0, 1):
        for p in range(3):
            for qq in range(3):
                if p == 0 and qq == 0:
                    continue
                trends = ("c",) if d == 0 else ("n", "t")
                for tr in trends:
                    try:
                        r = ARIMA(ts, order=(p, d, qq), trend=tr).fit()
                        fits[(p, d, qq, tr)] = r
                    except Exception:
                        continue
    if not fits:
        raise RuntimeError("no ARIMA model converged")
    best_key = min(fits, key=lambda k: fits[k].aic)
    best = fits[best_key]
    best_order = best_key[:3]
    print(f"  Best order: ARIMA{best_key}  AIC={best.aic:.2f}")
    for k in sorted(fits, key=lambda k: fits[k].aic)[:4]:
        print(f"    ARIMA{k}  AIC={fits[k].aic:.2f}")
    arma_orders = {str(k): float(fits[k].aic) for k in sorted(fits, key=lambda k: fits[k].aic)[:6]}

    print(f"\n[5/6] ARIMA forecast ({HORIZON} quarters) ...")
    fc = best.get_forecast(HORIZON)
    mean_fc = np.asarray(fc.predicted_mean)
    ci = np.asarray(fc.conf_int(alpha=0.05))
    try:
        sims = np.asarray(best.simulate(nsimulations=HORIZON, anchor="end",
                                        repetitions=N_MC,
                                        rng=np.random.default_rng(SEED)))
    except Exception as exc:
        print(f"  simulate() failed ({exc}); falling back to analytic draws")
        sims = None
    if sims is not None:
        if sims.ndim == 1:
            sims = sims.reshape(-1, 1)
        if sims.shape[0] != HORIZON and sims.shape[1] == HORIZON:
            sims = sims.T
        sims = sims[:HORIZON]
    sim_p = {}
    for i in range(HORIZON):
        row = sims[i] if sims is not None else np.random.default_rng(SEED + i).normal(
            mean_fc[i], (ci[i, 1] - ci[i, 0]) / 3.92, N_MC)
        sim_p[i] = (float(np.percentile(row, 5)), float(np.percentile(row, 50)),
                    float(np.percentile(row, 95)))

    future_idx = pd.date_range(ts.index.max() + pd.offsets.QuarterBegin(startingMonth=1),
                               periods=HORIZON, freq="QS")
    fcast = pd.DataFrame({
        "quarter": [f"{d.year}-Q{d.quarter}" for d in future_idx],
        "forecast_inr": mean_fc,
        "ci95_lower_inr": ci[:, 0],
        "ci95_upper_inr": ci[:, 1],
        "mc_p5_inr": [sim_p[i][0] for i in range(HORIZON)],
        "mc_p50_inr": [sim_p[i][1] for i in range(HORIZON)],
        "mc_p95_inr": [sim_p[i][2] for i in range(HORIZON)],
    })
    print(fcast.to_string(index=False, float_format=lambda x: f"{x:,.0f}"))

    print(f"\n[6/6] Monte Carlo project costs ({N_MC:,} draws) ...")
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(model_df), size=N_MC)
    draws_X = model_df[feats].iloc[idx].reset_index(drop=True)
    log_draws = reg.predict(draws_X) + rng.normal(0, resid_sd, N_MC)
    cost_draws = np.clip(np.expm1(log_draws), 0, None)
    area_draws = model_df["built_up_area_sqft"].iloc[idx].to_numpy()
    cps = np.divide(cost_draws, area_draws, out=np.zeros_like(cost_draws), where=area_draws > 0)
    cps = cps[np.isfinite(cps)]

    def pct(a):
        return {p: float(np.percentile(a, q)) for p, q in
                [("p5", 5), ("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95)]}

    cost_stats = {"n": int(len(cost_draws)), "mean": float(cost_draws.mean()),
                  "std": float(cost_draws.std()), **pct(cost_draws),
                  "min": float(cost_draws.min()), "max": float(cost_draws.max())}
    cps_stats = {"n": int(len(cps)), "mean": float(cps.mean()), "std": float(cps.std()),
                 **pct(cps), "min": float(cps.min()), "max": float(cps.max())}

    print("  Total project cost (INR):")
    for k in ("mean", "p5", "p25", "p50", "p75", "p95"):
        print(f"    {k:5s} {cost_stats[k]:,.0f}")
    print("  Cost per sqft (INR):")
    for k in ("p25", "p50", "p75", "p95"):
        print(f"    {k:5s} {cps_stats[k]:,.0f}")

    pd.DataFrame({
        "predicted_total_cost_inr": cost_draws,
        "built_up_area_sqft": area_draws,
        "predicted_cost_per_sqft_inr": np.divide(cost_draws, area_draws,
                                                 out=np.zeros_like(cost_draws), where=area_draws > 0),
    }).to_csv(os.path.join(OUTDIR, "monte_carlo_projects.csv"), index=False)

    print("\n[6b] Forward cost projection (all projects re-dated to 2026-Q1) ...")
    future_X = model_df[feats].copy()
    future_X["start_year"] = 2026
    future_X["start_month"] = 1
    future_X["start_quarter"] = 1
    mu_future = reg.predict(future_X)
    rr = np.random.default_rng(SEED + 1)
    draws_future = np.clip(np.expm1(mu_future[:, None] + rr.normal(0.0, resid_sd, (len(mu_future), 500))), 0, None)
    proj_rows = pd.DataFrame({
        "state": model_df["state"].to_numpy(),
        "building_type": model_df["building_type"].to_numpy(),
        "expected_cost_inr": np.expm1(mu_future),
        "p5_inr": np.percentile(draws_future, 5, axis=1),
        "p50_inr": np.percentile(draws_future, 50, axis=1),
        "p95_inr": np.percentile(draws_future, 95, axis=1),
    })
    proj_rows.to_csv(os.path.join(OUTDIR, "future_projection_by_project.csv"), index=False)
    by_state = proj_rows.groupby("state").agg(
        n_projects=("expected_cost_inr", "size"),
        expected_total_cost_inr=("expected_cost_inr", "sum"),
        expected_project_cost_inr=("expected_cost_inr", "mean"),
        p5_total_inr=("p5_inr", "sum"),
        p95_total_inr=("p95_inr", "sum"),
    ).reset_index().sort_values("expected_total_cost_inr", ascending=False)
    by_state.to_csv(os.path.join(OUTDIR, "future_projection_by_state_2026.csv"), index=False)
    print(by_state.to_string(index=False, float_format=lambda x: f"{x:,.0f}"))
    print(f"  Portfolio expected total (2026): INR {proj_rows['expected_cost_inr'].sum():,.0f}")
    print(f"  Portfolio p5-p95 total          : INR {proj_rows['p5_inr'].sum():,.0f} - {proj_rows['p95_inr'].sum():,.0f}")

    fcast.to_csv(os.path.join(OUTDIR, "forecast_quarterly.csv"), index=False)
    model_df[feats + ["log_cost", "total_project_cost_inr"]].to_csv(
        os.path.join(OUTDIR, "modeling_dataset_combined.csv"), index=False)
    ts.rename("total_project_cost_inr").to_csv(os.path.join(OUTDIR, "quarterly_series.csv"))

    fig, ax = plt.subplots(2, 1, figsize=(11, 8))
    ax[0].plot(ts.index, ts.values / 1e9, "o-", label="observed quarterly cost")
    ax[0].plot(future_idx, mean_fc / 1e9, "s--", color="tab:red", label="ARIMA forecast")
    ax[0].fill_between(future_idx, np.asarray([sim_p[i][0] for i in range(HORIZON)]) / 1e9,
                       np.asarray([sim_p[i][2] for i in range(HORIZON)]) / 1e9,
                       color="tab:red", alpha=0.2, label="MC p5-p95")
    ax[0].set_ylabel("INR billion"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].set_title(f"Quarterly programme cost - ARIMA{best_order} forecast")
    ax[1].hist(cost_draws / 1e6, bins=60, color="tab:blue", alpha=0.8)
    ax[1].axvline(cost_stats["p50"], color="k", ls="--", label=f"median {cost_stats['p50']/1e6:.0f}M")
    ax[1].set_xlabel("Total project cost (INR million)"); ax[1].set_ylabel("draws")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    ax[1].set_title(f"Monte Carlo project cost ({N_MC:,} draws)")
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "combined_forecast.png"), dpi=130)
    plt.close(fig)

    results = {
        "dataset": {
            "boq": "kaiwalyaatulraut/construction-and-electrical-boq-dataset (synthetic, CC0)",
            "integration": "integration/data_output/feature_matrix_asset[_month].csv (100 assets)",
            "join": "BOQ state -> nearest-centroid state of integration assets -> state-mean exposure",
        },
        "caveats": [
            "BOQ costs are synthetic, not real projects.",
            "State assignment is nearest-centroid; assets spanning states are approximated.",
            "Exposure features are state-level averages, not asset-specific.",
        ],
        "n_projects": int(len(proj)),
        "n_modeling_rows": int(len(model_df)),
        "exposure_features": exp_cols,
        "regression": metrics,
        "top_coefficients": {k: float(v) for k, v in top.items()},
        "timeseries": {"n_quarters": int(len(ts)), "adf_p_levels": adf_lvl, "adf_p_diff": adf_diff},
        "arima": {"best_order": list(best_order), "aic": float(best.aic),
                  "top_models_aic": arma_orders,
                  "params": {k: float(v) for k, v in best.params.items()}},
        "forecast": fcast.to_dict(orient="records"),
        "monte_carlo": {"n_sim": N_MC, "total_cost_inr": cost_stats,
                        "cost_per_sqft_inr": cps_stats},
        "future_projection_2026": {
            "basis": "all modelling rows re-dated to 2026-Q1, regression + residual noise",
            "portfolio_expected_total_inr": float(proj_rows["expected_cost_inr"].sum()),
            "portfolio_p5_total_inr": float(proj_rows["p5_inr"].sum()),
            "portfolio_p95_total_inr": float(proj_rows["p95_inr"].sum()),
            "by_state": by_state.to_dict(orient="records"),
        },
    }
    with open(os.path.join(OUTDIR, "combined_pipeline_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 72)
    print("SAVED TO", OUTDIR)
    for fn in sorted(os.listdir(OUTDIR)):
        print("  ", fn)
    print("=" * 72)


if __name__ == "__main__":
    main()
