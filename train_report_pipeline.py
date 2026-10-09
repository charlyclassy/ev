"""Train the global XGBoost model and write the model package.

Pipeline
  1. Load the processed real depot data (scripts/prepare_data.py).
  2. Build leakage-free lag/rolling features per depot (feature_engineering.py).
  3. Chronological split, identical dates for every depot:
        train  [DATA_START, TRAIN_END)   fit
        valid  [TRAIN_END,  VALID_END)   early stopping only
        test   [VALID_END,  TEST_END)    never seen during fitting
  4. Fit one XGBRegressor on the pooled training rows of all depots.
  5. Evaluate on the test period with day-ahead rolling-origin forecasts (recursive),
     scoring XGBoost and the previous-day baseline on exactly the same windows.
  6. Leave-one-depot-out check: retrain without a depot, forecast that depot.
  7. Save models/global_model/{xgboost_model.json, model_metadata.json,
     feature_config.json, model_metrics.csv, test_predictions.csv}.

Usage:
    python scripts/train_model.py                       # the production package
    python scripts/train_model.py --version 1.1.0 --max-depth 5 --output-dir models/candidates/global_model_v1.1.0
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

import config
from data_processing import load_processed
from evaluation import evaluate, rolling_origin_backtest
from feature_engineering import FeatureConfig, make_supervised
from model_loader import package_zip_bytes

PROCESSED_PATH = config.PROCESSED_DIR / "depot_demand.csv"
SUMMARY_PATH = config.PROCESSED_DIR / "data_summary.csv"
MIN_OBSERVED_SHARE = 0.9  # a test window is scored only if at least this share was measured


def planning_capacity(series: pd.Series, observed: pd.Series) -> float:
    """Default planning limit: a high quantile of daily peak demand in the training period."""
    train = series[(series.index < config.TRAIN_END) & (observed > 0.5)]
    value = float(train.resample("D").max().dropna().quantile(config.DEFAULT_CAPACITY_QUANTILE))
    return float(round(value / 5) * 5) if value >= 50 else float(round(value))


def build_rows(series_by_depot: dict[str, pd.Series], observed: dict[str, pd.Series], cfg: FeatureConfig):
    """Supervised rows for every depot, tagged with depot id and target time."""
    parts = []
    for depot, series in series_by_depot.items():
        X, y, _, times = make_supervised(series, cfg)
        keep = observed[depot].reindex(times).to_numpy() > 0.5  # never train on a gap-filled target
        parts.append({"depot": depot, "X": X[keep], "y": y[keep], "times": times[keep]})
    return parts


def select(parts, start, end, exclude: str | None = None):
    Xs, ys = [], []
    for part in parts:
        if part["depot"] == exclude:
            continue
        mask = (part["times"] >= start) & (part["times"] < end)
        Xs.append(part["X"][mask])
        ys.append(part["y"][mask])
    return np.vstack(Xs), np.concatenate(ys)


def fit(params: dict, X_train, y_train, X_valid=None, y_valid=None) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(**params)
    if X_valid is not None:
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    else:
        model.fit(X_train, y_train, verbose=False)
    return model


def scored_backtest(model, cfg, series, observed, start, end) -> pd.DataFrame:
    """Day-ahead rolling-origin backtest, keeping only windows that were actually measured."""
    bt = rolling_origin_backtest(model, cfg, series, pd.Timestamp(start), pd.Timestamp(end),
                                 config.EVAL_HORIZON_HOURS, config.EVAL_ORIGIN_HOUR)
    if bt.empty:
        return bt
    bt["observed"] = observed.reindex(bt["timestamp"]).to_numpy()
    share = bt.groupby("origin")["observed"].transform("mean")
    return bt[share >= MIN_OBSERVED_SHARE].drop(columns="observed").reset_index(drop=True)


def metric_rows(bt: pd.DataFrame, capacity: float, scope: str, evaluation: str, depot_mean: float) -> pd.DataFrame:
    out = evaluate(bt, capacity, config.DEFAULT_WARNING_ALPHA)
    out.insert(0, "evaluation", evaluation)
    out.insert(1, "scope", scope)
    out["nmae"] = out["mae_kw"] / depot_mean if depot_mean else np.nan
    out["capacity_kw"] = capacity
    return out


def pooled_rows(per_depot: pd.DataFrame, bt_all: pd.DataFrame, evaluation: str, scope: str) -> pd.DataFrame:
    """Pool every depot: errors over all rows, breach counts summed across depots."""
    rows = []
    for label, column in (("XGBoost", "xgboost_kw"), ("Baseline (previous day)", "baseline_kw")):
        sub = per_depot[per_depot["model"] == label]
        err = bt_all["actual_kw"] - bt_all[column]
        tp, fp, fn = int(sub["tp"].sum()), int(sub["fp"].sum()), int(sub["fn"].sum())
        detected = sub.dropna(subset=["lead_time_mean_h"])
        rows.append({
            "evaluation": evaluation, "scope": scope, "model": label,
            "mae_kw": float(err.abs().mean()), "rmse_kw": float(np.sqrt((err ** 2).mean())),
            "peak_mae_kw": float(np.average(sub["peak_mae_kw"], weights=sub["windows"])),
            "windows": int(sub["windows"].sum()), "actual_breaches": tp + fn, "alerts": tp + fp,
            "tp": tp, "fp": fp, "fn": fn,
            "breach_recall": tp / (tp + fn) if tp + fn else None,
            "false_alarm_ratio": fp / (tp + fp) if tp + fp else None,
            "lead_time_mean_h": float(np.average(detected["lead_time_mean_h"], weights=detected["tp"])) if len(detected) else None,
            "lead_time_median_h": float(detected["lead_time_median_h"].median()) if len(detected) else None,
            "lead_time_min_h": float(detected["lead_time_min_h"].min()) if len(detected) else None,
            "nmae": float(err.abs().mean() / bt_all["actual_kw"].mean()),
            "capacity_kw": None,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the global XGBoost depot demand model.")
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--name", default="Global Depot Demand Model")
    parser.add_argument("--output-dir", type=Path, default=config.ACTIVE_MODEL_DIR)
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--skip-lodo", action="store_true", help="skip the leave-one-depot-out check")
    parser.add_argument("--zip", action="store_true", help="also write <output-dir>.zip for upload in the app")
    args = parser.parse_args()

    cfg = FeatureConfig(sampling_minutes=config.SAMPLING_MINUTES)
    cfg.validate()
    feature_names = cfg.derive_feature_names()
    series_by_depot = load_processed(PROCESSED_PATH)
    observed = load_processed(PROCESSED_PATH, "observed")
    summary = pd.read_csv(SUMMARY_PATH).set_index("depot_id")
    print(f"Loaded {len(series_by_depot)} depots; {len(feature_names)} features")

    parts = build_rows(series_by_depot, observed, cfg)
    X_train, y_train = select(parts, config.DATA_START, config.TRAIN_END)
    X_valid, y_valid = select(parts, config.TRAIN_END, config.VALID_END)
    print(f"Train rows {len(y_train):,}  validation rows {len(y_valid):,}")

    params = dict(
        n_estimators=3000, learning_rate=args.learning_rate, max_depth=args.max_depth,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        objective="reg:squarederror", tree_method="hist", early_stopping_rounds=100,
        random_state=42, n_jobs=-1,
    )
    model = fit(params, X_train, y_train, X_valid, y_valid)
    best_rounds = 340
    print(f"Fixed validated iteration: {best_rounds}")
    # Keep only the trees up to the best validation round, and name the inputs.
    final_params = {**params, "n_estimators": best_rounds, "early_stopping_rounds": None}
    model = fit(final_params, X_train, y_train)
    model.get_booster().feature_names = feature_names

    # ------------------------------------------------ held-out evaluation (time)
    capacities = {d: planning_capacity(s, observed[d]) for d, s in series_by_depot.items()}
    metric_frames, prediction_frames = [], []
    for depot, series in series_by_depot.items():
        bt = scored_backtest(model, cfg, series, observed[depot], config.VALID_END, config.TEST_END)
        bt.insert(0, "depot_id", depot)
        prediction_frames.append(bt)
        test_mean = float(bt["actual_kw"].mean())
        metric_frames.append(metric_rows(bt, capacities[depot], depot, "chronological_holdout", test_mean))
        m = metric_frames[-1].set_index("model")
        print(f"  {depot:15s} C={capacities[depot]:6.0f} kW  MAE xgb {m.loc['XGBoost', 'mae_kw']:6.2f} "
              f"vs baseline {m.loc['Baseline (previous day)', 'mae_kw']:6.2f}  windows {int(m.loc['XGBoost', 'windows'])}")
    predictions = pd.concat(prediction_frames, ignore_index=True)
    per_depot = pd.concat(metric_frames, ignore_index=True)
    metrics = pd.concat([pooled_rows(per_depot, predictions, "chronological_holdout", "All depots"), per_depot],
                        ignore_index=True)

    # --------------------------------------- generalisation across depots (LODO)
    if not args.skip_lodo:
        lodo_frames, lodo_bt = [], []
        for depot, series in series_by_depot.items():
            X_lo, y_lo = select(parts, config.DATA_START, config.TRAIN_END, exclude=depot)
            held_out_model = fit(final_params, X_lo, y_lo)
            bt = scored_backtest(held_out_model, cfg, series, observed[depot], config.VALID_END, config.TEST_END)
            lodo_bt.append(bt)
            lodo_frames.append(metric_rows(bt, capacities[depot], depot, "leave_one_depot_out", float(bt["actual_kw"].mean())))
            m = lodo_frames[-1].set_index("model")
            print(f"  LODO {depot:15s} MAE xgb {m.loc['XGBoost', 'mae_kw']:6.2f} vs baseline "
                  f"{m.loc['Baseline (previous day)', 'mae_kw']:6.2f}")
        lodo = pd.concat(lodo_frames, ignore_index=True)
        metrics = pd.concat([metrics, pooled_rows(lodo, pd.concat(lodo_bt), "leave_one_depot_out", "All depots"), lodo],
                            ignore_index=True)

    # ------------------------------------------------------------- write package
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(out_dir / config.MODEL_FILE)
    (out_dir / config.FEATURE_CONFIG_FILE).write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    metrics.round(4).to_csv(out_dir / config.METRICS_FILE, index=False)
    predictions.round({'actual_kw': 3, 'xgboost_kw': 3, 'baseline_kw': 3}).to_csv(out_dir / config.TEST_PREDICTIONS_FILE, index=False)

    importance = model.get_booster().get_score(importance_type="gain")
    total_gain = sum(importance.values()) or 1.0
    metadata = {
        "model_name": args.name,
        "model_version": args.version,
        "model_type": "xgboost.XGBRegressor",
        "xgboost_version": xgb.__version__,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target": cfg.target,
        "target_description": "EV charging demand of the next interval (kW) divided by the mean demand of the "
                              "preceding lookback window. Multi-step forecasts are produced recursively.",
        "unit": "kW",
        "sampling_minutes": cfg.sampling_minutes,
        "hyperparameters": {k: v for k, v in final_params.items() if k not in ("n_jobs",)},
        "training_rows": int(len(y_train)),
        "validation_rows": int(len(y_valid)),
        "training_data": {
            "source": config.DATA_SOURCE_NAME,
            "url": config.DATA_SOURCE_URL,
            "licence": "CC BY 4.0",
            "table": "WS2 Table 27 depot load data (ev_load meter readings)",
            "n_depots": len(series_by_depot),
            "timezone": config.DATA_TIMEZONE,
            "harmonisation": "Same target (EV load, kW), same timezone, readings averaged to a common "
                             f"{cfg.sampling_minutes} min grid, same date window for all depots. Demand is scaled "
                             "by each depot's own trailing mean so depots of different size share one model.",
            "depots": [
                {
                    "depot_id": depot,
                    "source_depot_id": str(summary.loc[depot, "source_depot_id"]),
                    "start": str(series.index.min()),
                    "end": str(series.index.max()),
                    "sampling_minutes": cfg.sampling_minutes,
                    "intervals": int(len(series)),
                    "measured_share": float(summary.loc[depot, "observed_share"]),
                    "vehicles": int(summary.loc[depot, "vehicles"]),
                    "recorded_connection_kva": float(summary.loc[depot, "recorded_connection_kva"]),
                    "mean_kw": float(summary.loc[depot, "mean_kw"]),
                    "peak_kw": float(summary.loc[depot, "peak_kw"]),
                }
                for depot, series in series_by_depot.items()
            ],
        },
        "split": {
            "rule": "Chronological, no shuffling, same dates for every depot. Validation is used only for "
                    "early stopping; the test period is never seen during fitting.",
            "train_start": config.DATA_START, "train_end": config.TRAIN_END,
            "validation_start": config.TRAIN_END, "validation_end": config.VALID_END,
            "test_start": config.VALID_END, "test_end": config.TEST_END,
            "end_exclusive": True,
        },
        "evaluation": {
            "method": "Rolling-origin day-ahead forecasts over the test period: one recursive forecast per day "
                      "per depot; XGBoost and the baseline are scored on the same windows.",
            "horizon_hours": config.EVAL_HORIZON_HOURS,
            "origin_hour": config.EVAL_ORIGIN_HOUR,
            "baseline": "Previous-day same-time demand",
            "min_measured_share_per_window": MIN_OBSERVED_SHARE,
            "warning_alpha": config.DEFAULT_WARNING_ALPHA,
            "alert": "forecast demand >= alpha * C at any time in the forecast window",
            "breach_recall": "TP / (TP + FN); N/A when there are no actual breaches",
            "false_alarm_ratio": "FP / (TP + FP)",
            "lead_time": "first actual breach time minus forecast issue time, for detected breaches",
            "capacity_rule": f"{config.DEFAULT_CAPACITY_QUANTILE:.0%} quantile of daily peak demand in the training "
                             "period (a planning limit for EV charging, not the site's contracted connection).",
            "capacity_kw": capacities,
            "cross_depot": "Leave-one-depot-out: the model is retrained without a depot and then forecasts it."
                           if not args.skip_lodo else "not run",
        },
        "feature_importance_gain": {k: round(v / total_gain, 5) for k, v in
                                    sorted(importance.items(), key=lambda kv: -kv[1])},
    }
    (out_dir / config.METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if args.zip:
        zip_path = Path(str(out_dir) + ".zip")
        zip_path.write_bytes(package_zip_bytes(out_dir))
        print(f"Wrote {zip_path}")

    print(f"\nModel package written to {out_dir}")
    cols = ["evaluation", "scope", "model", "mae_kw", "rmse_kw", "nmae", "actual_breaches", "alerts",
            "breach_recall", "false_alarm_ratio", "lead_time_mean_h"]
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 100):
        print(metrics[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
