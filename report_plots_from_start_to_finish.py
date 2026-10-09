"""
EV Depot XGBoost Forecasting — Report Plot Generator
====================================================

Purpose
-------
Generate a reproducible set of report-quality figures for the EV depot demand
forecasting project using the real UK Power Networks Optimise Prime depot CSVs.

Core modelling choices match the Streamlit project:
- 15-minute demand data
- chronological train / validation / untouched held-out test split
- causal past-only features
- previous-day same-time baseline (Lag_96)
- genuine xgboost.XGBRegressor with 340 trees
- no synthetic demand generation
- planning threshold used for warning/breach analysis

Example
-------
python report_plots_from_start_to_finish.py --depot Bexleyheath

Optional
--------
python report_plots_from_start_to_finish.py \
    --depot "Mount Pleasant" \
    --data-dir . \
    --output-dir report_figures

Outputs
-------
01_data_split_timeline.png
02_heldout_actual_vs_predicted.png
03_actual_vs_predicted_scatter.png
04_residual_time_series.png
05_residual_distribution.png
06_xgboost_vs_baseline_metrics.png
07_feature_importance.png
08_recursive_24h_forecast.png
metrics_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor


# ---------------------------------------------------------------------
# PROJECT CONFIGURATION
# ---------------------------------------------------------------------

SAMPLING_MINUTES = 15
STEPS_PER_HOUR = 4
STEPS_PER_DAY = 96
STEPS_PER_WEEK = 672

DATA_START = pd.Timestamp("2021-07-01")
TRAIN_END = pd.Timestamp("2022-04-01")
VALID_END = pd.Timestamp("2022-05-10")
TEST_END = pd.Timestamp("2022-07-26")

WARNING_ALPHA = 0.80

DEPOT_CAPACITY_KW = {
    "Bexleyheath": 17.0,
    "Dartford": 45.0,
    "Islington": 13.0,
    "Mount Pleasant": 260.0,
    "Orpington": 14.0,
    "Premier Park": 70.0,
    "Whitechapel": 55.0,
    "Camden": 21.0,
    "Victoria": 29.0,
}

DEPOT_FILES = {
    "Bexleyheath": "UKPN_OptimisePrime_Bexleyheath.csv",
    "Camden": "UKPN_OptimisePrime_Camden.csv",
    "Dartford": "UKPN_OptimisePrime_Dartford.csv",
    "Islington": "UKPN_OptimisePrime_Islington.csv",
    "Mount Pleasant": "UKPN_OptimisePrime_Mount_Pleasant.csv",
    "Orpington": "UKPN_OptimisePrime_Orpington.csv",
    "Premier Park": "UKPN_OptimisePrime_Premier_Park.csv",
    "Victoria": "UKPN_OptimisePrime_Victoria.csv",
    "Whitechapel": "UKPN_OptimisePrime_Whitechapel.csv",
}

FEATURE_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "weekend_flag",
    "month",
    "Lag_1",
    "Lag_4",
    "Lag_96",
    "Lag_672",
    "Rolling_mean_4",
    "Rolling_mean_24",
    "Rolling_mean_96",
    "Rolling_std_96",
]


# ---------------------------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------------------------

def detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Detect timestamp and demand columns from a depot CSV."""
    columns = [str(c).strip() for c in df.columns]

    time_candidates = [
        c for c in columns
        if c == "Timestamp" or "timestamp" in c.lower()
        or "datetime" in c.lower()
        or "date" in c.lower()
        or "time" in c.lower()
    ]

    demand_candidates = [
        c for c in columns
        if c == "Demand_kW"
        or "demand" in c.lower()
        or "load" in c.lower()
        or "kw" in c.lower()
    ]

    if not time_candidates:
        raise ValueError(
            "Could not detect a timestamp/date/time column."
        )
    if not demand_candidates:
        raise ValueError(
            "Could not detect a demand/load/kW column."
        )

    return time_candidates[0], demand_candidates[0]


def load_and_standardise(csv_path: Path) -> pd.DataFrame:
    """Load one depot CSV and standardise it to a 15-minute demand series."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Depot CSV not found: {csv_path}"
        )

    raw = pd.read_csv(csv_path)
    raw.columns = [str(c).strip() for c in raw.columns]

    time_col, demand_col = detect_columns(raw)

    df = raw[[time_col, demand_col]].copy()
    df.columns = ["Timestamp", "Demand_kW"]

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce",
        dayfirst=False,
    )
    df["Demand_kW"] = pd.to_numeric(
        df["Demand_kW"],
        errors="coerce",
    )

    df = df.dropna(subset=["Timestamp", "Demand_kW"])
    df["Demand_kW"] = df["Demand_kW"].clip(lower=0)

    df = (
        df.sort_values("Timestamp")
        .drop_duplicates(subset=["Timestamp"], keep="last")
        .set_index("Timestamp")
    )

    # Put the series on the model's 15-minute grid.
    df = df.resample("15min").mean()

    # Short gaps are interpolated causally enough for preparation of a regular
    # grid; remaining edge gaps are forward/back filled.
    df["Demand_kW"] = (
        df["Demand_kW"]
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
    )

    return df


# ---------------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build causal features only from timestamps and past demand observations.

    Rolling features are shifted by one interval so the current target is not
    included in its own predictors.
    """
    feat = df.copy()

    hour = feat.index.hour + feat.index.minute / 60.0

    feat["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    feat["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    feat["day_of_week"] = feat.index.dayofweek
    feat["weekend_flag"] = (feat.index.dayofweek >= 5).astype(int)
    feat["month"] = feat.index.month

    feat["Lag_1"] = feat["Demand_kW"].shift(1)
    feat["Lag_4"] = feat["Demand_kW"].shift(4)
    feat["Lag_96"] = feat["Demand_kW"].shift(STEPS_PER_DAY)
    feat["Lag_672"] = feat["Demand_kW"].shift(STEPS_PER_WEEK)

    past = feat["Demand_kW"].shift(1)
    feat["Rolling_mean_4"] = past.rolling(4).mean()
    feat["Rolling_mean_24"] = past.rolling(24).mean()
    feat["Rolling_mean_96"] = past.rolling(96).mean()
    feat["Rolling_std_96"] = past.rolling(96).std()

    feat = feat.dropna(subset=FEATURE_COLUMNS + ["Demand_kW"])

    return feat


# ---------------------------------------------------------------------
# MODEL + METRICS
# ---------------------------------------------------------------------

def make_model() -> XGBRegressor:
    """Return the project XGBoost configuration."""
    return XGBRegressor(
        n_estimators=340,
        max_depth=7,
        learning_rate=0.03,
        min_child_weight=10,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )


def metric_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    capacity_kw: float,
) -> dict:
    """Regression + operational threshold metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    actual_breach = y_true >= capacity_kw
    pred_breach = y_pred >= capacity_kw

    tp = int(np.sum(actual_breach & pred_breach))
    fp = int(np.sum((~actual_breach) & pred_breach))
    fn = int(np.sum(actual_breach & (~pred_breach)))

    recall = tp / (tp + fn) if (tp + fn) else np.nan
    far = fp / (tp + fp) if (tp + fp) else np.nan

    return {
        "MAE_kW": float(mae),
        "RMSE_kW": float(rmse),
        "R2": float(r2),
        "Breach_Recall": float(recall) if np.isfinite(recall) else np.nan,
        "False_Alarm_Ratio": float(far) if np.isfinite(far) else np.nan,
        "TP": tp,
        "FP": fp,
        "FN": fn,
    }


# ---------------------------------------------------------------------
# RECURSIVE FUTURE FORECAST
# ---------------------------------------------------------------------

def one_feature_row(
    timestamp: pd.Timestamp,
    history: list[float],
) -> pd.DataFrame:
    """Construct one future feature row using only known/predicted past values."""
    if len(history) < STEPS_PER_WEEK:
        raise ValueError(
            "At least one week of history is required for recursive forecasting."
        )

    hour = timestamp.hour + timestamp.minute / 60.0
    history_arr = np.asarray(history, dtype=float)

    row = {
        "hour_sin": np.sin(2 * np.pi * hour / 24.0),
        "hour_cos": np.cos(2 * np.pi * hour / 24.0),
        "day_of_week": timestamp.dayofweek,
        "weekend_flag": int(timestamp.dayofweek >= 5),
        "month": timestamp.month,
        "Lag_1": history_arr[-1],
        "Lag_4": history_arr[-4],
        "Lag_96": history_arr[-STEPS_PER_DAY],
        "Lag_672": history_arr[-STEPS_PER_WEEK],
        "Rolling_mean_4": history_arr[-4:].mean(),
        "Rolling_mean_24": history_arr[-24:].mean(),
        "Rolling_mean_96": history_arr[-96:].mean(),
        "Rolling_std_96": history_arr[-96:].std(ddof=1),
    }

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def recursive_forecast_24h(
    model: XGBRegressor,
    history_series: pd.Series,
) -> pd.Series:
    """Generate a genuine 24-hour recursive multi-step forecast."""
    history = history_series.astype(float).tolist()
    last_time = history_series.index.max()

    future_times = pd.date_range(
        last_time + pd.Timedelta(minutes=SAMPLING_MINUTES),
        periods=STEPS_PER_DAY,
        freq="15min",
    )

    preds = []

    for ts in future_times:
        X_next = one_feature_row(ts, history)
        pred = float(model.predict(X_next)[0])
        pred = max(0.0, pred)

        preds.append(pred)
        history.append(pred)

    return pd.Series(preds, index=future_times, name="Forecast_kW")


# ---------------------------------------------------------------------
# PLOT HELPERS
# ---------------------------------------------------------------------

def save_figure(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_data_split(
    feat: pd.DataFrame,
    output_dir: Path,
):
    fig, ax = plt.subplots(figsize=(12, 2.6))

    ax.axvspan(
        feat.index.min(),
        TRAIN_END,
        alpha=0.20,
        label="Training",
    )
    ax.axvspan(
        TRAIN_END,
        VALID_END,
        alpha=0.20,
        label="Validation",
    )
    ax.axvspan(
        VALID_END,
        feat.index.max(),
        alpha=0.20,
        label="Held-out test",
    )

    ax.set_yticks([])
    ax.set_xlabel("Date")
    ax.set_title("Chronological Data Split")
    ax.legend(ncol=3, loc="upper center")

    save_figure(
        fig,
        output_dir / "01_data_split_timeline.png",
    )


def plot_actual_vs_predicted(
    y_test: pd.Series,
    pred: np.ndarray,
    baseline: np.ndarray,
    capacity_kw: float,
    output_dir: Path,
):
    warning_kw = WARNING_ALPHA * capacity_kw

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(
        y_test.index,
        y_test.values,
        label="Actual Demand",
        linewidth=1.2,
    )
    ax.plot(
        y_test.index,
        pred,
        label="XGBoost Prediction",
        linewidth=1.0,
    )
    ax.plot(
        y_test.index,
        baseline,
        label="Previous-Day Baseline",
        linewidth=0.8,
        alpha=0.75,
    )
    ax.axhline(
        warning_kw,
        linestyle="--",
        linewidth=1.1,
        label=f"Warning Threshold ({warning_kw:.1f} kW)",
    )
    ax.axhline(
        capacity_kw,
        linestyle="--",
        linewidth=1.1,
        label=f"Planning Capacity ({capacity_kw:.1f} kW)",
    )

    ax.set_title("Held-Out Test: Actual vs Predicted Demand")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Demand (kW)")
    ax.legend(ncol=2)
    ax.grid(alpha=0.20)

    save_figure(
        fig,
        output_dir / "02_heldout_actual_vs_predicted.png",
    )


def plot_scatter(
    y_test: pd.Series,
    pred: np.ndarray,
    output_dir: Path,
):
    actual = y_test.to_numpy(dtype=float)
    pred = np.asarray(pred, dtype=float)

    upper = max(actual.max(), pred.max()) * 1.05

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.scatter(actual, pred, s=9, alpha=0.28)
    ax.plot(
        [0, upper],
        [0, upper],
        linestyle="--",
        linewidth=1.4,
        label="Ideal: Predicted = Actual",
    )

    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Actual vs Predicted Demand")
    ax.set_xlabel("Actual Demand (kW)")
    ax.set_ylabel("Predicted Demand (kW)")
    ax.legend()
    ax.grid(alpha=0.20)

    save_figure(
        fig,
        output_dir / "03_actual_vs_predicted_scatter.png",
    )


def plot_residual_time_series(
    y_test: pd.Series,
    pred: np.ndarray,
    output_dir: Path,
):
    residual = y_test.to_numpy(dtype=float) - np.asarray(pred, dtype=float)

    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(y_test.index, residual, linewidth=0.9)
    ax.axhline(0, linestyle="--", linewidth=1.1)

    ax.set_title("Held-Out Residuals Over Time")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Residual (Actual − Predicted) kW")
    ax.grid(alpha=0.20)

    save_figure(
        fig,
        output_dir / "04_residual_time_series.png",
    )


def plot_residual_distribution(
    y_test: pd.Series,
    pred: np.ndarray,
    output_dir: Path,
):
    residual = y_test.to_numpy(dtype=float) - np.asarray(pred, dtype=float)
    mean_r = residual.mean()
    median_r = np.median(residual)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.hist(residual, bins=60, alpha=0.80, label="Residuals")
    ax.axvline(0, linestyle="--", linewidth=1.2, label="Zero error")
    ax.axvline(
        mean_r,
        linestyle=":",
        linewidth=1.2,
        label=f"Mean: {mean_r:.2f} kW",
    )
    ax.axvline(
        median_r,
        linestyle="-.",
        linewidth=1.2,
        label=f"Median: {median_r:.2f} kW",
    )

    ax.set_title("Held-Out Residual Distribution")
    ax.set_xlabel("Residual (Actual − Predicted) kW")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.18)

    save_figure(
        fig,
        output_dir / "05_residual_distribution.png",
    )


def plot_model_vs_baseline(
    xgb_metrics: dict,
    baseline_metrics: dict,
    output_dir: Path,
):
    labels = ["MAE", "RMSE"]
    xgb_vals = [
        xgb_metrics["MAE_kW"],
        xgb_metrics["RMSE_kW"],
    ]
    base_vals = [
        baseline_metrics["MAE_kW"],
        baseline_metrics["RMSE_kW"],
    ]

    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(
        x - width / 2,
        base_vals,
        width,
        label="Previous-Day Baseline",
    )
    ax.bar(
        x + width / 2,
        xgb_vals,
        width,
        label="XGBoost",
    )

    for i, value in enumerate(base_vals):
        ax.text(
            i - width / 2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    for i, value in enumerate(xgb_vals):
        ax.text(
            i + width / 2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Error (kW) — lower is better")
    ax.set_title("XGBoost vs Previous-Day Baseline")
    ax.legend()
    ax.grid(axis="y", alpha=0.20)

    save_figure(
        fig,
        output_dir / "06_xgboost_vs_baseline_metrics.png",
    )


def plot_feature_importance(
    model: XGBRegressor,
    output_dir: Path,
):
    importance = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLUMNS,
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(
        importance.index,
        importance.values,
    )

    ax.set_title("XGBoost Feature Importance")
    ax.set_xlabel("Relative importance")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.20)

    save_figure(
        fig,
        output_dir / "07_feature_importance.png",
    )


def plot_recursive_forecast(
    history: pd.Series,
    forecast: pd.Series,
    capacity_kw: float,
    output_dir: Path,
):
    # Last two days of observed data, followed by the 24-hour recursive forecast.
    recent = history.iloc[-2 * STEPS_PER_DAY:]
    warning_kw = WARNING_ALPHA * capacity_kw

    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.plot(
        recent.index,
        recent.values,
        label="Observed Demand",
        linewidth=1.3,
    )
    ax.plot(
        forecast.index,
        forecast.values,
        linestyle="--",
        linewidth=1.5,
        label="24-Hour XGBoost Forecast",
    )
    ax.axhline(
        warning_kw,
        linestyle="--",
        linewidth=1.1,
        label=f"Warning Threshold ({warning_kw:.1f} kW)",
    )
    ax.axhline(
        capacity_kw,
        linestyle="--",
        linewidth=1.1,
        label=f"Planning Capacity ({capacity_kw:.1f} kW)",
    )
    ax.axvline(
        forecast.index.min(),
        linestyle=":",
        linewidth=1.0,
        label="Forecast Start",
    )

    ax.set_title("Observed Demand and 24-Hour Recursive Forecast")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Demand (kW)")
    ax.legend(ncol=2)
    ax.grid(alpha=0.20)

    save_figure(
        fig,
        output_dir / "08_recursive_24h_forecast.png",
    )


# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------

def run_report(
    depot: str,
    data_dir: Path,
    output_dir: Path,
):
    if depot not in DEPOT_FILES:
        raise ValueError(
            f"Unknown depot: {depot}. "
            f"Choose one of: {', '.join(DEPOT_FILES)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / DEPOT_FILES[depot]
    capacity_kw = DEPOT_CAPACITY_KW[depot]

    print(f"Loading {depot}: {csv_path}")
    df = load_and_standardise(csv_path)

    print("Building causal features...")
    feat = build_features(df)

    train_df = feat[
        (feat.index >= DATA_START)
        & (feat.index < TRAIN_END)
    ].copy()

    valid_df = feat[
        (feat.index >= TRAIN_END)
        & (feat.index < VALID_END)
    ].copy()

    test_df = feat[
        (feat.index >= VALID_END)
        & (feat.index < TEST_END)
    ].copy()

    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError(
            "The depot data does not cover the fixed project split periods."
        )

    # Validation was used during development. Final fitting uses training +
    # validation only; held-out test remains untouched until evaluation.
    development_df = pd.concat(
        [train_df, valid_df],
        axis=0,
    )

    X_development = development_df[FEATURE_COLUMNS]
    y_development = development_df["Demand_kW"]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["Demand_kW"]

    print("Training XGBoost (340 trees)...")
    model = make_model()
    model.fit(
        X_development,
        y_development,
    )

    print("Evaluating untouched held-out test rows...")
    pred = np.clip(
        model.predict(X_test),
        0,
        None,
    )

    baseline = test_df["Lag_96"].to_numpy(dtype=float)

    xgb_metrics = metric_dict(
        y_test.to_numpy(dtype=float),
        pred,
        capacity_kw,
    )
    baseline_metrics = metric_dict(
        y_test.to_numpy(dtype=float),
        baseline,
        capacity_kw,
    )

    print("Generating genuine recursive 24-hour forecast...")
    forecast = recursive_forecast_24h(
        model,
        df["Demand_kW"],
    )

    print("Saving report figures...")
    plot_data_split(feat, output_dir)
    plot_actual_vs_predicted(
        y_test,
        pred,
        baseline,
        capacity_kw,
        output_dir,
    )
    plot_scatter(
        y_test,
        pred,
        output_dir,
    )
    plot_residual_time_series(
        y_test,
        pred,
        output_dir,
    )
    plot_residual_distribution(
        y_test,
        pred,
        output_dir,
    )
    plot_model_vs_baseline(
        xgb_metrics,
        baseline_metrics,
        output_dir,
    )
    plot_feature_importance(
        model,
        output_dir,
    )
    plot_recursive_forecast(
        df["Demand_kW"],
        forecast,
        capacity_kw,
        output_dir,
    )

    summary = pd.DataFrame(
        [
            {
                "Depot": depot,
                "Model": "XGBoost",
                "Trees": 340,
                "Planning_Capacity_kW": capacity_kw,
                "Warning_Threshold_kW": WARNING_ALPHA * capacity_kw,
                "Train_Start": train_df.index.min(),
                "Train_End": train_df.index.max(),
                "Validation_Start": valid_df.index.min(),
                "Validation_End": valid_df.index.max(),
                "Test_Start": test_df.index.min(),
                "Test_End": test_df.index.max(),
                "Test_Rows": len(test_df),
                "XGB_MAE_kW": xgb_metrics["MAE_kW"],
                "XGB_RMSE_kW": xgb_metrics["RMSE_kW"],
                "XGB_R2": xgb_metrics["R2"],
                "XGB_Breach_Recall": xgb_metrics["Breach_Recall"],
                "XGB_False_Alarm_Ratio": xgb_metrics["False_Alarm_Ratio"],
                "Baseline_MAE_kW": baseline_metrics["MAE_kW"],
                "Baseline_RMSE_kW": baseline_metrics["RMSE_kW"],
                "Baseline_Breach_Recall": baseline_metrics["Breach_Recall"],
                "Baseline_False_Alarm_Ratio": baseline_metrics["False_Alarm_Ratio"],
                "Forecast_Peak_kW": float(forecast.max()),
                "Forecast_Start": forecast.index.min(),
                "Forecast_End": forecast.index.max(),
                "Synthetic_Data_Used": "No",
            }
        ]
    )

    summary.to_csv(
        output_dir / "metrics_summary.csv",
        index=False,
    )

    print()
    print("Complete.")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"XGBoost MAE:  {xgb_metrics['MAE_kW']:.3f} kW")
    print(f"XGBoost RMSE: {xgb_metrics['RMSE_kW']:.3f} kW")
    print(f"XGBoost R²:   {xgb_metrics['R2']:.3f}")
    print(f"Baseline MAE: {baseline_metrics['MAE_kW']:.3f} kW")
    print(f"Forecast peak: {forecast.max():.2f} kW")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate report plots for one EV depot."
    )
    parser.add_argument(
        "--depot",
        default="Bexleyheath",
        choices=list(DEPOT_FILES.keys()),
        help="Depot to analyse.",
    )
    parser.add_argument(
        "--data-dir",
        default=".",
        help="Directory containing the nine UKPN depot CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="report_figures",
        help="Directory where figures and metrics are saved.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_report(
        depot=args.depot,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
    )
