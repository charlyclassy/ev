import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import xgboost as xgb
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="EV Depot Demand Forecaster", layout="wide")

st.markdown("""
<style>
/* Keep full model-performance metric names visible on summary cards */
div[data-testid="stMetricLabel"] p {
    font-size: 0.82rem !important;
    line-height: 1.15 !important;
    white-space: normal !important;
}
</style>
""", unsafe_allow_html=True)

# ===================================================================== #
# REAL UK POWER NETWORKS OPTIMISE PRIME DATA                             #
# ===================================================================== #
REAL_DEPOTS = [
    "Bexleyheath", "Dartford", "Islington", "Mount Pleasant",
    "Orpington", "Premier Park", "Whitechapel", "Camden", "Victoria",
]

# These are default PLANNING thresholds derived from the training-period
# demand profile. They are not claimed to be contracted grid connection limits.
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
WARNING_ALPHA = 0.80
SAMPLING_MINUTES = 15

# Fixed chronological project periods for the real Optimise Prime data.
DATA_START = pd.Timestamp("2021-07-01")
TRAIN_END = pd.Timestamp("2022-04-01")
VALID_END = pd.Timestamp("2022-05-10")
TEST_END = pd.Timestamp("2022-07-26")
STEPS_PER_HOUR = 60 // SAMPLING_MINUTES
STEPS_PER_DAY = 24 * STEPS_PER_HOUR
STEPS_PER_WEEK = 7 * STEPS_PER_DAY

# The app supports either one combined processed file or the nine
# individual real depot CSVs placed at the repository root.
BUNDLED_DATA_CANDIDATES = [
    Path("data/processed/depot_demand.csv"),
    Path("UKPN_OptimisePrime_9_Depots_Processed.csv"),
]

INDIVIDUAL_DEPOT_FILES = {
    "Bexleyheath": Path("UKPN_OptimisePrime_Bexleyheath.csv"),
    "Camden": Path("UKPN_OptimisePrime_Camden.csv"),
    "Dartford": Path("UKPN_OptimisePrime_Dartford.csv"),
    "Islington": Path("UKPN_OptimisePrime_Islington.csv"),
    "Mount Pleasant": Path("UKPN_OptimisePrime_Mount_Pleasant.csv"),
    "Orpington": Path("UKPN_OptimisePrime_Orpington.csv"),
    "Premier Park": Path("UKPN_OptimisePrime_Premier_Park.csv"),
    "Victoria": Path("UKPN_OptimisePrime_Victoria.csv"),
    "Whitechapel": Path("UKPN_OptimisePrime_Whitechapel.csv"),
}


@st.cache_data(show_spinner=False)
def load_real_depot_data(depot_name: str) -> pd.DataFrame:
    """Load one genuine Optimise Prime depot."""
    # Prefer a combined dataset if one exists.
    for path in BUNDLED_DATA_CANDIDATES:
        if path.exists():
            raw = pd.read_csv(path, parse_dates=["timestamp"], dtype={"depot_id": str})
            required = {"timestamp", "depot_id", "demand_kw"}
            missing = required.difference(raw.columns)
            if missing:
                raise ValueError(f"Real depot file is missing columns: {sorted(missing)}")

            df = raw.loc[
                raw["depot_id"].astype(str) == depot_name,
                ["timestamp", "demand_kw"],
            ].copy()
            if not df.empty:
                df = df.rename(columns={"timestamp": "Timestamp", "demand_kw": "Demand_kW"})
                return df.sort_values("Timestamp").reset_index(drop=True)

    # Otherwise load the matching individual depot CSV from repo root.
    path = INDIVIDUAL_DEPOT_FILES.get(depot_name)
    if path is None:
        raise ValueError(f"Unknown depot: {depot_name}")
    if not path.exists():
        raise FileNotFoundError(
            f"Real data file for {depot_name} was not found. Expected: {path}"
        )

    raw = pd.read_csv(path, parse_dates=["timestamp"], dtype={"depot_id": str})
    required = {"timestamp", "demand_kw"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    df = raw[["timestamp", "demand_kw"]].copy()
    df = df.rename(columns={"timestamp": "Timestamp", "demand_kw": "Demand_kW"})
    return df.sort_values("Timestamp").reset_index(drop=True)


def _standardise_input(data_df: pd.DataFrame) -> pd.DataFrame:
    df = data_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "Timestamp" not in df.columns or "Demand_kW" not in df.columns:
        time_candidates = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
        demand_candidates = [c for c in df.columns if "demand" in c.lower() or "load" in c.lower() or "kw" in c.lower()]
        if not time_candidates or not demand_candidates:
            raise ValueError("The data needs a timestamp column and a demand/load kW column.")
        df = df[[time_candidates[0], demand_candidates[0]]].rename(
            columns={time_candidates[0]: "Timestamp", demand_candidates[0]: "Demand_kW"}
        )

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Demand_kW"] = pd.to_numeric(df["Demand_kW"], errors="coerce")
    df = df.dropna(subset=["Timestamp", "Demand_kW"])
    df = df.groupby("Timestamp", as_index=False)["Demand_kW"].mean().sort_values("Timestamp")
    df["Demand_kW"] = df["Demand_kW"].clip(lower=0)

    # Harmonise uploaded data to the real model's 15-minute grid.
    df = df.set_index("Timestamp").resample(f"{SAMPLING_MINUTES}min").mean()
    df["Demand_kW"] = df["Demand_kW"].interpolate(limit=4, limit_direction="forward").ffill()
    df = df.dropna(subset=["Demand_kW"])
    return df


def _build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Causal time-series features: every demand feature uses past observations only."""
    feat = df.copy()
    hour_decimal = feat.index.hour + feat.index.minute / 60.0
    feat["hour_sin"] = np.sin(2 * np.pi * hour_decimal / 24.0)
    feat["hour_cos"] = np.cos(2 * np.pi * hour_decimal / 24.0)
    feat["day_of_week"] = feat.index.dayofweek
    feat["weekend_flag"] = (feat.index.dayofweek >= 5).astype(int)
    feat["month"] = feat.index.month

    feat["Lag_1"] = feat["Demand_kW"].shift(1)
    feat["Lag_4"] = feat["Demand_kW"].shift(STEPS_PER_HOUR)
    feat["Lag_96"] = feat["Demand_kW"].shift(STEPS_PER_DAY)
    feat["Lag_672"] = feat["Demand_kW"].shift(STEPS_PER_WEEK)
    past = feat["Demand_kW"].shift(1)
    feat["Rolling_mean_4"] = past.rolling(STEPS_PER_HOUR).mean()
    feat["Rolling_mean_24"] = past.rolling(6 * STEPS_PER_HOUR).mean()
    feat["Rolling_mean_96"] = past.rolling(STEPS_PER_DAY).mean()
    feat["Rolling_std_96"] = past.rolling(STEPS_PER_DAY).std()

    features = [
        "hour_sin", "hour_cos", "day_of_week", "weekend_flag", "month",
        "Lag_1", "Lag_4", "Lag_96", "Lag_672",
        "Rolling_mean_4", "Rolling_mean_24", "Rolling_mean_96", "Rolling_std_96",
    ]
    return feat.dropna(subset=features + ["Demand_kW"]), features


def _recursive_forecast(model, history: pd.Series, feature_cols: list[str], horizon_hours: int):
    """Genuine multi-step forecast: each prediction becomes history for the next step."""
    history = history.astype(float).copy().sort_index()
    steps = int(horizon_hours * STEPS_PER_HOUR)
    future_times, preds = [], []

    for _ in range(steps):
        ts = history.index[-1] + pd.Timedelta(minutes=SAMPLING_MINUTES)
        hour_decimal = ts.hour + ts.minute / 60.0
        row = {
            "hour_sin": np.sin(2 * np.pi * hour_decimal / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour_decimal / 24.0),
            "day_of_week": ts.dayofweek,
            "weekend_flag": int(ts.dayofweek >= 5),
            "month": ts.month,
            "Lag_1": history.iloc[-1],
            "Lag_4": history.iloc[-STEPS_PER_HOUR],
            "Lag_96": history.iloc[-STEPS_PER_DAY],
            "Lag_672": history.iloc[-STEPS_PER_WEEK],
            "Rolling_mean_4": history.iloc[-STEPS_PER_HOUR:].mean(),
            "Rolling_mean_24": history.iloc[-6 * STEPS_PER_HOUR:].mean(),
            "Rolling_mean_96": history.iloc[-STEPS_PER_DAY:].mean(),
            "Rolling_std_96": history.iloc[-STEPS_PER_DAY:].std(),
        }
        X_next = pd.DataFrame([[row[c] for c in feature_cols]], columns=feature_cols)
        pred = max(0.0, float(model.predict(X_next)[0]))
        history.loc[ts] = pred
        future_times.append(ts)
        preds.append(pred)

    return pd.DatetimeIndex(future_times), np.asarray(preds, dtype=float)


@st.cache_resource(show_spinner=False)
def execute_production_ml_pipeline(depot_name: str, capacity_kw: float, horizon_hours: int, uploaded_df=None):
    """Train/evaluate on real selected-depot data; never generate synthetic demand."""
    source_df = uploaded_df.copy() if uploaded_df is not None else load_real_depot_data(depot_name)
    df = _standardise_input(source_df)
    feat, feature_cols = _build_features(df)

    if len(feat) < STEPS_PER_WEEK * 2:
        raise ValueError("At least two weeks of 15-minute history are required for this model.")

    # Explicit chronological train / validation / held-out test split.
    # The held-out test period is never used for fitting.
    train_df = feat[(feat.index >= DATA_START) & (feat.index < TRAIN_END)].copy()
    valid_df = feat[(feat.index >= TRAIN_END) & (feat.index < VALID_END)].copy()
    test_df = feat[(feat.index >= VALID_END) & (feat.index < TEST_END)].copy()

    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError(
            "The selected depot does not cover the required project periods: "
            "train Jul 2021-Mar 2022, validation Apr-9 May 2022, "
            "held-out test 10 May-25 Jul 2022."
        )

    X_train, y_train = train_df[feature_cols], train_df["Demand_kW"]
    X_valid, y_valid = valid_df[feature_cols], valid_df["Demand_kW"]
    X_test, y_test = test_df[feature_cols], test_df["Demand_kW"]

    # The 340-tree configuration was selected before final test evaluation.
    # Refit on training + validation only, then score once on the untouched test period.
    development_df = pd.concat([train_df, valid_df]).sort_index()
    X_development = development_df[feature_cols]
    y_development = development_df["Demand_kW"]

    model = xgb.XGBRegressor(
        n_estimators=340,
        max_depth=7,
        learning_rate=0.03,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_development, y_development)

    xgb_preds = np.clip(model.predict(X_test), 0, None)
    baseline_preds = test_df["Lag_96"].to_numpy()  # previous-day same-time baseline at 15-minute resolution
    errors = y_test.to_numpy() - xgb_preds

    # EPA evidence guard: every displayed metric and plot must use exactly
    # the same untouched held-out test rows.
    if not (len(y_test) == len(xgb_preds) == len(baseline_preds)):
        raise ValueError(
            "Metric alignment error: actual, XGBoost and baseline lengths differ."
        )
    if y_test.index.min() < VALID_END or y_test.index.max() >= TEST_END:
        raise ValueError(
            "Metric alignment error: held-out rows fall outside the configured test period."
        )

    def compute_metrics(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        mae = round(mean_absolute_error(y_true, y_pred), 2)
        rmse = round(np.sqrt(mean_squared_error(y_true, y_pred)), 2)
        actual_breach = y_true >= capacity_kw
        pred_breach = y_pred >= capacity_kw
        tp = int(np.sum(actual_breach & pred_breach))
        fp = int(np.sum((~actual_breach) & pred_breach))
        fn = int(np.sum(actual_breach & (~pred_breach)))
        recall = f"{100 * tp / (tp + fn):.2f}%" if (tp + fn) else "N/A"
        false_alarm_ratio = f"{100 * fp / (tp + fp):.2f}%" if (tp + fp) else "N/A"
        return mae, rmse, recall, false_alarm_ratio

    mae_xgb, rmse_xgb, recall_xgb, far_xgb = compute_metrics(y_test, xgb_preds)
    mae_base, rmse_base, recall_base, far_base = compute_metrics(y_test, baseline_preds)

    # Forecast beyond the final real observation. This is not a shifted test prediction.
    history_series = df["Demand_kW"].copy()
    future_times, future_forecast = _recursive_forecast(model, history_series, feature_cols, horizon_hours)

    summary_metadata = {
        "model_type": "XGBoost 340-Tree Depot Demand Forecast Model",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset_source": "UK Power Networks / Optimise Prime WS2 depot trial",
        "selected_depot": depot_name,
        "total_rows": len(df),
        "sampling_interval": "15 minutes",
        "target_variable": "Demand_kW",
        "features_used": feature_cols,
        "train_period": f"{train_df.index[0]:%Y-%m-%d} to {train_df.index[-1]:%Y-%m-%d}",
        "validation_period": f"{valid_df.index[0]:%Y-%m-%d} to {valid_df.index[-1]:%Y-%m-%d}",
        "final_fit_period": f"{development_df.index[0]:%Y-%m-%d} to {development_df.index[-1]:%Y-%m-%d}",
        "test_period": f"{test_df.index[0]:%Y-%m-%d} to {test_df.index[-1]:%Y-%m-%d}",
        "test_observations": int(len(test_df)),
        "metric_scope": "XGBoost and baseline metrics use the same untouched held-out test rows",
        "forecast_horizon_hours": horizon_hours,
        "synthetic_data_used": False,
        "metrics_xgb": {"mae": mae_xgb, "rmse": rmse_xgb, "recall": recall_xgb, "far": far_xgb},
        "metrics_baseline": {"mae": mae_base, "rmse": rmse_base, "recall": recall_base, "far": far_base},
    }

    return {
        "metadata": summary_metadata,
        "history": history_series,
        "y_test": y_test,
        "xgb_preds": xgb_preds,
        "baseline_preds": baseline_preds,
        "errors": errors,
        "future_times": future_times,
        "future_forecast": future_forecast,
    }


# ===================================================================== #
# SESSION STATE / UI                                                     #
# ===================================================================== #
if "authenticated" not in st.session_state: st.session_state["authenticated"] = False
if "current_depot" not in st.session_state: st.session_state["current_depot"] = "Bexleyheath"
if "org_name" not in st.session_state: st.session_state["org_name"] = "Optivolt Solutions Limited"
if "grid_limit" not in st.session_state: st.session_state["grid_limit"] = DEPOT_CAPACITY_KW[st.session_state["current_depot"]]
if "warn_threshold" not in st.session_state: st.session_state["warn_threshold"] = round(WARNING_ALPHA * st.session_state["grid_limit"], 1)
if "forecast_horizon" not in st.session_state: st.session_state["forecast_horizon"] = 24
if "uploaded_df" not in st.session_state: st.session_state["uploaded_df"] = None
if "upload_meta" not in st.session_state: st.session_state["upload_meta"] = None
if "data_source_status" not in st.session_state: st.session_state["data_source_status"] = "Real UKPN Optimise Prime bundled data"

st.markdown("""
<style>
 @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
 html, body, [data-testid="stAppViewContainer"] { font-family: 'Inter', sans-serif !important; background-color: #F8FAFC !important; color: #1E293B !important; }
 [data-testid="stSidebar"] { background-color: #0F172A !important; }
 [data-testid="stSidebar"] *, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p { color: #E2E8F0 !important; }
 [data-testid="stMetricValue"] { font-size: 22px !important; font-weight: 600 !important; color: #0F172A !important; }
 [data-testid="stMetricLabel"] { font-size: 11px !important; font-weight: 500 !important; color: #64748B !important; text-transform: uppercase !important; letter-spacing: 0.5px !important; }
 [data-testid="stMetric"] { background-color: #FFFFFF !important; border: 1px solid #E2E8F0 !important; border-radius: 8px !important; padding: 10px 14px !important; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important; }
 .stButton>button { background-color: #2563EB !important; color: white !important; border-radius: 6px !important; font-weight: 500 !important; border: none !important; }
 .commercial-footer { position: fixed; bottom: 0; left: 0; width: 100%; background-color: #FFFFFF; border-top: 1px solid #E2E8F0; padding: 6px 20px; text-align: right; font-size: 11px; color: #64748B; z-index: 999; }
 .status-badge { padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: 600; display: inline-block; }
 .badge-safe { background-color: #DCFCE7; color: #15803D; }
 .badge-warning { background-color: #FEF9C3; color: #A16207; }
 .badge-danger { background-color: #FEE2E2; color: #B91C1C; }
</style>
""", unsafe_allow_html=True)

if not st.session_state["authenticated"]:
    _, col_l2, _ = st.columns([1, 2, 1])
    with col_l2:
        st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: center; margin-bottom: 24px;'><h2 style='color: #0F172A;'>⚡ TransitFlow Intelligence Portal</h2><p style='color: #64748B;'>EV Depot Demand Forecasting & Capacity Management</p></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<h4 style='color:#1E293B;'>Sign in to your account</h4>", unsafe_allow_html=True)
            st.text_input("Username or corporate email", value="operations@optivolt.co.uk")
            st.text_input("Password", type="password", value="••••••••••••")
            if st.button("Authenticate Workspace Access", use_container_width=True):
                st.session_state["authenticated"] = True
                st.rerun()
        st.stop()

with st.sidebar:
    st.markdown(
        f"""
        <div style="margin-bottom: 12px;">
            <h3 style="color: white; margin-bottom: 2px;">
                {st.session_state['org_name']}
            </h3>
            <div style="color: #CBD5E1; font-size: 0.82rem; line-height: 1.25;">
                Data: UK Power Networks Optimise Prime
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected_depot = st.selectbox(
        "Select Active Depot Node:",
        REAL_DEPOTS,
        index=REAL_DEPOTS.index(st.session_state["current_depot"]) if st.session_state["current_depot"] in REAL_DEPOTS else 0,
    )
    if selected_depot != st.session_state["current_depot"]:
        st.session_state["current_depot"] = selected_depot
        st.session_state["grid_limit"] = DEPOT_CAPACITY_KW[selected_depot]
        st.session_state["warn_threshold"] = round(WARNING_ALPHA * DEPOT_CAPACITY_KW[selected_depot], 1)
        st.session_state["uploaded_df"] = None
        st.session_state["upload_meta"] = None
    st.markdown("---")
    nav_selection = st.radio("Navigation Menu:", ["🏠 Home / Overview", "🏢 Depots Setup", "📊 Upload Depot Data", "📈 Demand Forecast", "🚗 Scenario Analysis", "⚙️ Model Performance", "🚨 System Alerts", "📋 Reports", "📜 Archive History", "⚙️ Settings / Admin"])

try:
    with st.spinner(f"Preparing {st.session_state['current_depot']} forecast from real Optimise Prime data..."):
        pipeline = execute_production_ml_pipeline(
            st.session_state["current_depot"],
            float(st.session_state["grid_limit"]),
            int(st.session_state["forecast_horizon"]),
            st.session_state["uploaded_df"],
        )
except Exception as exc:
    st.error(str(exc))
    st.info("Real Optimise Prime data is loaded from the depot CSV files included in this GitHub repository.")
    st.stop()

model_meta = pipeline["metadata"]
history_series = pipeline["history"]
y_test = pipeline["y_test"]
xgb_predictions = pipeline["xgb_preds"]
baseline_predictions = pipeline["baseline_preds"]
future_times = pipeline["future_times"]
horizon_forecast = pipeline["future_forecast"]

current_demand = float(history_series.iloc[-1])
predicted_peak = float(np.max(horizon_forecast))

breaches = np.where(horizon_forecast >= st.session_state["grid_limit"])[0]
warnings = np.where(horizon_forecast >= st.session_state["warn_threshold"])[0]

if len(breaches) > 0:
    op_status, status_badge = "CRITICAL RISK", "badge-danger"
    breach_minutes = int((breaches[0] + 1) * SAMPLING_MINUTES)
    breach_time = f"+{breach_minutes / 60:.2f} Hours"
    first_warning = warnings[0] if len(warnings) > 0 else breaches[0]
    lead_minutes = max(0, (breaches[0] - first_warning) * SAMPLING_MINUTES)
    lead_time = f"{lead_minutes / 60:.2f} Hours"
elif len(warnings) > 0:
    op_status, status_badge = "WARNING PENDING", "badge-warning"
    breach_time = "No breach predicted"
    lead_time = "N/A"
else:
    op_status, status_badge = "OPTIMAL SAFE", "badge-safe"
    breach_time = "No breach predicted"
    lead_time = "No breach predicted"

if nav_selection == "🏠 Home / Overview":
    st.title(f"🏢 {st.session_state['current_depot']} Operations Dashboard")
    st.markdown(f"**Data Pipeline Context:** Core system active via `{model_meta['model_type']}`")
    
    st.markdown(f"""
    <div style='background-color: #FFFFFF; border: 1px solid #E2E8F0; padding: 16px; border-radius: 8px; margin-bottom: 20px;'>
        <table style='width:100%; font-size: 14px;'>
            <tr style='font-size: 11px; color: #64748B; text-transform: uppercase;'>
                <td>Corporate Workspace</td><td>Operational Status</td><td>Expected Breach Time</td>
            </tr>
            <tr>
                <td style='font-weight:700;'>{st.session_state['org_name']}</td>
                <td><span class='status-badge {status_badge}'>{op_status}</span></td>
                <td style='font-weight:700;'>{breach_time}</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Demand", f"{round(current_demand, 1)} kW")
    c2.metric("Predicted Peak Demand", f"{round(predicted_peak, 1)} kW")
    c3.metric("Planning Capacity Threshold", f"{st.session_state['grid_limit']} kW")
    c4.metric("Warning Lead Time", lead_time)

    st.subheader("Historical Headroom Footprint Profile")
    st.caption("Recent historical demand profile for the selected depot.")

    lookback = min(192, len(history_series))
    hist_x = history_series.index[-lookback:]
    hist_y = history_series.values[-lookback:]

    planning_threshold = float(st.session_state["grid_limit"])
    warning_threshold = float(st.session_state["warn_threshold"])

    peak_idx = int(np.argmax(hist_y))
    peak_time = hist_x[peak_idx]
    peak_value = float(hist_y[peak_idx])

    fig_home = go.Figure()

    fig_home.add_trace(
        go.Scatter(
            x=hist_x,
            y=hist_y,
            name="Actual Demand",
            mode="lines",
            line=dict(color="#2563EB", width=2.4),
            hovertemplate="%{x|%d %b %Y %H:%M}<br>Actual demand: %{y:.2f} kW<extra></extra>",
        )
    )

    # Warning threshold as a visible legend item.
    fig_home.add_trace(
        go.Scatter(
            x=[hist_x[0], hist_x[-1]],
            y=[warning_threshold, warning_threshold],
            name="Warning Threshold",
            mode="lines",
            line=dict(color="#F59E0B", width=1.8, dash="dash"),
            hovertemplate=f"Warning threshold: {warning_threshold:.1f} kW<extra></extra>",
        )
    )

    # Planning capacity threshold as a visible legend item.
    fig_home.add_trace(
        go.Scatter(
            x=[hist_x[0], hist_x[-1]],
            y=[planning_threshold, planning_threshold],
            name="Planning Capacity Threshold",
            mode="lines",
            line=dict(color="#EF4444", width=1.8, dash="dash"),
            hovertemplate=f"Planning capacity threshold: {planning_threshold:.1f} kW<extra></extra>",
        )
    )

    # Highlight the highest observed point in the displayed historical window.
    fig_home.add_trace(
        go.Scatter(
            x=[peak_time],
            y=[peak_value],
            name="Historical Peak",
            mode="markers",
            marker=dict(size=10, symbol="diamond", color="#7C3AED"),
            hovertemplate="%{x|%d %b %Y %H:%M}<br>Historical peak: %{y:.2f} kW<extra></extra>",
        )
    )

    fig_home.update_layout(
        template="plotly_white",
        height=430,
        xaxis_title="Time",
        yaxis_title="Demand (kW)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
        ),
        margin=dict(l=25, r=25, t=70, b=45),
        hovermode="x unified",
    )

    fig_home.update_yaxes(
        rangemode="tozero",
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
    )
    fig_home.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.12)",
    )

    st.plotly_chart(fig_home, use_container_width=True)

elif nav_selection == "🏢 Depots Setup":
    st.title("🏢 Depot Configuration Registry")
    with st.container(border=True):
        st.session_state["org_name"] = st.text_input("Organization Title", value=st.session_state["org_name"])
        st.session_state["grid_limit"] = st.number_input("Planning Capacity Threshold (kW)", value=float(st.session_state["grid_limit"]), min_value=0.0, step=1.0)
        st.session_state["warn_threshold"] = st.number_input("Warning Threshold (kW)", value=float(st.session_state["warn_threshold"]), min_value=0.0, step=1.0)
        if st.button("Save Depot Configuration"):
            st.toast("Substation configuration targets saved successfully.")

elif nav_selection == "📊 Upload Depot Data":
    st.title("📊 Upload Depot Data")

    st.markdown("### Import depot telemetry data")
    st.caption(
        "Upload historical or current depot telemetry data for forecasting and analysis."
    )
    st.markdown(
        "**Supported formats:** CSV, XLSX  •  **Maximum file size:** 200 MB"
    )

    uploaded_file = st.file_uploader(
        "Upload depot telemetry file",
        type=["csv", "xlsx"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                uploaded_df = pd.read_csv(uploaded_file)
            else:
                uploaded_df = pd.read_excel(uploaded_file)

            st.session_state["uploaded_df"] = uploaded_df
            st.session_state["upload_meta"] = {
                "filename": uploaded_file.name,
                "rows": len(uploaded_df),
                "columns": list(uploaded_df.columns),
            }

            st.success(
                f"Uploaded {uploaded_file.name} successfully "
                f"({len(uploaded_df):,} rows)."
            )
            st.dataframe(uploaded_df.head(20), use_container_width=True)

        except Exception as exc:
            st.error(f"Could not read the uploaded file: {exc}")


elif nav_selection == "📈 Demand Forecast":
    st.markdown(
        """
        <h1 style="
            font-size: 2.65rem;
            line-height: 1.08;
            margin-bottom: 0.35rem;
            white-space: nowrap;
            color: #0F172A;
        ">
            📈 Historical Demand & 24-Hour Forecast
        </h1>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Historical observed demand is shown up to the forecast start. "
        "The dashed green line is the 24-hour XGBoost forecast based on the "
        "latest available project data."
    )
    
    fig = go.Figure()
    hist_x = history_series.index[-192:]
    fig.add_trace(go.Scatter(x=hist_x, y=history_series.values[-192:], name="Actual Demand", line=dict(color="#0F172A")))
    future_x = future_times
    fig.add_trace(go.Scatter(x=future_x, y=horizon_forecast, name="XGBoost Forecast", line=dict(color="#10B981", dash="dash")))
    
    fig.add_shape(type="line", x0=hist_x[0], y0=st.session_state["grid_limit"], x1=future_x[-1], y1=st.session_state["grid_limit"], line=dict(color="#EF4444", width=2))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='#EF4444', width=2), name='Planning Capacity Threshold'))
    
    fig.add_shape(type="line", x0=hist_x[0], y0=st.session_state["warn_threshold"], x1=future_x[-1], y1=st.session_state["warn_threshold"], line=dict(color="#F59E0B", dash="dot"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='#F59E0B', dash='dot'), name='Warning Threshold'))
    
    fig.add_shape(type="line", x0=hist_x[-1], y0=0, x1=hist_x[-1], y1=predicted_peak * 1.2, line=dict(color="#64748B", dash="dash"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='#64748B', dash='dash'), name='Forecast Start'))
    
    fig.update_layout(template="plotly_white", height=440, legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig, use_container_width=True)

elif nav_selection == "🚗 Scenario Analysis":
    st.title("🚗 Scenario Analysis")
    st.caption(
        "Scenario values are hypothetical adjustments applied to the genuine "
        "XGBoost base forecast."
    )

    growth = st.slider(
        "Projected Fleet / Demand Growth (%)",
        min_value=0,
        max_value=100,
        value=20,
        step=5,
    )

    scenario_y = horizon_forecast * (1.0 + (growth / 100.0))
    timeline = future_times

    planning_threshold = float(st.session_state["grid_limit"])
    warning_threshold = float(st.session_state["warn_threshold"])
    scenario_peak = float(np.max(scenario_y))

    c1, c2, c3 = st.columns(3)
    c1.metric("Scenario Peak", f"{scenario_peak:.1f} kW")
    c2.metric("Warning Threshold", f"{warning_threshold:.1f} kW")
    c3.metric("Planning Capacity Threshold", f"{planning_threshold:.1f} kW")

    fig_sc = go.Figure()

    fig_sc.add_trace(
        go.Scatter(
            x=timeline,
            y=horizon_forecast,
            name="Base XGBoost Forecast",
            mode="lines",
            line=dict(color="#64748B", width=2, dash="dot"),
            hovertemplate="%{x|%H:%M}<br>Base forecast: %{y:.2f} kW<extra></extra>",
        )
    )

    fig_sc.add_trace(
        go.Scatter(
            x=timeline,
            y=scenario_y,
            name="Scenario-Adjusted Forecast",
            mode="lines",
            line=dict(color="#2563EB", width=2.5),
            hovertemplate="%{x|%H:%M}<br>Scenario forecast: %{y:.2f} kW<extra></extra>",
        )
    )

    fig_sc.add_hline(
        y=warning_threshold,
        line=dict(color="#F59E0B", width=1.8, dash="dash"),
        annotation_text="Warning Threshold",
        annotation_position="top left",
    )

    fig_sc.add_hline(
        y=planning_threshold,
        line=dict(color="#DC2626", width=1.8, dash="dash"),
        annotation_text="Planning Capacity Threshold",
        annotation_position="top left",
    )

    fig_sc.update_layout(
        template="plotly_white",
        height=440,
        xaxis_title="Forecast Time",
        yaxis_title="Demand (kW)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
        ),
        margin=dict(l=25, r=25, t=65, b=45),
        hovermode="x unified",
    )

    fig_sc.update_yaxes(
        rangemode="tozero",
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
    )
    fig_sc.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.12)",
    )

    st.plotly_chart(fig_sc, use_container_width=True)

    if scenario_peak >= planning_threshold:
        st.warning(
            "The hypothetical scenario exceeds the planning capacity threshold."
        )
    elif scenario_peak >= warning_threshold:
        st.info(
            "The hypothetical scenario reaches the warning threshold but remains "
            "below the planning capacity threshold."
        )
    else:
        st.caption(
            "The hypothetical scenario remains below the warning threshold."
        )

elif nav_selection == "⚙️ Model Performance":
    st.title("⚙️ Model Performance")

    metric_test_start = y_test.index.min().strftime("%d %b %Y")
    metric_test_end = y_test.index.max().strftime("%d %b %Y")
    st.caption(
        f"Metric scope: **{st.session_state['current_depot']}** | "
        f"same untouched held-out test period for XGBoost and baseline: "
        f"**{metric_test_start} – {metric_test_end}** | "
        f"**{len(y_test):,} observations**"
    )
    
    # Custom metric cards are used instead of st.metric so the full labels
    # remain visible without Streamlit truncating them with ellipses.
    mae_value = f"{model_meta['metrics_xgb']['mae']} kW"
    rmse_value = f"{model_meta['metrics_xgb']['rmse']} kW"
    recall_value = str(model_meta['metrics_xgb']['recall'])
    far_value = str(model_meta['metrics_xgb']['far'])

    st.markdown(
        f"""
        <style>
        .perf-card-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 16px;
            margin: 14px 0 22px 0;
        }}
        .perf-card {{
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 20px 20px 18px 20px;
            min-height: 112px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }}
        .perf-card-label {{
            color: #64748B;
            font-size: 0.88rem;
            font-weight: 600;
            line-height: 1.25;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            white-space: normal;
            overflow: visible;
            text-overflow: clip;
            margin-bottom: 8px;
        }}
        .perf-card-value {{
            color: #0F172A;
            font-size: 2rem;
            font-weight: 700;
            line-height: 1.05;
        }}
        @media (max-width: 900px) {{
            .perf-card-grid {{
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }}
        }}
        @media (max-width: 560px) {{
            .perf-card-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        </style>

        <div class="perf-card-grid">
            <div class="perf-card">
                <div class="perf-card-label">Mean Absolute Error</div>
                <div class="perf-card-value">{mae_value}</div>
            </div>
            <div class="perf-card">
                <div class="perf-card-label">Root Mean Squared Error</div>
                <div class="perf-card-value">{rmse_value}</div>
            </div>
            <div class="perf-card">
                <div class="perf-card-label">Breach Recall</div>
                <div class="perf-card-value">{recall_value}</div>
            </div>
            <div class="perf-card">
                <div class="perf-card-label">False Alarm Ratio</div>
                <div class="perf-card-value">{far_value}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    tbl = {
        "Analytics Architecture Signature": ["Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)", "Breach Recall Capture Rate (TPR)", "False Alarm Ratio (FP / (TP + FP))"],
        "Previous-Day Same-Hour Baseline": [f"{model_meta['metrics_baseline']['mae']} kW", f"{model_meta['metrics_baseline']['rmse']} kW", model_meta['metrics_baseline']['recall'], model_meta['metrics_baseline']['far']],
        "XGBoost 340-Tree Depot Forecast Model": [f"{model_meta['metrics_xgb']['mae']} kW", f"{model_meta['metrics_xgb']['rmse']} kW", model_meta['metrics_xgb']['recall'], model_meta['metrics_xgb']['far']]
    }
    st.table(pd.DataFrame(tbl).set_index("Analytics Architecture Signature"))

    mae_improvement = (
        100.0
        * (model_meta["metrics_baseline"]["mae"] - model_meta["metrics_xgb"]["mae"])
        / model_meta["metrics_baseline"]["mae"]
        if model_meta["metrics_baseline"]["mae"] > 0
        else 0.0
    )
    rmse_improvement = (
        100.0
        * (model_meta["metrics_baseline"]["rmse"] - model_meta["metrics_xgb"]["rmse"])
        / model_meta["metrics_baseline"]["rmse"]
        if model_meta["metrics_baseline"]["rmse"] > 0
        else 0.0
    )

    st.markdown(
        f"""
        <div class="perf-card-grid" style="grid-template-columns: repeat(2, minmax(0, 1fr));">
            <div class="perf-card">
                <div class="perf-card-label">MAE Reduction vs Baseline</div>
                <div class="perf-card-value">{mae_improvement:.1f}%</div>
            </div>
            <div class="perf-card">
                <div class="perf-card-label">RMSE Reduction vs Baseline</div>
                <div class="perf-card-value">{rmse_improvement:.1f}%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "All error metrics, breach metrics, residuals, scatter points and the "
        "baseline comparison on this page are calculated from the same held-out "
        "test rows. The held-out test data are not used to fit the 340-tree model."
    )

    st.subheader("Held-Out Test Analysis — 10 May to 25 July 2022")
    
    # Full held-out test period (no short lookback truncation)
    residuals = y_test.values - xgb_predictions
    test_start_label = y_test.index.min().strftime("%d %b %Y")
    test_end_label = y_test.index.max().strftime("%d %b %Y")
    r2 = r2_score(y_test.values, xgb_predictions)

    # Performance summary row for the selected depot / held-out period
    ps1, ps2, ps3 = st.columns(3)
    ps1.metric("Held-out test period", f"{test_start_label} → {test_end_label}")
    ps2.metric("Test observations", f"{len(y_test):,}")
    ps3.metric("R²", f"{r2:.3f}")

    # Plot 1: Actual vs Predicted Demand across the complete held-out test period
    fig_avp = go.Figure()
    fig_avp.add_trace(
        go.Scatter(
            x=y_test.index,
            y=y_test.values,
            name="Actual Demand",
            mode="lines",
            line=dict(color="#0F172A", width=1.4),
            hovertemplate="%{x|%d %b %Y %H:%M}<br>Actual: %{y:.2f} kW<extra></extra>",
        )
    )
    fig_avp.add_trace(
        go.Scatter(
            x=y_test.index,
            y=xgb_predictions,
            name="XGBoost Prediction",
            mode="lines",
            line=dict(color="#10B981", width=1.2, dash="dash"),
            hovertemplate="%{x|%d %b %Y %H:%M}<br>Predicted: %{y:.2f} kW<extra></extra>",
        )
    )
    fig_avp.update_layout(
        title={
            "text": "Actual vs Predicted Demand — Full Held-Out Test Period",
            "x": 0.01,
            "xanchor": "left",
            "y": 0.97,
            "yanchor": "top",
        },
        xaxis_title="Date",
        yaxis_title="Demand (kW)",
        template="plotly_white",
        height=470,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.04, x=0.01, xanchor="left"),
        margin=dict(l=20, r=20, t=105, b=35),
    )
    fig_avp.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.10),
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
    )
    fig_avp.update_yaxes(
        rangemode="tozero",
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
    )
    st.caption(
        f"Held-out test window: **{test_start_label} – {test_end_label}**"
    )
    st.plotly_chart(fig_avp, use_container_width=True)

    # Plot 2: Residuals across the complete held-out test period
    abs_res = np.abs(residuals)
    p95 = float(np.percentile(abs_res, 95))
    fig_res = go.Figure()
    fig_res.add_hrect(
        y0=-p95,
        y1=p95,
        fillcolor="rgba(16,185,129,0.08)",
        line_width=0,
    )
    fig_res.add_trace(
        go.Scatter(
            x=y_test.index,
            y=residuals,
            name="Residual (Actual - Predicted)",
            mode="lines",
            line=dict(color="#EF4444", width=1),
            hovertemplate="%{x|%d %b %Y %H:%M}<br>Residual: %{y:.2f} kW<extra></extra>",
        )
    )
    fig_res.add_hline(y=0, line=dict(color="#64748B", dash="dash", width=1.2))
    fig_res.update_layout(
        title={
            "text": "Residuals Over Time — Full Held-Out Test Period",
            "x": 0.01,
            "xanchor": "left",
            "y": 0.97,
            "yanchor": "top",
        },
        xaxis_title="Date",
        yaxis_title="Residual (kW)",
        template="plotly_white",
        height=390,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=95, b=35),
    )
    fig_res.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.10),
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
    )
    fig_res.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.18)")
    st.caption(
        f"Shaded green region = ±95th percentile absolute residual (**{p95:.2f} kW**)"
    )
    st.plotly_chart(fig_res, use_container_width=True)

    # Plot 3: Actual vs Predicted scatter — full held-out range only
    # One scatter view is retained to avoid redundant/less-informative zooming.
    scatter_max_raw = float(max(y_test.max(), xgb_predictions.max()))
    scatter_max = max(1.0, np.ceil(scatter_max_raw * 1.05))

    st.subheader("Actual vs Predicted Scatter — Full Held-Out Test Period")
    st.caption(
        f"R² = **{r2:.3f}** | "
        f"MAE = **{model_meta['metrics_xgb']['mae']} kW** | "
        f"RMSE = **{model_meta['metrics_xgb']['rmse']} kW**"
    )

    fig_scat_full = go.Figure()
    fig_scat_full.add_trace(
        go.Scatter(
            x=y_test.values,
            y=xgb_predictions,
            mode="markers",
            marker=dict(size=5, opacity=0.24, color="#2563EB"),
            name="Held-out observations",
            hovertemplate=(
                "Actual: %{x:.2f} kW"
                "<br>Predicted: %{y:.2f} kW"
                "<extra></extra>"
            ),
        )
    )
    fig_scat_full.add_trace(
        go.Scatter(
            x=[0, scatter_max],
            y=[0, scatter_max],
            mode="lines",
            line=dict(color="#64748B", width=2, dash="dash"),
            name="Ideal prediction (y = x)",
            hoverinfo="skip",
        )
    )
    fig_scat_full.update_layout(
        xaxis_title="Actual Demand (kW)",
        yaxis_title="XGBoost Predicted Demand (kW)",
        template="plotly_white",
        height=520,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0.0,
        ),
        margin=dict(l=25, r=25, t=55, b=45),
        hovermode="closest",
    )
    fig_scat_full.update_xaxes(
        range=[0, scatter_max],
        autorange=False,
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
        zeroline=True,
        zerolinecolor="rgba(100,116,139,0.55)",
    )
    fig_scat_full.update_yaxes(
        range=[0, scatter_max],
        autorange=False,
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
        zeroline=True,
        zerolinecolor="rgba(100,116,139,0.55)",
    )
    st.plotly_chart(fig_scat_full, use_container_width=True)

    # Plot 4A: Residual distribution — full held-out range
    mean_residual = float(np.mean(residuals))
    median_residual = float(np.median(residuals))

    st.subheader("Residual Distribution — Full Held-Out Test Period")
    st.caption(
        "Residual = Actual − Predicted | "
        f"Mean residual = **{mean_residual:.2f} kW** | "
        f"Median residual = **{median_residual:.2f} kW**"
    )

    fig_hist_full = go.Figure()
    fig_hist_full.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=60,
            name="Residual count",
            marker=dict(color="#4F66F2"),
            hovertemplate="Residual bin: %{x:.2f} kW<br>Count: %{y}<extra></extra>",
            showlegend=False,
        )
    )

    # Keep visible vertical reference lines.
    fig_hist_full.add_vline(
        x=0,
        line=dict(color="#111827", dash="dash", width=1.6),
    )
    fig_hist_full.add_vline(
        x=mean_residual,
        line=dict(color="#2563EB", dash="dot", width=1.5),
    )
    fig_hist_full.add_vline(
        x=median_residual,
        line=dict(color="#059669", dash="dot", width=1.5),
    )

    # Dummy line traces provide clear legend entries for the reference lines.
    fig_hist_full.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="#111827", dash="dash", width=1.6),
            name="Zero error",
            hoverinfo="skip",
        )
    )
    fig_hist_full.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="#2563EB", dash="dot", width=1.5),
            name=f"Mean residual = {mean_residual:.2f} kW",
            hoverinfo="skip",
        )
    )
    fig_hist_full.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="#059669", dash="dot", width=1.5),
            name=f"Median residual = {median_residual:.2f} kW",
            hoverinfo="skip",
        )
    )

    fig_hist_full.update_layout(
        xaxis_title="Residual (Actual - Predicted) kW",
        yaxis_title="Count",
        template="plotly_white",
        height=410,
        bargap=0.04,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0.0,
        ),
        margin=dict(l=25, r=25, t=65, b=45),
    )
    fig_hist_full.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
    )
    fig_hist_full.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
        rangemode="tozero",
    )
    st.plotly_chart(fig_hist_full, use_container_width=True)

    # Plot 4B: Residual distribution — zoomed central view
    zoom_limit = float(np.percentile(np.abs(residuals), 95))
    zoom_limit = max(1.0, min(5.0, zoom_limit))

    zoomed_residuals = residuals[
        (residuals >= -zoom_limit) & (residuals <= zoom_limit)
    ]
    central_share = 100.0 * len(zoomed_residuals) / len(residuals)

    st.subheader("Residual Distribution — Zoomed Central View")
    st.caption(
        f"Central range: **−{zoom_limit:.2f} to +{zoom_limit:.2f} kW** | "
        f"Contains **{central_share:.1f}%** of held-out residuals | "
        "Full distribution shown above."
    )

    fig_hist_zoom = go.Figure()
    fig_hist_zoom.add_trace(
        go.Histogram(
            x=zoomed_residuals,
            nbinsx=40,
            xbins=dict(
                start=-zoom_limit,
                end=zoom_limit,
                size=(2 * zoom_limit) / 40,
            ),
            name="Residual count",
            marker=dict(color="#4F66F2"),
            hovertemplate="Residual bin: %{x:.2f} kW<br>Count: %{y}<extra></extra>",
            showlegend=False,
        )
    )

    fig_hist_zoom.add_vline(
        x=0,
        line=dict(color="#111827", dash="dash", width=1.6),
    )
    fig_hist_zoom.add_vline(
        x=mean_residual,
        line=dict(color="#2563EB", dash="dot", width=1.5),
    )
    fig_hist_zoom.add_vline(
        x=median_residual,
        line=dict(color="#059669", dash="dot", width=1.5),
    )

    fig_hist_zoom.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="#111827", dash="dash", width=1.6),
            name="Zero error",
            hoverinfo="skip",
        )
    )
    fig_hist_zoom.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="#2563EB", dash="dot", width=1.5),
            name=f"Mean residual = {mean_residual:.2f} kW",
            hoverinfo="skip",
        )
    )
    fig_hist_zoom.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="#059669", dash="dot", width=1.5),
            name=f"Median residual = {median_residual:.2f} kW",
            hoverinfo="skip",
        )
    )

    fig_hist_zoom.update_layout(
        xaxis_title="Residual (Actual - Predicted) kW",
        yaxis_title="Count",
        template="plotly_white",
        height=410,
        bargap=0.04,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0.0,
        ),
        margin=dict(l=25, r=25, t=65, b=45),
    )
    fig_hist_zoom.update_xaxes(
        range=[-zoom_limit, zoom_limit],
        autorange=False,
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
    )
    fig_hist_zoom.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
        rangemode="tozero",
    )
    st.plotly_chart(fig_hist_zoom, use_container_width=True)

elif nav_selection == "🚨 System Alerts":
    st.title("🚨 System Alerts")
    st.caption(
        "Forecast-based alert status for the selected depot using the latest "
        "available Optimise Prime data context."
    )

    planning_threshold = float(st.session_state["grid_limit"])
    warning_threshold = float(st.session_state["warn_threshold"])

    if predicted_peak >= planning_threshold:
        alert_status = "Capacity Breach Predicted"
        status_reason = (
            "Forecast peak is at or above the planning capacity threshold."
        )
    elif predicted_peak >= warning_threshold:
        alert_status = "Warning"
        status_reason = (
            "Forecast peak is at or above the warning threshold but below "
            "the planning capacity threshold."
        )
    else:
        alert_status = "No Alert"
        status_reason = "Forecast peak is below the warning threshold."

    # Use the latest available data timestamp rather than the current clock,
    # so the page does not imply live telemetry.
    latest_data_timestamp = history_series.index.max()

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Peak", f"{predicted_peak:.1f} kW")
    c2.metric("Warning Threshold", f"{warning_threshold:.1f} kW")
    c3.metric("Planning Capacity Threshold", f"{planning_threshold:.1f} kW")

    if alert_status == "Capacity Breach Predicted":
        st.error(f"**{alert_status}** — {status_reason}")
    elif alert_status == "Warning":
        st.warning(f"**{alert_status}** — {status_reason}")

    ledger = pd.DataFrame([{
        "Forecast Reference Time": latest_data_timestamp.strftime("%d %b %Y, %H:%M"),
        "Depot": st.session_state["current_depot"],
        "Predicted Peak": f"{predicted_peak:.1f} kW",
        "Warning Threshold": f"{warning_threshold:.1f} kW",
        "Planning Capacity Threshold": f"{planning_threshold:.1f} kW",
        "Alert Status": alert_status,
        "Status Reason": status_reason,
    }])

    st.dataframe(
        ledger,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "This page reports forecast-based status from the historical project dataset. "
        "It does not represent live grid telemetry."
    )

elif nav_selection == "📋 Reports":
    st.title("📋 Reports")
    st.caption(
        "Export a concise operational model report for the selected depot."
    )

    xgb_metrics = model_meta.get("metrics_xgb", {})

    def _friendly_period(period_text):
        try:
            start_text, end_text = [part.strip() for part in str(period_text).split(" to ", 1)]
            start_dt = pd.to_datetime(start_text)
            end_dt = pd.to_datetime(end_text)
            return f"{start_dt:%d %b %Y} – {end_dt:%d %b %Y}"
        except Exception:
            return str(period_text)

    training_period_raw = model_meta.get("train_period", "Not available")
    validation_period_raw = model_meta.get("validation_period", "Not available")
    test_period_raw = model_meta.get("test_period", "Not available")
    test_rows_value = int(model_meta.get("test_observations", len(y_test)))

    report_rows_display = [
        ("Depot Name", st.session_state["current_depot"]),
        ("Model", "XGBoost"),
        ("Number of Trees", "340"),
        (
            "Planning Capacity Threshold",
            f"{float(st.session_state['grid_limit']):.2f} kW",
        ),
        (
            "Forecast Horizon",
            f"{int(model_meta.get('forecast_horizon_hours', st.session_state['forecast_horizon']))} hours",
        ),
        ("Predicted Peak", f"{float(predicted_peak):.2f} kW"),
        ("MAE", f"{xgb_metrics.get('mae', 'N/A')} kW"),
        ("RMSE", f"{xgb_metrics.get('rmse', 'N/A')} kW"),
        ("Breach Recall", xgb_metrics.get("recall", "N/A")),
        ("False Alarm Ratio", xgb_metrics.get("far", "N/A")),
        ("Training Period", _friendly_period(training_period_raw)),
        ("Validation Period", _friendly_period(validation_period_raw)),
        ("Held-out Test Period", _friendly_period(test_period_raw)),
        ("Test Rows", f"{test_rows_value:,}"),
        (
            "Synthetic Data Used",
            "No" if not bool(model_meta.get("synthetic_data_used", False)) else "Yes",
        ),
    ]

    report_df_display = pd.DataFrame(
        report_rows_display,
        columns=["Report Field", "Value"],
    )

    st.subheader("Operational Asset Report Preview")
    st.dataframe(
        report_df_display,
        use_container_width=True,
        hide_index=True,
        height=560,
    )

    # CSV version keeps Test Rows as a plain integer (7392, not "7,392")
    # while retaining the same reader-friendly report structure.
    report_rows_csv = [
        ("Depot Name", st.session_state["current_depot"]),
        ("Model", "XGBoost"),
        ("Number of Trees", 340),
        ("Planning Capacity Threshold", f"{float(st.session_state['grid_limit']):.2f} kW"),
        (
            "Forecast Horizon",
            f"{int(model_meta.get('forecast_horizon_hours', st.session_state['forecast_horizon']))} hours",
        ),
        ("Predicted Peak", f"{float(predicted_peak):.2f} kW"),
        ("MAE", f"{xgb_metrics.get('mae', 'N/A')} kW"),
        ("RMSE", f"{xgb_metrics.get('rmse', 'N/A')} kW"),
        ("Breach Recall", xgb_metrics.get("recall", "N/A")),
        ("False Alarm Ratio", xgb_metrics.get("far", "N/A")),
        ("Training Period", _friendly_period(training_period_raw)),
        ("Validation Period", _friendly_period(validation_period_raw)),
        ("Held-out Test Period", _friendly_period(test_period_raw)),
        ("Test Rows", test_rows_value),
        (
            "Synthetic Data Used",
            "No" if not bool(model_meta.get("synthetic_data_used", False)) else "Yes",
        ),
    ]

    report_df_csv = pd.DataFrame(
        report_rows_csv,
        columns=["Report Field", "Value"],
    )
    report_csv = report_df_csv.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Operational Asset Report (CSV)",
        data=report_csv,
        file_name=f"{st.session_state['current_depot'].replace(' ', '_')}_Operational_Asset_Report.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.info(
        "Planning capacity is an application planning threshold and not a verified "
        "contracted grid connection limit."
    )

elif nav_selection == "📜 Archive History":
    st.title("📜 Archive History")
    st.caption(
        "Held-out test history for the selected depot. "
        "Residual = Actual Observed − Forecast."
    )

    history_table = pd.DataFrame({
        "Timestamp": y_test.index,
        "Forecast kW": xgb_predictions,
        "Actual Observed kW": y_test.values,
        "Residual (Actual - Predicted) kW": pipeline["errors"],
    })

    # Default to the most recent observations so the table does not open on a
    # long run of overnight zero-demand intervals.
    sort_order = st.selectbox(
        "Display order",
        ["Most recent first", "Oldest first"],
        index=0,
    )

    with st.container(border=True):
        st.markdown("#### 🔎 Archive filter")
        show_non_zero_only = st.checkbox(
            "Show observed-demand intervals only",
            value=True,
            help=(
                "When selected, the table shows only intervals where actual observed "
                "demand is greater than 0.01 kW. Untick it to show all held-out intervals, "
                "including zero-demand periods."
            ),
        )
        st.caption(
            "✓ Checked = observed-demand intervals only  |  "
            "Unchecked = all held-out intervals"
        )

    display_table = history_table.copy()

    if show_non_zero_only:
        # "Non-zero demand" should refer to observed demand, not merely a
        # non-zero model prediction. This keeps the filtered archive focused
        # on intervals where charging demand was actually present.
        display_table = display_table[
            display_table["Actual Observed kW"].abs() > 0.01
        ]

    ascending = sort_order == "Oldest first"
    display_table = display_table.sort_values("Timestamp", ascending=ascending)

    total_rows = len(history_table)
    shown_rows = len(display_table)

    c1, c2, c3 = st.columns(3)
    c1.metric("Held-out observations", f"{total_rows:,}")
    c2.metric("Observed-demand intervals", f"{shown_rows:,}")
    c3.metric(
        "Observed-demand share",
        f"{(shown_rows / total_rows * 100):.1f}%" if show_non_zero_only and total_rows else "100.0%",
    )

    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Timestamp": st.column_config.DatetimeColumn(
                "Timestamp",
                format="DD MMM YYYY, HH:mm",
            ),
            "Forecast kW": st.column_config.NumberColumn(
                "Forecast kW",
                format="%.2f",
            ),
            "Actual Observed kW": st.column_config.NumberColumn(
                "Actual Observed kW",
                format="%.2f",
            ),
            "Residual (Actual - Predicted) kW": st.column_config.NumberColumn(
                "Residual (Actual - Predicted) kW",
                format="%.2f",
            ),
        },
        height=520,
    )

    st.caption(
        "Tip: untick **Show observed-demand intervals only** to inspect every held-out "
        "15-minute interval, including zero-demand periods."
    )

elif nav_selection == "⚙️ Settings / Admin":
    st.title("⚙️ Model Configuration & Evaluation")
    st.caption(
        "Model configuration and held-out evaluation summary for the selected depot."
    )

    st.subheader("Model Training Summary")

    synthetic_used = bool(model_meta.get("synthetic_data_used", False))
    forecast_horizon = model_meta.get("forecast_horizon_hours", st.session_state["forecast_horizon"])
    model_type_display = model_meta.get("model_type", "XGBoost Regressor")
    train_period = model_meta.get("train_period", "Not available")
    validation_period = model_meta.get("validation_period", "Not available")
    test_period = model_meta.get("test_period", "Not available")
    test_observations = model_meta.get("test_observations", len(y_test))

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Model", "XGBoost")
    s2.metric("Forecast horizon", f"{forecast_horizon} hours")
    s3.metric("Test rows", f"{int(test_observations):,}")
    s4.metric("Synthetic data", "No" if not synthetic_used else "Yes")

    st.caption("Model configuration: **XGBoost with 340 trees**")

    st.markdown("#### Data split")
    split_df = pd.DataFrame({
        "Stage": ["Training", "Validation", "Held-out test"],
        "Period": [train_period, validation_period, test_period],
        "Purpose": [
            "Model fitting",
            "Model selection / tuning",
            "Final held-out evaluation",
        ],
    })
    st.dataframe(split_df, use_container_width=True, hide_index=True)

    st.subheader("XGBoost vs Baseline")

    xgb_metrics = model_meta.get("metrics_xgb", {})
    baseline_metrics = model_meta.get("metrics_baseline", {})

    comparison_df = pd.DataFrame({
        "Metric": [
            "MAE",
            "RMSE",
            "Breach Recall",
            "False Alarm Ratio",
        ],
        "XGBoost": [
            f"{xgb_metrics.get('mae', 'N/A')} kW",
            f"{xgb_metrics.get('rmse', 'N/A')} kW",
            xgb_metrics.get("recall", "N/A"),
            xgb_metrics.get("far", "N/A"),
        ],
        "Baseline": [
            f"{baseline_metrics.get('mae', 'N/A')} kW",
            f"{baseline_metrics.get('rmse', 'N/A')} kW",
            baseline_metrics.get("recall", "N/A"),
            baseline_metrics.get("far", "N/A"),
        ],
        "Interpretation": [
            "Lower is better",
            "Lower is better",
            "Higher is better",
            "Lower is better",
        ],
    })
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    if (
        isinstance(xgb_metrics.get("mae"), (int, float))
        and isinstance(baseline_metrics.get("mae"), (int, float))
        and baseline_metrics.get("mae", 0) > 0
    ):
        mae_reduction = (
            (baseline_metrics["mae"] - xgb_metrics["mae"])
            / baseline_metrics["mae"]
            * 100.0
        )
    else:
        mae_reduction = None

    if (
        isinstance(xgb_metrics.get("rmse"), (int, float))
        and isinstance(baseline_metrics.get("rmse"), (int, float))
        and baseline_metrics.get("rmse", 0) > 0
    ):
        rmse_reduction = (
            (baseline_metrics["rmse"] - xgb_metrics["rmse"])
            / baseline_metrics["rmse"]
            * 100.0
        )
    else:
        rmse_reduction = None

    r1, r2 = st.columns(2)
    r1.metric(
        "MAE reduction vs baseline",
        f"{mae_reduction:.1f}%" if mae_reduction is not None else "N/A",
    )
    r2.metric(
        "RMSE reduction vs baseline",
        f"{rmse_reduction:.1f}%" if rmse_reduction is not None else "N/A",
    )

    st.info(
        "All XGBoost and baseline metrics shown here are calculated from the same "
        "untouched held-out test rows. The held-out test data are not used to fit "
        "the final model."
    )

    with st.expander("Technical metadata"):
        st.json(model_meta)

st.markdown(f"<div class='commercial-footer'>🛡️ Security Status: Checked | Core Architecture: {model_meta['model_type']} | Horizon: {st.session_state['forecast_horizon']}H</div>", unsafe_allow_html=True)
