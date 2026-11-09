
from pathlib import Path
from datetime import datetime, timedelta
import sqlite3
import uuid
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import streamlit as st
from scipy.stats import nbinom
from streamlit_autorefresh import st_autorefresh


# =========================================================
# 1. PATHS AND PAGE CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

DB_PATH = DATA_DIR / "live_hospital.db"
HIST_FILE = OUTPUT_DIR / "hourly_arrivals_by_hospital.csv"
VISITS_FILE = DATA_DIR / "edstays.csv.gz"
MODEL_FILE = OUTPUT_DIR / "hospital_rush_model.joblib"

st.set_page_config(
    page_title="Live Hospital Rush AI",
    page_icon="🏥",
    layout="wide"
)


# =========================================================
# 2. DATABASE FUNCTIONS
# =========================================================

def connect_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(DB_PATH, check_same_thread=False)

    con.execute("""
        CREATE TABLE IF NOT EXISTS live_arrivals (
            id TEXT PRIMARY KEY,
            hospital TEXT NOT NULL,
            department TEXT NOT NULL,
            arrival_time TEXT NOT NULL,
            source TEXT NOT NULL,
            age_band TEXT,
            transport TEXT,
            status TEXT NOT NULL,
            expected_departure TEXT NOT NULL
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS staffing_assumptions (
            hospital TEXT PRIMARY KEY,
            available_doctors INTEGER NOT NULL DEFAULT 1,
            minutes_per_patient REAL NOT NULL DEFAULT 20,
            productive_minutes_per_hour REAL NOT NULL DEFAULT 60
        )
    """)

    con.commit()
    return con


def add_arrival(
    con,
    hospital,
    department,
    source,
    age_band="Not recorded",
    transport="Walk-in",
    when=None
):
    when = when or datetime.now()

    stay_minutes = int(
        np.random.default_rng().integers(45, 181)
    )

    con.execute("""
        INSERT INTO live_arrivals
        (
            id, hospital, department, arrival_time,
            source, age_band, transport, status,
            expected_departure
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        str(uuid.uuid4())[:8].upper(),
        hospital,
        department,
        when.isoformat(timespec="seconds"),
        source,
        age_band,
        transport,
        "Waiting",
        (when + timedelta(minutes=stay_minutes)).isoformat(
            timespec="seconds"
        )
    ))

    con.commit()


def get_live(con):
    df = pd.read_sql_query(
        "SELECT * FROM live_arrivals ORDER BY arrival_time DESC",
        con
    )

    if not df.empty:
        df["arrival_time"] = pd.to_datetime(df["arrival_time"])
        df["expected_departure"] = pd.to_datetime(
            df["expected_departure"]
        )

    return df


# =========================================================
# 3. STAFFING DATABASE FUNCTIONS
# =========================================================

def save_staffing_assumptions(
    hospital,
    available_doctors,
    minutes_per_patient,
    productive_minutes_per_hour
):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO staffing_assumptions (
                hospital,
                available_doctors,
                minutes_per_patient,
                productive_minutes_per_hour
            )
            VALUES (?, ?, ?, ?)

            ON CONFLICT(hospital) DO UPDATE SET
                available_doctors = excluded.available_doctors,
                minutes_per_patient = excluded.minutes_per_patient,
                productive_minutes_per_hour =
                    excluded.productive_minutes_per_hour
        """, (
            hospital,
            available_doctors,
            minutes_per_patient,
            productive_minutes_per_hour
        ))

        conn.commit()


def get_staffing_assumptions(hospital):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("""
            SELECT
                available_doctors,
                minutes_per_patient,
                productive_minutes_per_hour
            FROM staffing_assumptions
            WHERE hospital = ?
        """, (hospital,)).fetchone()

    if row is None:
        return 1, 20.0, 60.0

    return row


def calculate_doctors_required(
    expected_arrivals,
    minutes_per_patient,
    productive_minutes_per_hour
):
    if minutes_per_patient <= 0:
        raise ValueError("Minutes per patient must be positive.")

    if productive_minutes_per_hour <= 0:
        raise ValueError("Productive minutes must be positive.")

    workload = max(float(expected_arrivals), 0) * minutes_per_patient

    return math.ceil(
        workload / productive_minutes_per_hour
    )


# =========================================================
# 4. LOAD DATA
# =========================================================

@st.cache_data(show_spinner=False)
def load_history():
    if not HIST_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(
        HIST_FILE,
        parse_dates=["arrival_hour"]
    )

    df["arrival_count"] = pd.to_numeric(
        df["arrival_count"],
        errors="coerce"
    ).fillna(0)

    return df


def load_visits_sample():
    if not VISITS_FILE.exists():
        return pd.DataFrame()

    return pd.read_csv(
        VISITS_FILE,
        compression="gzip",
        nrows=10000
    )


@st.cache_resource(show_spinner=False)
def load_rf_model():
    if not MODEL_FILE.exists():
        return None

    return joblib.load(MODEL_FILE)


# =========================================================
# 5. BAYESIAN FORECAST
# =========================================================

def hourly_pattern(hist):
    x = hist.copy()

    x["hour"] = x["arrival_hour"].dt.hour
    x["dayofweek"] = x["arrival_hour"].dt.dayofweek

    overall = max(
        float(x["arrival_count"].mean()),
        0.05
    )

    by_slot = x.groupby(
        ["hour", "dayofweek"]
    )["arrival_count"].mean()

    by_hour = x.groupby(
        "hour"
    )["arrival_count"].mean()

    def factor(ts):
        value = by_slot.get(
            (ts.hour, ts.weekday()),
            np.nan
        )

        if pd.isna(value):
            value = by_hour.get(ts.hour, overall)

        return float(value / overall) if overall else 1.0

    return overall, factor


def bayesian_forecast(
    hist,
    live,
    hospital,
    horizon=12,
    capacity=20
):
    h = hist.loc[
        hist.hospital == hospital
    ].copy()

    if h.empty:
        return pd.DataFrame(), None

    mean_rate, factor_for = hourly_pattern(h)

    alpha0 = 2.0
    beta0 = alpha0 / max(mean_rate, 0.05)

    factors = np.array([
        factor_for(t)
        for t in h.arrival_hour
    ])

    alpha = alpha0 + float(h.arrival_count.sum())
    beta = beta0 + float(factors.sum())

    now = datetime.now().replace(
        minute=0,
        second=0,
        microsecond=0
    )

    current_live = 0

    elapsed_hours = max(
        (datetime.now() - now).total_seconds() / 3600,
        1 / 3600
    )

    if not live.empty:
        current_live = int((
            (live.hospital == hospital)
            & (live.arrival_time >= pd.Timestamp(now))
            & (
                live.arrival_time
                < pd.Timestamp(now + timedelta(hours=1))
            )
        ).sum())

    current_exposure = factor_for(now) * elapsed_hours

    alpha_live = alpha + current_live
    beta_live = beta + current_exposure

    rows = []

    for step in range(1, horizon + 1):
        future_hour = now + timedelta(hours=step)

        exposure = factor_for(future_hour)

        p = beta_live / (beta_live + exposure)

        mean = alpha_live * exposure / beta_live

        lower, upper = nbinom.ppf(
            [0.05, 0.95],
            alpha_live,
            p
        )

        overload_probability = float(
            nbinom.sf(int(capacity), alpha_live, p)
        )

        rows.append({
            "Forecast hour": future_hour,
            "Expected arrivals": mean,
            "5th percentile": int(lower),
            "95th percentile": int(upper),
            "P(arrivals > capacity)": overload_probability
        })

    summary = {
        "historical_mean": mean_rate,
        "posterior_rate": alpha_live / beta_live,
        "current_hour_arrivals": current_live,
        "alpha": alpha_live,
        "beta": beta_live,
        "current_exposure": current_exposure
    }

    return pd.DataFrame(rows), summary


# =========================================================
# 6. RANDOM FOREST NETWORK FORECAST
# =========================================================

def random_forest_forecast(history, horizon):
    saved = load_rf_model()

    if saved is None:
        return pd.DataFrame(), "Random Forest model file not found."

    if not isinstance(saved, dict):
        return pd.DataFrame(), "Unexpected saved model format."

    model = saved.get("baseline_model")

    if model is None:
        return pd.DataFrame(), "Baseline Random Forest model not found."

    baseline_features = saved.get(
        "baseline_features",
        ["hour", "day_of_week", "month", "day_of_year"]
    )

    last_hist_time = pd.to_datetime(
        history["arrival_hour"]
    ).max()

    start = last_hist_time.floor("h") + pd.Timedelta(hours=1)

    future_times = pd.date_range(
        start=start,
        periods=horizon,
        freq="h"
    )

    feature_values = {
        "hour": future_times.hour,
        "day_of_week": future_times.dayofweek,
        "month": future_times.month,
        "day_of_year": future_times.dayofyear
    }

    unknown = [
        f for f in baseline_features
        if f not in feature_values
    ]

    if unknown:
        return pd.DataFrame(), (
            "The saved baseline model expects unsupported features: "
            + ", ".join(unknown)
        )

    future_features = pd.DataFrame({
        f: feature_values[f]
        for f in baseline_features
    })

    estimates = np.maximum(
        model.predict(future_features),
        0
    )

    result = pd.DataFrame({
        "Forecast hour": future_times,
        "Estimated network arrivals": estimates
    })

    return result, None


# =========================================================
# 7. LIVE SIMULATION
# =========================================================

def make_live_record(
    con,
    hospital,
    source,
    department,
    age_band,
    transport
):
    add_arrival(
        con,
        hospital,
        department,
        source,
        age_band,
        transport
    )


# =========================================================
# 8. GLOBAL SETUP
# =========================================================

con = connect_db()
history = load_history()

if history.empty:
    st.title("🏥 Live Hospital Rush Prediction")

    st.error(
        "Historical hourly data is not prepared yet. "
        "Run your setup commands first."
    )

    st.code(
        "python generate_synthetic_data.py\n"
        "python data_processing.py\n"
        "python train_model.py\n"
        "streamlit run app.py",
        language="bash"
    )

    st.stop()

hospitals = sorted(
    history.hospital.dropna().unique().tolist()
)

if not hospitals:
    st.error("No hospital names found in historical data.")
    st.stop()


# =========================================================
# 9. SIDEBAR
# =========================================================

st.sidebar.title("🏥 Hospital Rush AI")

st.sidebar.caption(
    "Synthetic live simulation · Lahore demonstration"
)

selected_hospital = st.sidebar.selectbox(
    "Select hospital",
    hospitals
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Live Operations",
        "Bayesian Intelligence",
        "Random Forest Forecast",
        "Doctor Staffing",
        "Historical Insights",
        "Model Evaluation",
        "About"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Synthetic hospital arrivals. "
    "No real hospital feed is connected."
)


# =========================================================
# 10. AUTOMATIC SIMULATION
# =========================================================

if "auto_simulation" not in st.session_state:
    st.session_state.auto_simulation = False

if "last_sim_tick" not in st.session_state:
    st.session_state.last_sim_tick = -1

interval_seconds = st.sidebar.select_slider(
    "Auto-arrival interval",
    options=[5, 10, 15, 20, 30],
    value=10
)

patients_per_tick = st.sidebar.slider(
    "Average synthetic arrivals per tick",
    0.1,
    2.0,
    0.7,
    0.1
)

if st.session_state.auto_simulation:
    tick = st_autorefresh(
        interval=interval_seconds * 1000,
        key="hospital_auto_refresh"
    )

    if tick != st.session_state.last_sim_tick:
        st.session_state.last_sim_tick = tick

        rng = np.random.default_rng()

        generated = int(
            rng.poisson(patients_per_tick)
        )

        for _ in range(generated):
            make_live_record(
                con,
                selected_hospital,
                "Automatic simulation",
                "Emergency",
                str(rng.choice([
                    "Child",
                    "Adult",
                    "Older adult"
                ])),
                str(rng.choice([
                    "Ambulance",
                    "Walk-in",
                    "Private vehicle"
                ]))
            )

live_all = get_live(con)


# =========================================================
# 11. LIVE OPERATIONS
# =========================================================

if page == "Live Operations":

    st.title("🏥 Live Hospital Operations")

    st.caption(
        f"Selected facility: {selected_hospital} · "
        f"{datetime.now():%d %b %Y, %I:%M:%S %p}"
    )

    top1, top2, top3 = st.columns(3)

    with top1:
        if st.session_state.auto_simulation:

            st.success("Automatic simulation is running")

            if st.button("⏸ Pause automatic arrivals"):
                st.session_state.auto_simulation = False
                st.rerun()

        else:

            st.info("Automatic simulation is paused")

            if st.button("▶ Start automatic arrivals"):
                st.session_state.auto_simulation = True
                st.session_state.last_sim_tick = -1
                st.rerun()

    top2.metric(
        "Auto interval",
        f"{interval_seconds} seconds"
    )

    top3.metric(
        "Average arrivals / tick",
        f"{patients_per_tick:.1f}"
    )

    st.subheader("Register a synthetic patient")

    with st.form("manual_patient_form", clear_on_submit=True):

        c1, c2, c3 = st.columns(3)

        with c1:
            department = st.selectbox(
                "Department",
                ["Emergency"]
            )

            age_band = st.selectbox(
                "Age group",
                ["Child", "Adult", "Older adult", "Not recorded"]
            )

        with c2:
            transport = st.selectbox(
                "Arrival transport",
                ["Walk-in", "Ambulance", "Private vehicle"]
            )

        with c3:
            st.caption("A demonstration ID is generated automatically.")

        submitted = st.form_submit_button(
            "＋ Register patient",
            use_container_width=True
        )

        if submitted:
            make_live_record(
                con,
                selected_hospital,
                "Manual registration",
                department,
                age_band,
                transport
            )

            st.success("Synthetic patient registered.")
            st.rerun()

    live = get_live(con)

    hospital_live = (
        live[live.hospital == selected_hospital].copy()
        if not live.empty else live
    )

    now = pd.Timestamp.now()

    active = (
        hospital_live[
            (hospital_live.expected_departure > now)
            & (hospital_live.status != "Completed")
        ]
        if not hospital_live.empty else hospital_live
    )

    waiting = (
        int((hospital_live.status == "Waiting").sum())
        if not hospital_live.empty else 0
    )

    capacity = st.sidebar.number_input(
        "Illustrative bed / handling capacity",
        min_value=5,
        max_value=500,
        value=80,
        step=5
    )

    occupancy = min(
        100.0,
        len(active) / capacity * 100
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Live arrivals", len(hospital_live))
    m2.metric("Active simulated patients", len(active))
    m3.metric("Waiting queue", waiting)
    m4.metric("Illustrative capacity", f"{occupancy:.1f}%")

    st.progress(occupancy / 100)

    if occupancy >= 85:
        st.warning("High simulated occupancy.")
    elif waiting >= 10:
        st.warning("The simulated waiting queue is growing.")
    else:
        st.success("No demonstration capacity alert.")

    st.subheader("Recent arrivals")

    if hospital_live.empty:
        st.info("No live arrivals yet.")

    else:
        display = hospital_live.copy()

        display["arrival_time"] = (
            display.arrival_time.dt.strftime("%Y-%m-%d %H:%M:%S")
        )

        display["expected_departure"] = (
            display.expected_departure.dt.strftime("%Y-%m-%d %H:%M")
        )

        st.dataframe(
            display[
                [
                    "id",
                    "arrival_time",
                    "source",
                    "department",
                    "age_band",
                    "transport",
                    "status",
                    "expected_departure"
                ]
            ].head(100),
            use_container_width=True,
            hide_index=True
        )

        ids = hospital_live.id.tolist()

        selected_id = st.selectbox(
            "Patient ID",
            ids
        )

        new_status = st.selectbox(
            "New status",
            ["Waiting", "In service", "Completed"]
        )

        if st.button("Update patient status"):
            con.execute(
                "UPDATE live_arrivals SET status=? WHERE id=?",
                (new_status, selected_id)
            )

            con.commit()
            st.rerun()

    st.caption(
        "All arrivals and stay durations are synthetic demonstration data."
    )


# =========================================================
# 12. BAYESIAN INTELLIGENCE
# =========================================================

elif page == "Bayesian Intelligence":

    st.title("🧠 Bayesian Hospital Rush Intelligence")

    st.write(
        "Gamma–Poisson updating with a Negative Binomial "
        "posterior predictive distribution."
    )

    horizon = st.slider(
        "Forecast horizon (hours)",
        1, 24, 12,
        key="bayes_horizon"
    )

    capacity_threshold = st.number_input(
        "Illustrative arrivals threshold",
        1, 100, 10,
        key="bayes_capacity"
    )

    forecast, summary = bayesian_forecast(
        history,
        get_live(con),
        selected_hospital,
        horizon,
        capacity_threshold
    )

    if forecast.empty:
        st.warning("No historical observations for this hospital.")

    else:
        a, b, c = st.columns(3)

        a.metric(
            "Historical hourly mean",
            f"{summary['historical_mean']:.2f}"
        )

        b.metric(
            "Current-hour live arrivals",
            summary["current_hour_arrivals"]
        )

        c.metric(
            "Updated baseline rate",
            f"{summary['posterior_rate']:.2f}"
        )

        st.info(
            "Live synthetic registrations update the current-hour "
            "observation. Forecasts are based on the demonstration "
            "Gamma–Poisson assumptions."
        )

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=forecast["Forecast hour"],
            y=forecast["95th percentile"],
            mode="lines",
            line=dict(width=0),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=forecast["Forecast hour"],
            y=forecast["5th percentile"],
            mode="lines",
            fill="tonexty",
            line=dict(width=0),
            name="90% predictive interval"
        ))

        fig.add_trace(go.Scatter(
            x=forecast["Forecast hour"],
            y=forecast["Expected arrivals"],
            mode="lines+markers",
            name="Expected arrivals"
        ))

        fig.update_layout(
            title="Bayesian posterior predictive forecast",
            xaxis_title="Forecast hour",
            yaxis_title="Synthetic arrivals"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            forecast.round(2),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "This is an academic prototype, not a clinically validated model."
        )


# =========================================================
# 13. RANDOM FOREST FORECAST
# =========================================================

elif page == "Random Forest Forecast":

    st.title("🌳 Random Forest Network Forecast")

    horizon_rf = st.slider(
        "Network forecast horizon (hours)",
        1, 24, 12,
        key="rf_horizon"
    )

    rf_future, error = random_forest_forecast(
        history,
        horizon_rf
    )

    if error:
        st.warning(error)

    else:
        st.info(
            "This time-only baseline forecast represents the combined "
            "hospital network. It is not an individual hospital prediction. "
            "Future weather is not supplied to this baseline model."
        )

        st.metric(
            "Average estimated network arrivals",
            f"{rf_future['Estimated network arrivals'].mean():.2f}"
        )

        st.plotly_chart(
            px.line(
                rf_future,
                x="Forecast hour",
                y="Estimated network arrivals",
                markers=True,
                title="Random Forest network forecast"
            ),
            use_container_width=True
        )

        st.dataframe(
            rf_future.round(2),
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# 14. DOCTOR STAFFING REQUIREMENT PREDICTION
# =========================================================

elif page == "Doctor Staffing":

    st.title("👨‍⚕️ Doctor Staffing Requirement Prediction")

    st.write(
        "Estimate doctor-equivalent workload for each hospital "
        "using Bayesian forecasts and Random Forest network allocations."
    )

    st.warning(
        "Educational demonstration only. These estimates are not "
        "clinical staffing standards and must not be used for real "
        "hospital staffing decisions."
    )

    st.subheader("1. Staffing assumptions")

    available, minutes, productive = get_staffing_assumptions(
        selected_hospital
    )

    with st.form("staffing_assumptions_form"):

        available_doctors = st.number_input(
            "Currently available doctors",
            min_value=0,
            max_value=500,
            value=int(available),
            step=1
        )

        minutes_per_patient = st.number_input(
            "Illustrative doctor minutes per patient",
            min_value=1.0,
            max_value=240.0,
            value=float(minutes),
            step=1.0
        )

        productive_minutes = st.number_input(
            "Productive minutes per doctor per hour",
            min_value=1.0,
            max_value=60.0,
            value=float(productive),
            step=1.0
        )

        save = st.form_submit_button(
            "Save assumptions for this hospital"
        )

        if save:
            save_staffing_assumptions(
                selected_hospital,
                available_doctors,
                minutes_per_patient,
                productive_minutes
            )

            st.success(
                f"Assumptions saved for {selected_hospital}."
            )

    st.divider()

    horizon_staff = st.slider(
        "Staffing forecast horizon (hours)",
        1, 24, 12,
        key="staffing_horizon"
    )

    st.subheader("2. Bayesian hospital-wise staffing estimates")

    bayes_rows = []

    live = get_live(con)

    for hospital in hospitals:

        forecast, summary = bayesian_forecast(
            history,
            live,
            hospital,
            horizon_staff
        )

        if forecast.empty:
            continue

        av, mins, prod = get_staffing_assumptions(hospital)

        for _, row in forecast.iterrows():

            required = calculate_doctors_required(
                row["Expected arrivals"],
                mins,
                prod
            )

            bayes_rows.append({
                "Hospital": hospital,
                "Forecast hour": row["Forecast hour"],
                "Expected arrivals": row["Expected arrivals"],
                "Estimated doctors required": required,
                "Available doctors": av,
                "Estimated workload gap": max(required - av, 0)
            })

    bayes_staff = pd.DataFrame(bayes_rows)

    if bayes_staff.empty:
        st.warning("No Bayesian staffing estimates are available.")

    else:
        selected_bayes = bayes_staff[
            bayes_staff.Hospital == selected_hospital
        ].copy()

        st.write(f"Selected hospital: **{selected_hospital}**")

        st.dataframe(
            selected_bayes.round(2),
            use_container_width=True,
            hide_index=True
        )

        if not selected_bayes.empty:

            st.plotly_chart(
                px.line(
                    selected_bayes,
                    x="Forecast hour",
                    y="Estimated doctors required",
                    markers=True,
                    title="Bayesian estimated doctor workload"
                ),
                use_container_width=True
            )

    st.divider()

    st.subheader("3. Random Forest hospital-wise staffing estimates")

    rf_future, rf_error = random_forest_forecast(
        history,
        horizon_staff
    )

    if rf_error:
        st.warning(rf_error)

    else:

        hospital_totals = history.groupby(
            "hospital"
        )["arrival_count"].sum()

        network_total = float(hospital_totals.sum())

        if network_total <= 0:
            st.warning("Historical hospital shares cannot be calculated.")

        else:

            shares = hospital_totals / network_total

            rf_staff_rows = []

            for _, row in rf_future.iterrows():

                for hospital in hospitals:

                    share = float(shares.get(hospital, 0))

                    allocated_arrivals = (
                        row["Estimated network arrivals"] * share
                    )

                    av, mins, prod = get_staffing_assumptions(
                        hospital
                    )

                    required = calculate_doctors_required(
                        allocated_arrivals,
                        mins,
                        prod
                    )

                    rf_staff_rows.append({
                        "Hospital": hospital,
                        "Forecast hour": row["Forecast hour"],
                        "Network arrivals": row["Estimated network arrivals"],
                        "Historical hospital share": share,
                        "Allocated arrivals": allocated_arrivals,
                        "Estimated doctors required": required,
                        "Available doctors": av,
                        "Estimated workload gap": max(required - av, 0)
                    })

            rf_staff = pd.DataFrame(rf_staff_rows)

            selected_rf = rf_staff[
                rf_staff.Hospital == selected_hospital
            ].copy()

            st.info(
                "Important: the Random Forest predicts network-wide "
                "arrivals. Hospital estimates below are allocated using "
                "each hospital's share of historical synthetic arrivals. "
                "They are not separate hospital-specific RF predictions."
            )

            st.dataframe(
                selected_rf.round(3),
                use_container_width=True,
                hide_index=True
            )

            st.plotly_chart(
                px.line(
                    selected_rf,
                    x="Forecast hour",
                    y="Estimated doctors required",
                    markers=True,
                    title="Random Forest allocated staffing estimate"
                ),
                use_container_width=True
            )

    st.caption(
        "The formula estimates doctor-equivalent workload as "
        "ceil(expected arrivals × assumed minutes per patient ÷ "
        "productive minutes per hour). Results depend on user-entered "
        "assumptions and synthetic forecasts."
    )


# =========================================================
# 15. HISTORICAL INSIGHTS
# =========================================================

elif page == "Historical Insights":

    st.title("📈 Historical Synthetic Data Insights")

    selected = history[
        history.hospital == selected_hospital
    ].copy()

    k1, k2, k3 = st.columns(3)

    k1.metric(
        "Historical hourly records",
        f"{len(selected):,}"
    )

    k2.metric(
        "Total synthetic arrivals",
        f"{int(selected.arrival_count.sum()):,}"
    )

    k3.metric(
        "Mean arrivals / hour",
        f"{selected.arrival_count.mean():.2f}"
    )

    selected["hour"] = selected.arrival_hour.dt.hour

    avg_hour = selected.groupby(
        "hour",
        as_index=False
    ).arrival_count.mean()

    st.plotly_chart(
        px.line(
            avg_hour,
            x="hour",
            y="arrival_count",
            markers=True,
            title="Average arrivals by hour"
        ),
        use_container_width=True
    )

    selected["weekday"] = (
        selected.arrival_hour.dt.day_name()
    )

    weekday = selected.groupby(
        "weekday",
        as_index=False
    ).arrival_count.mean()

    order = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]

    weekday["weekday"] = pd.Categorical(
        weekday.weekday,
        categories=order,
        ordered=True
    )

    weekday = weekday.sort_values("weekday")

    st.plotly_chart(
        px.bar(
            weekday,
            x="weekday",
            y="arrival_count",
            title="Average arrivals by weekday"
        ),
        use_container_width=True
    )

    st.caption(
        "These patterns were created by the synthetic data generator."
    )


# =========================================================
# 16. MODEL EVALUATION
# =========================================================

elif page == "Model Evaluation":

    st.title("🌳 Model Evaluation")

    comparison_file = OUTPUT_DIR / "random_forest_comparison.csv"
    evaluation_file = OUTPUT_DIR / "model_evaluation.csv"

    if comparison_file.exists():

        st.subheader("Random Forest comparison")

        st.dataframe(
            pd.read_csv(comparison_file),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("Random Forest comparison file not found.")

    if evaluation_file.exists():

        st.subheader("Chronological holdout details")

        st.dataframe(
            pd.read_csv(evaluation_file),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("""
    **Bayesian model:** The application displays posterior predictive
    estimates and uncertainty intervals.

    **Random Forest model:** The baseline uses time-based features.

    For a fair comparison, both models should be evaluated on the same
    held-out period using MAE, RMSE and predictive interval coverage.
    """)

    st.warning(
        "Historical weather is not currently used as a predictor in "
        "the synthetic arrival generator or the time-only baseline "
        "forecast. Do not interpret the results as weather-aware "
        "hospital predictions."
    )


# =========================================================
# 17. ABOUT
# =========================================================

elif page == "About":

    st.title("About this academic prototype")

    st.markdown("""
    **Project:** Live Hospital Rush Prediction using Random Forest
    and Bayesian Gamma–Poisson updating.

    **Historical data:** Synthetic hospital arrivals across
    demonstration hospitals.

    **Live input modes:**
    - Manual synthetic patient registration.
    - Accelerated automatic synthetic arrival simulation.

    **Bayesian method:** Gamma prior updating and Negative
    Binomial posterior predictive distribution.

    **Random Forest:** Time-based network-level baseline forecast.

    **Doctor staffing:** Illustrative doctor-equivalent workload
    estimates using both forecasting approaches.

    **Persistence:** SQLite stores demonstration arrivals,
    queue statuses and hospital-specific staffing assumptions.

    **Limitations:** No real hospital feed, real patient records,
    validated staffing standards, or clinically validated model
    is connected. The application is for academic demonstration
    and software development only.
    """)


# =========================================================
# 18. FOOTER
# =========================================================

st.divider()

st.caption(
    "Lahore Hospital Rush Intelligence System · "
    "Academic synthetic-data prototype"
)
