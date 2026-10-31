
from pathlib import Path
import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
API_URL = "http://127.0.0.1:8000"

FEATURE_DIR = ROOT / "data" / "features"
MODEL_DIR = ROOT / "models"

st.set_page_config(
    page_title="AeroSentinel | UAV Intelligence",
    page_icon="🛩️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PROFESSIONAL UI THEME
# ============================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(ellipse at 15% 0%, #142b42 0%, transparent 38%),
        linear-gradient(135deg, #080f1b 0%, #0b1422 55%, #101d2d 100%);
    color: #e8f0f8;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 1rem;
    max-width: 1500px;
}

section[data-testid="stSidebar"] {
    background: #0b1523;
    border-right: 1px solid #24364a;
}

h1, h2, h3 {
    color: #f1f7ff !important;
    font-weight: 700 !important;
}

p, label, span {
    color: #c5d2e0;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #142438, #101c2c);
    border: 1px solid #263e56;
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 5px 18px rgba(0,0,0,0.18);
}

[data-testid="stMetricLabel"] {
    color: #91a9bf !important;
}

[data-testid="stMetricValue"] {
    color: #eaf8ff !important;
    font-weight: 700;
}

.stButton > button {
    background: linear-gradient(90deg, #087eaa, #08a6bb);
    color: white;
    border: none;
    border-radius: 9px;
    padding: 0.55rem 1.2rem;
    font-weight: 650;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #069bc8, #13c4d4);
    color: white;
}

hr {
    border-color: #263b51;
}

.hero {
    background: linear-gradient(115deg, #142b42, #102238 65%, #12394a);
    border: 1px solid #28506a;
    border-radius: 18px;
    padding: 28px 32px;
    margin-bottom: 25px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.2);
}

.hero-title {
    color: #f2f9ff;
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: 1px;
}

.hero-subtitle {
    color: #9ec4d9;
    font-size: 1rem;
    margin-top: 7px;
}

.eyebrow {
    color: #25c4d9;
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.section-heading {
    color: #eaf5ff;
    font-size: 1.15rem;
    font-weight: 700;
    margin: 15px 0;
}

.status-card {
    background: #122337;
    border: 1px solid #2a425b;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
}

.footer {
    border-top: 1px solid #263b51;
    margin-top: 55px;
    padding: 22px 8px 12px 8px;
    text-align: center;
    color: #8196aa;
    font-size: 0.82rem;
}

.footer strong {
    color: #20bfd2;
}

.footer-brand {
    color: #e5f2fc;
    font-size: 1rem;
    font-weight: 800;
    letter-spacing: 1.5px;
    margin-bottom: 7px;
}

.footer-small {
    color: #71879c;
    font-size: 0.74rem;
    margin-top: 6px;
}

#MainMenu {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA AND API FUNCTIONS
# ============================================================

@st.cache_data(ttl=5)
def get_health():
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=3
        )
        response.raise_for_status()
        return response.json()

    except requests.RequestException:
        return None


@st.cache_data
def load_features(hardware):
    path = FEATURE_DIR / f"{hardware}_window_features.parquet"

    if not path.exists():
        return pd.DataFrame()

    return pd.read_parquet(path)


@st.cache_data
def load_forecast_data():
    path = FEATURE_DIR / "holybro_future_remaining_time_30s.parquet"

    if not path.exists():
        return pd.DataFrame()

    return pd.read_parquet(path)


@st.cache_resource
def load_model(path):
    return joblib.load(path)


def send_prediction(
    endpoint,
    hardware,
    features,
    flight_id=None,
    window_index=None
):
    clean_features = {}

    for key, value in features.items():
        try:
            number = float(value)

            clean_features[key] = (
                number if pd.notna(number) else None
            )

        except (ValueError, TypeError):
            clean_features[key] = None

    payload = {
        "hardware_config": hardware,
        "features": clean_features,
        "flight_id": (
            str(flight_id)
            if flight_id is not None else None
        ),
        "window_index": window_index
    }

    try:
        response = requests.post(
            f"{API_URL}{endpoint}",
            json=payload,
            timeout=30
        )

        response.raise_for_status()
        return response.json()

    except requests.RequestException as error:
        st.error(f"API request failed: {error}")
        return None


def show_hero(
    title,
    subtitle,
    eyebrow="AEROSENTINEL INTELLIGENCE PLATFORM"
):
    st.markdown(f"""
    <div class="hero">
        <div class="eyebrow">{eyebrow}</div>
        <div class="hero-title">{title}</div>
        <div class="hero-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def show_footer():
    st.markdown("""
    <div class="footer">
        <div class="footer-brand">AEROSENTINEL</div>
        <div>
            AI-Powered UAV Anomaly Detection & Flight Intelligence
        </div>
        <div style="margin-top: 12px;">
            Developed by <strong>Sagar Karosiya</strong>
        </div>
        <div class="footer-small">
            © 2026 AeroSentinel. All Rights Reserved.
        </div>
        <div class="footer-small">
            Research & Development Project · UAV Intelligence System
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.markdown("""
<div style="text-align:center; padding: 8px 0 24px 0;">
    <div style="font-size:2.5rem;">🛩️</div>
    <div style="
        color:#eaf6ff;
        font-size:1.35rem;
        font-weight:850;
        letter-spacing:2px;">
        AEROSENTINEL
    </div>
    <div style="
        color:#27bfd4;
        font-size:0.7rem;
        letter-spacing:2px;">
        UAV INTELLIGENCE
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Anomaly Detection",
        "Remaining-Time Forecast",
        "Prediction History",
        "Model Status"
    ]
)

st.sidebar.markdown("---")

health = get_health()

if health:
    st.sidebar.success("● API SYSTEM ONLINE")
else:
    st.sidebar.error("● API SYSTEM OFFLINE")

st.sidebar.markdown("""
<div style="
    color:#71879c;
    font-size:0.75rem;
    padding-top:20px;
    line-height:1.8;">
    <b style="color:#a9c2d5;">AEROSENTINEL v1.1</b><br>
    Machine Learning Engine<br>
    Flight Intelligence System<br><br>
    Developer: Sagar Karosiya
</div>
""", unsafe_allow_html=True)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    show_hero(
        "UAV INTELLIGENCE CENTER",
        "Monitor aircraft telemetry, detect unusual flight patterns "
        "and forecast future battery estimates."
    )

    holybro = load_features("holybro")
    speedybee = load_features("speedybee")
    forecast = load_forecast_data()

    st.markdown(
        '<div class="section-heading">SYSTEM OVERVIEW</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Usable Flights", "228", "Dataset total")
    c2.metric("Holybro Flights", "118", "PX4 hardware")
    c3.metric("SpeedyBee Flights", "110", "INAV hardware")
    c4.metric("Forecast Horizon", "30 sec", "Future BMS estimate")

    st.markdown("---")

    left, right = st.columns([1.25, 1])

    with left:

        st.markdown(
            '<div class="section-heading">HARDWARE DISTRIBUTION</div>',
            unsafe_allow_html=True
        )

        distribution = pd.DataFrame({
            "Hardware": ["Holybro", "SpeedyBee"],
            "Flights": [118, 110]
        })

        fig = px.pie(
            distribution,
            names="Hardware",
            values="Flights",
            hole=0.62,
            template="plotly_dark"
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=20, b=20, l=10, r=10),
            legend_title_text="Hardware"
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:

        st.markdown(
            '<div class="section-heading">PROCESSED FEATURE WINDOWS</div>',
            unsafe_allow_html=True
        )

        st.metric("Holybro Windows", f"{len(holybro):,}")
        st.metric("SpeedyBee Windows", f"{len(speedybee):,}")
        st.metric("Forecast Windows", f"{len(forecast):,}")

    st.markdown("---")

    st.markdown(
        '<div class="section-heading">PLATFORM CAPABILITIES</div>',
        unsafe_allow_html=True
    )

    a, b, c = st.columns(3)

    with a:
        st.markdown("### ◈ Anomaly Detection")
        st.write(
            "Isolation Forest models identify unusual telemetry patterns."
        )

    with b:
        st.markdown("### ◈ Predictive Intelligence")
        st.write(
            "Forecasts future BMS remaining-time estimates."
        )

    with c:
        st.markdown("### ◈ Multi-Hardware Support")
        st.write(
            "Separate anomaly models for Holybro and SpeedyBee."
        )

    st.info(
        "This dashboard currently operates on historical processed "
        "telemetry. Live UAV telemetry streaming has not yet been integrated."
    )


# ============================================================
# ANOMALY DETECTION
# ============================================================

elif page == "Anomaly Detection":

    show_hero(
        "ANOMALY DETECTION",
        "Analyze individual telemetry windows using trained "
        "Isolation Forest models."
    )

    hardware = st.selectbox(
        "Select UAV Hardware",
        ["holybro", "speedybee"],
        format_func=lambda x: x.upper()
    )

    df = load_features(hardware)

    if df.empty:

        st.warning("Feature dataset not found.")

    else:

        flights = sorted(df["flight_id"].unique())

        flight_id = st.selectbox(
            "Select Flight ID",
            flights
        )

        flight_df = df[
            df["flight_id"] == flight_id
        ].reset_index(drop=True)

        window_index = st.selectbox(
            "Select Telemetry Window",
            range(len(flight_df))
        )

        selected = flight_df.iloc[window_index]

        st.markdown("---")

        st.write(
            f"**Hardware:** {hardware.upper()}  \n"
            f"**Flight ID:** {flight_id}  \n"
            f"**Window Index:** {window_index}"
        )

        if st.button("RUN ANOMALY ANALYSIS", type="primary"):

            model_path = (
                MODEL_DIR / f"{hardware}_isolation_forest.joblib"
            )

            bundle = load_model(model_path)
            feature_columns = bundle["feature_columns"]

            features = selected.reindex(feature_columns)

            features = features.replace(
                [float("inf"), float("-inf")],
                None
            )

            result = send_prediction(
                "/predict/anomaly",
                hardware,
                features.to_dict(),
                flight_id=flight_id,
                window_index=window_index
            )

            if result and result.get("status") == "success":

                prediction = result["prediction"]

                score = prediction["anomaly_score"]
                threshold = prediction["threshold"]
                is_anomaly = prediction["is_anomaly"]

                st.markdown("---")

                c1, c2, c3 = st.columns(3)

                c1.metric("Anomaly Score", f"{score:.5f}")
                c2.metric("Detection Threshold", f"{threshold:.5f}")

                c3.metric(
                    "Detection Result",
                    "ANOMALY" if is_anomaly else "NORMAL"
                )

                if is_anomaly:
                    st.error("Unusual telemetry pattern detected.")
                else:
                    st.success("No anomaly flagged in this window.")

                st.caption(
                    f"Database record ID: {result.get('record_id', 'Unavailable')}"
                )

                score_chart = pd.DataFrame({
                    "Metric": ["Anomaly Score", "Threshold"],
                    "Value": [score, threshold]
                })

                fig = px.bar(
                    score_chart,
                    x="Metric",
                    y="Value",
                    color="Metric",
                    template="plotly_dark",
                    title="Anomaly Score vs Detection Threshold"
                )

                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )

                st.plotly_chart(fig, use_container_width=True)

                st.caption(
                    "Anomaly flags indicate unusual telemetry patterns, "
                    "not confirmed hardware faults."
                )


# ============================================================
# REMAINING-TIME FORECAST
# ============================================================

elif page == "Remaining-Time Forecast":

    show_hero(
        "PREDICTIVE FLIGHT INTELLIGENCE",
        "Forecast the future BMS remaining-time estimate using "
        "the Holybro forecasting model."
    )

    st.info(
        "Forecast target: future BMS remaining-time estimate "
        "approximately 30 seconds ahead. This is not a prediction "
        "of actual battery exhaustion or true end-of-flight time."
    )

    df = load_forecast_data()

    if df.empty:

        st.warning("Forecast dataset not found.")

    else:

        test_df = df[
            df["split"] == "test"
        ].reset_index(drop=True)

        if test_df.empty:

            st.warning("No test windows available.")

        else:

            flights = sorted(test_df["flight_id"].unique())

            flight_id = st.selectbox(
                "Select Holybro Test Flight",
                flights
            )

            flight_df = test_df[
                test_df["flight_id"] == flight_id
            ].reset_index(drop=True)

            window_index = st.selectbox(
                "Select Forecast Window",
                range(len(flight_df)),
                key="forecast_window"
            )

            selected = flight_df.iloc[window_index]

            if st.button("GENERATE FORECAST", type="primary"):

                model = load_model(
                    MODEL_DIR / "holybro_future_remaining_time_30s.joblib"
                )

                feature_columns = list(model.feature_names_in_)

                features = selected.reindex(feature_columns)

                result = send_prediction(
                    "/predict/remaining-time",
                    "holybro",
                    features.to_dict(),
                    flight_id=flight_id,
                    window_index=window_index
                )

                if result and result.get("status") == "success":

                    prediction = result["prediction"]

                    seconds = prediction[
                        "predicted_bms_remaining_time_seconds"
                    ]

                    minutes = seconds / 60

                    st.markdown("---")

                    c1, c2 = st.columns(2)

                    c1.metric(
                        "Predicted BMS Remaining Time",
                        f"{minutes:.2f} min"
                    )

                    c2.metric(
                        "Forecast Horizon",
                        f"{prediction['forecast_horizon_seconds']} sec"
                    )

                    st.markdown(
                        f"""
                        <div class="status-card">
                            <div class="eyebrow">MODEL OUTPUT</div>
                            <h2 style="color:#28c7d8;">
                                {seconds:,.2f} seconds
                            </h2>
                            <p>Predicted future BMS remaining-time estimate</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.caption(
                        f"Database record ID: {result.get('record_id', 'Unavailable')}"
                    )


# ============================================================
# PREDICTION HISTORY
# ============================================================

elif page == "Prediction History":

    show_hero(
        "PREDICTION HISTORY",
        "Review previously recorded model predictions and monitoring activity."
    )

    # Total database records
    try:

        count_response = requests.get(
            f"{API_URL}/monitoring/count",
            timeout=5
        )

        count_response.raise_for_status()

        total_records = count_response.json()["total_predictions"]

    except requests.RequestException:

        total_records = None

    st.markdown("### Monitoring Filters")

    col1, col2, col3 = st.columns(3)

    with col1:

        hardware_filter = st.selectbox(
            "Hardware",
            ["All", "Holybro", "SpeedyBee"],
            key="history_hardware"
        )

    with col2:

        type_filter = st.selectbox(
            "Prediction Type",
            ["All", "Anomaly", "Remaining Time"],
            key="history_type"
        )

    with col3:

        record_limit = st.selectbox(
            "Records to display",
            [50, 100, 250, 500, 1000, 5000],
            index=1
        )

    params = {
        "limit": record_limit
    }

    if hardware_filter != "All":
        params["hardware_config"] = hardware_filter.lower()

    if type_filter != "All":

        params["prediction_type"] = (
            "remaining_time"
            if type_filter == "Remaining Time"
            else "anomaly"
        )

    if st.button("REFRESH HISTORY"):

        st.rerun()

    try:

        response = requests.get(
            f"{API_URL}/monitoring/history",
            params=params,
            timeout=10
        )

        response.raise_for_status()

        records = response.json()["predictions"]

    except requests.RequestException as error:

        records = []
        st.error(f"Unable to retrieve prediction history: {error}")

    st.divider()

    anomaly_records = [
        r for r in records
        if r["prediction_type"] == "anomaly"
    ]

    forecast_records = [
        r for r in records
        if r["prediction_type"] == "remaining_time"
    ]

    anomaly_flags = [
        r for r in anomaly_records
        if r["is_anomaly"] == 1
    ]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Database Records",
        total_records if total_records is not None else "Unavailable"
    )

    c2.metric("Displayed Records", len(records))
    c3.metric("Anomaly Predictions", len(anomaly_records))
    c4.metric("Forecast Predictions", len(forecast_records))

    st.divider()

    # Anomaly statistics
    if anomaly_records:

        st.markdown("### Anomaly Monitoring")

        flagged = len(anomaly_flags)
        normal = len(anomaly_records) - flagged

        a, b = st.columns(2)

        a.metric("Unusual Windows Flagged", flagged)
        b.metric("Windows Not Flagged", normal)

        anomaly_chart = pd.DataFrame({
            "Detection": ["Anomaly", "Normal"],
            "Count": [flagged, normal]
        })

        fig = px.pie(
            anomaly_chart,
            names="Detection",
            values="Count",
            hole=0.55,
            template="plotly_dark",
            title="Recorded Anomaly Results"
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, use_container_width=True)

    # Forecast statistics
    if forecast_records:

        st.markdown("### Remaining-Time Forecast Records")

        forecast_values = [
            r["predicted_bms_remaining_time_seconds"]
            for r in forecast_records
            if r["predicted_bms_remaining_time_seconds"] is not None
        ]

        if forecast_values:

            st.metric(
                "Average Predicted BMS Remaining Time",
                f"{sum(forecast_values) / len(forecast_values):.2f} seconds"
            )

    st.divider()

    # Prediction table
    st.markdown("### Prediction Records")

    if records:

        history_df = pd.DataFrame(records)

        display_columns = [
            "id",
            "timestamp",
            "prediction_type",
            "hardware_config",
            "flight_id",
            "window_index",
            "anomaly_score",
            "threshold",
            "is_anomaly",
            "forecast_horizon_seconds",
            "predicted_bms_remaining_time_seconds"
        ]

        history_df = history_df[
            [c for c in display_columns if c in history_df.columns]
        ]

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="DOWNLOAD HISTORY AS CSV",
            data=csv_data,
            file_name="aerosentinel_prediction_history.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "No prediction records found for the selected filters. "
            "Run an anomaly detection or forecast to create records."
        )


# ============================================================
# MODEL STATUS
# ============================================================

elif page == "Model Status":

    show_hero(
        "MODEL & API STATUS",
        "Monitor the availability of AeroSentinel inference services."
    )

    if health:

        models = health.get("models_loaded", {})

        st.markdown(
            '<div class="section-heading">DEPLOYMENT STATUS</div>',
            unsafe_allow_html=True
        )

        for model_name, loaded in models.items():

            col1, col2 = st.columns([3, 1])

            col1.markdown(
                f"**{model_name.replace('_', ' ').title()}**"
            )

            if loaded:
                col2.success("LOADED")
            else:
                col2.error("NOT LOADED")

        st.divider()

        st.success("FastAPI backend is responding.")

    else:

        st.error("API is unavailable. Start the FastAPI server.")

    st.markdown(
        '<div class="section-heading">API ENDPOINTS</div>',
        unsafe_allow_html=True
    )

    st.code(
        "GET  /health\n"
        "POST /predict/anomaly\n"
        "POST /predict/remaining-time\n"
        "POST /predict\n"
        "GET  /monitoring/history\n"
        "GET  /monitoring/count",
        language="text"
    )

    st.markdown(
        "Interactive API documentation: "
        "[Open Swagger UI](http://127.0.0.1:8000/docs)"
    )


# ============================================================
# FOOTER
# ============================================================

show_footer()
