import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ADRA Smart Grid Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

TEST_FILE = Path("data/processed/test.csv")
MODEL_FILE = Path("results/models/adra_model.pkl")
TABLE_FOLDER = Path("results/tables")
GRAPH_FOLDER = Path("results/graphs")


# ============================================================
# UTILS
# ============================================================

GRAPH_DESCRIPTIONS = {
    "energy_consumption_timeseries.png": "Tracks the overall smart-grid demand pattern over time and highlights temporal load variations.",
    "energy_consumption_distribution.png": "Shows how electricity consumption is distributed across the dataset and whether the load is skewed or concentrated.",
    "hourly_consumption.png": "Captures the typical hourly demand profile, useful for peak-load and demand-response analysis.",
    "daily_consumption.png": "Summarizes daily average consumption to reveal persistent operational patterns and seasonality effects.",
    "weekday_vs_weekend.png": "Compares behavioral demand across weekdays and weekends to highlight usage habit differences.",
    "time_period_consumption.png": "Breaks energy demand by night, morning, afternoon, and evening periods for better forecasting context.",
    "temperature_vs_consumption.png": "Examines the relationship between ambient temperature and electricity demand, a key operational factor.",
    "anomaly_consumption.png": "Distinguishes normal and abnormal consumption patterns to support anomaly detection and validation.",
    "adra_energy_classification.png": "Shows the final ADRA-based classification of consumption states across the test set.",
    "adra_probability_analysis.png": "Displays the fitted probability analysis used to rank distributions and justify the selected model.",
    "adra_test_distribution.png": "Illustrates the predicted distribution behavior of the smart-grid test observations.",
    "probability_distribution_comparison.png": "Compares the goodness-of-fit statistics across candidate probability distributions for selection.",
}

TABLE_DESCRIPTIONS = {
    "table_1_dataset_statistics.csv": "Overview of the dataset size, coverage, and key structural properties used in the experiment.",
    "table_2_descriptive_statistics.csv": "Descriptive statistics summarizing the central tendency and variability of smart-grid variables.",
    "table_3_distribution_parameters.csv": "Estimated parameters for each candidate distribution considered during model fitting.",
    "table_4_goodness_of_fit.csv": "Goodness-of-fit metrics such as KS statistics used to compare distribution candidates.",
    "table_5_adra_ranking.csv": "Final ADRA ranking that orders the best-fitting distributions according to model score.",
    "table_6_test_evaluation.csv": "Evaluation of the trained model against the test dataset for performance and reliability insights.",
    "table_7_baseline_comparison.csv": "Comparison of ADRA results with baseline methods to confirm the competitive quality of the approach.",
}


st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(180deg, #f6fbff 0%, #eef7ff 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .hero {
            background: linear-gradient(135deg, #0b3d91 0%, #1c7ad6 100%);
            padding: 2rem 2rem 1.5rem 2rem;
            border-radius: 1.2rem;
            box-shadow: 0 8px 30px rgba(12, 60, 135, 0.18);
            margin-bottom: 1.5rem;
        }
        .hero h1 {
            color: white !important;
            margin-bottom: 0.2rem;
        }
        .hero p {
            color: rgba(255,255,255,0.9) !important;
            margin-bottom: 0;
        }
        .section-card {
            background: rgba(255,255,255,0.85);
            border: 1px solid rgba(15, 76, 140, 0.08);
            border-radius: 0.9rem;
            padding: 1rem 1rem 0.2rem 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 6px 18px rgba(8, 38, 74, 0.06);
        }
        .metric-box {
            background: white;
            border: 1px solid rgba(10, 50, 90, 0.08);
            border-radius: 0.9rem;
            padding: 1rem;
            box-shadow: 0 6px 18px rgba(8, 38, 74, 0.05);
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.3rem;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_test_data():
    df = pd.read_csv(TEST_FILE)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


@st.cache_resource
def load_model():
    with open(MODEL_FILE, "rb") as file:
        return pickle.load(file)


@st.cache_data
def load_table(filename: str):
    return pd.read_csv(TABLE_FOLDER / filename)


# ============================================================
# BUILDER HELPERS
# ============================================================

def classification_counts(df: pd.DataFrame) -> pd.DataFrame:
    if "Energy_Classification" in df.columns:
        counts = df["Energy_Classification"].value_counts().reset_index()
        counts.columns = ["Classification", "Count"]
        return counts

    prediction_df = load_table("adra_prediction_results.csv")
    counts = prediction_df["Energy_Classification"].value_counts().reset_index()
    counts.columns = ["Classification", "Count"]
    return counts


# ============================================================
# PAGE CONTENT
# ============================================================

df = load_test_data()
model = load_model()

st.markdown(
    """
    <div class="hero">
        <h1>ADRA Smart Grid Energy Analytics</h1>
        <p>Adaptive Distribution Ranking Algorithm for modeling and analyzing electricity consumption behavior.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class='section-card'>
        <h3 style='margin-top:0;'>Project overview</h3>
        <p>This dashboard combines the smart-grid demand profile, the fitted probability distribution model, and the experimental result tables into one analytical view.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cols = st.columns(4)
with metric_cols[0]:
    st.metric("Test records", len(df))
with metric_cols[1]:
    st.metric("Best distribution", model["selected_distribution"])
with metric_cols[2]:
    st.metric("ADRA score", f"{model['adra_score']:.6f}")
with metric_cols[3]:
    st.metric("ADRA rank", model["adra_rank"])

st.sidebar.title("ADRA Dashboard")
st.sidebar.caption("Smart-grid analytics and distribution-ranking analysis")
st.sidebar.markdown(
    """
    - Explore demand patterns over time
    - Compare distributions and fit quality
    - Review experiment tables and results
    - Track consumption anomalies and trends
    """
)

st.sidebar.markdown("### Model summary")
st.sidebar.write(f"Selected distribution: {model['selected_distribution']}")
st.sidebar.write(f"ADRA score: {model['adra_score']:.6f}")
st.sidebar.write(f"Rank: {model['adra_rank']}")

st.divider()

# ---------- Core time-series section ----------
with st.container():
    st.subheader("1. Energy consumption pattern")
    st.caption("The time-series view reveals load variability and how demand evolves across the test period.")

    col_left, col_right = st.columns(2)

    with col_left:
        line_fig = px.line(
            df,
            x="Timestamp",
            y="Electricity_Consumed",
            title="Electricity consumption over time",
            color_discrete_sequence=["#1f77b4"],
        )
        line_fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Electricity consumed",
            template="plotly_white",
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(line_fig, use_container_width=True)

    with col_right:
        hist_fig = px.histogram(
            df,
            x="Electricity_Consumed",
            nbins=40,
            title="Consumption distribution",
            color_discrete_sequence=["#4dabf7"],
        )
        hist_fig.update_layout(
            xaxis_title="Electricity consumed",
            yaxis_title="Frequency",
            template="plotly_white",
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(hist_fig, use_container_width=True)

# ---------- Classification and ranking ----------
with st.container():
    st.subheader("2. ADRA classification and distribution fit")
    st.caption("The classification view shows how the model segments the demand profile, while the fit comparison ranks the best-performing probability distributions.")

    class_df = classification_counts(df)
    class_fig = px.bar(
        class_df,
        x="Classification",
        y="Count",
        title="Energy classification counts",
        color="Classification",
        color_discrete_sequence=["#2ca02c", "#ff7f0e", "#1f77b4", "#d62728"],
    )
    class_fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=30, b=10))

    adra_df = load_table("table_5_adra_ranking.csv")
    gof_df = load_table("table_4_goodness_of_fit.csv")

    fit_fig = px.bar(
        gof_df,
        x="Distribution",
        y="KS_Statistic",
        title="KS statistic comparison",
        color="Distribution",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fit_fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=30, b=10), xaxis_tickangle=-25)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(class_fig, use_container_width=True)
    with col2:
        st.plotly_chart(fit_fig, use_container_width=True)

    st.markdown("#### ADRA ranking table")
    st.dataframe(adra_df, use_container_width=True, hide_index=True)

# ---------- Advanced demand insights ----------
with st.container():
    st.subheader("3. Additional consumption insights")
    st.caption("These views support operational decision-making by exposing hourly load, day-type differences, weather sensitivity, and anomaly behavior.")

    hourly_df = df.groupby("Hour")["Electricity_Consumed"].mean().reset_index()
    hourly_fig = px.line(
        hourly_df,
        x="Hour",
        y="Electricity_Consumed",
        title="Average hourly electricity consumption",
        markers=True,
    )
    hourly_fig.update_layout(xaxis_title="Hour of day", yaxis_title="Average consumption", template="plotly_white")

    day_type_df = df.groupby("Day_Type")["Electricity_Consumed"].mean().reset_index()
    day_type_fig = px.bar(
        day_type_df,
        x="Day_Type",
        y="Electricity_Consumed",
        title="Weekday vs weekend consumption",
        color="Day_Type",
        color_discrete_sequence=["#5B8FF9", "#FF7F50"],
    )
    day_type_fig.update_layout(xaxis_title="Day type", yaxis_title="Average consumption", template="plotly_white")

    time_period_order = ["Night", "Morning", "Afternoon", "Evening"]
    period_df = (
        df.groupby("Time_Period")["Electricity_Consumed"]
        .mean()
        .reindex(time_period_order)
        .reset_index()
    )
    period_df.columns = ["Time_Period", "Average_Consumption"]
    period_fig = px.bar(
        period_df,
        x="Time_Period",
        y="Average_Consumption",
        title="Consumption by time period",
        color="Time_Period",
        color_discrete_sequence=["#6c5ce7", "#00b894", "#fdcb6e", "#e17055"],
    )
    period_fig.update_layout(xaxis_title="Time period", yaxis_title="Average consumption", template="plotly_white")

    temp_fig = px.scatter(
        df,
        x="Temperature",
        y="Electricity_Consumed",
        color="Time_Period",
        title="Temperature vs electricity consumption",
        opacity=0.7,
    )
    temp_fig.update_layout(template="plotly_white")

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(hourly_fig, use_container_width=True)
        st.plotly_chart(day_type_fig, use_container_width=True)
    with col_b:
        st.plotly_chart(period_fig, use_container_width=True)
        st.plotly_chart(temp_fig, use_container_width=True)

# ---------- Graph gallery ----------
with st.container():
    st.subheader("4. Graph gallery")
    st.caption("These project visuals summarize the Smart Grid analysis and can be used in reports or presentations.")

    files = sorted(GRAPH_FOLDER.glob("*.png"))
    if files:
        columns = st.columns(2)
        for idx, image_path in enumerate(files):
            with columns[idx % 2]:
                image_name = image_path.name
                description = GRAPH_DESCRIPTIONS.get(image_name, "Project graph summary.")
                st.image(str(image_path), caption=image_name)
                st.caption(description)
    else:
        st.info("No graph images are available yet. Generate them from the analysis workflow.")

# ---------- Experimental results tables ----------
with st.container():
    st.subheader("5. Experimental result tables")
    st.caption("Each table below captures a key analytical output from the ADRA workflow and related comparison work.")

    table_files = [
        "table_1_dataset_statistics.csv",
        "table_2_descriptive_statistics.csv",
        "table_3_distribution_parameters.csv",
        "table_4_goodness_of_fit.csv",
        "table_5_adra_ranking.csv",
        "table_6_test_evaluation.csv",
        "table_7_baseline_comparison.csv",
    ]

    for filename in table_files:
        file_path = TABLE_FOLDER / filename
        if file_path.exists():
            with st.expander(f"{filename}", expanded=False):
                st.caption(TABLE_DESCRIPTIONS.get(filename, "Experimental output table."))
                st.dataframe(load_table(filename), use_container_width=True, hide_index=True)

st.divider()
st.caption("Adaptive Distribution Ranking Algorithm (ADRA) for Smart Grid Energy Consumption Modeling")
