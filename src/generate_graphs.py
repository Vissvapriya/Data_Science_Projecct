import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = Path("data/processed/train.csv")

GRAPH_FOLDER = Path("results/graphs")


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\nLoading training dataset...")

    df = pd.read_csv(INPUT_FILE)

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    df = df.sort_values("Timestamp")

    print(f"Training records: {len(df)}")

    return df


# ============================================================
# SAVE FIGURE
# ============================================================

def save_figure(filename, fig=None):

    GRAPH_FOLDER.mkdir(parents=True, exist_ok=True)

    if fig is None:
        fig = plt.gcf()

    fig.tight_layout()
    fig.savefig(GRAPH_FOLDER / filename, dpi=300)
    plt.close(fig)

    print(f"Saved: {filename}")


# ============================================================
# GRAPH 1
# ENERGY CONSUMPTION DISTRIBUTION
# ============================================================

def energy_distribution(df):

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        df["Electricity_Consumed"],
        bins=40,
        edgecolor="black"
    )

    ax.set_xlabel("Electricity Consumption")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Electricity Consumption")
    ax.grid(alpha=0.3)

    save_figure("energy_consumption_distribution.png", fig)


# ============================================================
# GRAPH 2
# ENERGY CONSUMPTION TIME SERIES
# ============================================================

def energy_timeseries(df):

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        df["Timestamp"],
        df["Electricity_Consumed"],
        linewidth=0.8
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Electricity Consumption")
    ax.set_title("Smart Grid Electricity Consumption Over Time")
    ax.grid(alpha=0.3)

    save_figure("energy_consumption_timeseries.png", fig)


# ============================================================
# GRAPH 3
# HOURLY CONSUMPTION
# ============================================================

def hourly_consumption(df):

    hourly = (
        df.groupby("Hour")["Electricity_Consumed"]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        hourly.index,
        hourly.values,
        marker="o"
    )

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Electricity Consumption")
    ax.set_title("Average Hourly Electricity Consumption")
    ax.set_xticks(range(0, 24))
    ax.grid(alpha=0.3)

    save_figure("hourly_consumption.png", fig)


# ============================================================
# GRAPH 4
# DAILY CONSUMPTION
# ============================================================

def daily_consumption(df):

    daily = (
        df.groupby(df["Timestamp"].dt.date)
        ["Electricity_Consumed"]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        daily.index,
        daily.values
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Average Electricity Consumption")
    ax.set_title("Daily Average Electricity Consumption")
    plt.setp(ax.get_xticklabels(), rotation=45)
    ax.grid(alpha=0.3)

    save_figure("daily_consumption.png", fig)


# ============================================================
# GRAPH 5
# WEEKDAY VS WEEKEND
# ============================================================

def weekday_weekend(df):

    comparison = (
        df.groupby("Day_Type")
        ["Electricity_Consumed"]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.bar(
        comparison.index,
        comparison.values
    )

    ax.set_xlabel("Day Type")
    ax.set_ylabel("Average Electricity Consumption")
    ax.set_title("Weekday vs Weekend Electricity Consumption")
    ax.grid(axis="y", alpha=0.3)

    save_figure("weekday_vs_weekend.png", fig)


# ============================================================
# GRAPH 6
# TIME PERIOD CONSUMPTION
# ============================================================

def time_period_consumption(df):

    period_order = [
        "Night",
        "Morning",
        "Afternoon",
        "Evening"
    ]

    period_data = (
        df.groupby("Time_Period")
        ["Electricity_Consumed"]
        .mean()
        .reindex(period_order)
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.bar(
        period_data.index,
        period_data.values
    )

    ax.set_xlabel("Time Period")
    ax.set_ylabel("Average Electricity Consumption")
    ax.set_title("Electricity Consumption by Time Period")
    ax.grid(axis="y", alpha=0.3)

    save_figure("time_period_consumption.png", fig)


# ============================================================
# GRAPH 7
# TEMPERATURE VS CONSUMPTION
# ============================================================

def temperature_consumption(df):

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        df["Temperature"],
        df["Electricity_Consumed"],
        alpha=0.5
    )

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Electricity Consumption")
    ax.set_title("Temperature vs Electricity Consumption")
    ax.grid(alpha=0.3)

    save_figure("temperature_vs_consumption.png", fig)


# ============================================================
# GRAPH 8
# ANOMALY ANALYSIS
# ============================================================

def anomaly_analysis(df):

    anomaly_data = (
        df.groupby("Anomaly_Label")
        ["Electricity_Consumed"]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.bar(
        anomaly_data.index,
        anomaly_data.values
    )

    ax.set_xlabel("Anomaly Label")
    ax.set_ylabel("Average Electricity Consumption")
    ax.set_title("Normal vs Abnormal Energy Consumption")
    ax.grid(axis="y", alpha=0.3)

    save_figure("anomaly_consumption.png", fig)


# ============================================================
# GRAPH 9
# MONTHLY CONSUMPTION TREND
# ============================================================

def monthly_consumption_trend(df):

    monthly = (
        df.groupby(df["Timestamp"].dt.to_period("M"))
        ["Electricity_Consumed"]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        monthly.index.astype(str),
        monthly.values,
        marker="o"
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Average Electricity Consumption")
    ax.set_title("Monthly Electricity Consumption Trend")
    plt.setp(ax.get_xticklabels(), rotation=45)
    ax.grid(alpha=0.3)

    save_figure("monthly_consumption_trend.png", fig)


# ============================================================
# GRAPH 10
# PEAK LOAD ANALYSIS
# ============================================================

def peak_load_analysis(df):

    peak_df = (
        df.groupby("Peak_Flag")["Electricity_Consumed"]
        .mean()
        .reset_index()
    )

    peak_df.columns = ["Peak_Flag", "Average_Consumption"]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        peak_df["Peak_Flag"].astype(str),
        peak_df["Average_Consumption"]
    )

    ax.set_xlabel("Peak Flag")
    ax.set_ylabel("Average Electricity Consumption")
    ax.set_title("Peak vs Non-Peak Consumption Comparison")
    ax.grid(axis="y", alpha=0.3)

    save_figure("peak_load_analysis.png", fig)


# ============================================================
# GRAPH 11
# WEATHER-ENERGY RELATIONSHIP
# ============================================================

def weather_energy_relation(df):

    fig, ax = plt.subplots(figsize=(9, 6))

    scatter = ax.scatter(
        df["Temperature"],
        df["Humidity"],
        c=df["Electricity_Consumed"],
        cmap="viridis",
        alpha=0.7
    )

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Humidity")
    ax.set_title("Temperature-Humidity Relationship with Energy Demand")

    colorbar = plt.colorbar(scatter, ax=ax)
    colorbar.set_label("Electricity Consumed")

    save_figure("weather_energy_relationship.png", fig)


# ============================================================
# GRAPH 12
# ANOMALY LABEL DISTRIBUTION
# ============================================================

def anomaly_label_distribution(df):

    counts = df["Anomaly_Label"].value_counts().reset_index()
    counts.columns = ["Anomaly_Label", "Count"]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.bar(counts["Anomaly_Label"], counts["Count"])
    ax.set_xlabel("Anomaly Label")
    ax.set_ylabel("Count")
    ax.set_title("Anomaly Label Distribution")
    ax.grid(axis="y", alpha=0.3)

    save_figure("anomaly_label_distribution.png", fig)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print("ADRA - EXPLORATORY DATA ANALYSIS")

    print("=" * 70)

    # Create graph directory
    GRAPH_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load data
    df = load_data()

    print("\nGenerating graphs...\n")

    # Generate graphs
    energy_distribution(df)
    energy_timeseries(df)
    hourly_consumption(df)
    daily_consumption(df)
    weekday_weekend(df)
    time_period_consumption(df)
    temperature_consumption(df)
    anomaly_analysis(df)
    monthly_consumption_trend(df)
    peak_load_analysis(df)
    weather_energy_relation(df)
    anomaly_label_distribution(df)

    print("\n" + "=" * 70)

    print("ALL GRAPHS GENERATED SUCCESSFULLY")

    print("=" * 70)

    print(f"\nGraphs saved in:")
    print(GRAPH_FOLDER)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()