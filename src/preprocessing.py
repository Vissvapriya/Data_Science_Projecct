import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/raw/smart_meter_data (1).csv")

PROCESSED_FOLDER = Path("data/processed")

OUTPUT_FILE = PROCESSED_FOLDER / "cleaned_data.csv"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\nLoading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Dataset loaded successfully.")
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    return df


# ============================================================
# DATA PREPROCESSING
# ============================================================

def preprocess_data(df):

    print("\nStarting preprocessing...")


    # --------------------------------------------------------
    # 1. Convert Timestamp
    # --------------------------------------------------------

    print("\n1. Converting Timestamp...")

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce"
    )


    # --------------------------------------------------------
    # 2. Remove invalid timestamps
    # --------------------------------------------------------

    invalid_timestamp = df["Timestamp"].isna().sum()

    if invalid_timestamp > 0:

        print(
            f"Removing {invalid_timestamp} invalid timestamps..."
        )

        df = df.dropna(subset=["Timestamp"])

    else:

        print("No invalid timestamps found.")


    # --------------------------------------------------------
    # 3. Remove duplicate records
    # --------------------------------------------------------

    duplicate_count = df.duplicated().sum()

    print(f"\nDuplicate records found: {duplicate_count}")

    if duplicate_count > 0:

        df = df.drop_duplicates()

        print("Duplicate records removed.")

    else:

        print("No duplicate records found.")


    # --------------------------------------------------------
    # 4. Check missing values
    # --------------------------------------------------------

    print("\nChecking missing values...")

    missing_values = df.isnull().sum()

    print(missing_values)

    if missing_values.sum() > 0:

        print("Missing values detected.")

        # Numerical columns
        numerical_columns = [
            "Electricity_Consumed",
            "Temperature",
            "Humidity",
            "Wind_Speed",
            "Avg_Past_Consumption"
        ]

        for column in numerical_columns:

            if column in df.columns:

                df[column] = df[column].interpolate(
                    method="linear"
                )

        # Categorical column
        if "Anomaly_Label" in df.columns:

            df["Anomaly_Label"] = (
                df["Anomaly_Label"]
                .ffill()
                .bfill()
            )

    else:

        print("No missing values found.")


    # --------------------------------------------------------
    # 5. Sort by Timestamp
    # --------------------------------------------------------

    print("\nSorting data chronologically...")

    df = df.sort_values("Timestamp")

    df = df.reset_index(drop=True)


    # --------------------------------------------------------
    # 6. Validate Energy Consumption
    # --------------------------------------------------------

    print("\nValidating Electricity_Consumed...")

    invalid_energy = (
        (df["Electricity_Consumed"] < 0) |
        (df["Electricity_Consumed"] > 1)
    ).sum()

    print(f"Invalid energy values: {invalid_energy}")

    if invalid_energy > 0:

        df = df[
            (df["Electricity_Consumed"] >= 0) &
            (df["Electricity_Consumed"] <= 1)
        ]


    # --------------------------------------------------------
    # 7. Temporal Feature Engineering
    # --------------------------------------------------------

    print("\nCreating temporal features...")

    df["Year"] = df["Timestamp"].dt.year

    df["Month"] = df["Timestamp"].dt.month

    df["Day"] = df["Timestamp"].dt.day

    df["Hour"] = df["Timestamp"].dt.hour

    df["Minute"] = df["Timestamp"].dt.minute

    df["Day_of_Week"] = df["Timestamp"].dt.dayofweek

    df["Day_Name"] = df["Timestamp"].dt.day_name()


    # --------------------------------------------------------
    # 8. Weekday / Weekend
    # --------------------------------------------------------

    df["Day_Type"] = np.where(
        df["Day_of_Week"] < 5,
        "Weekday",
        "Weekend"
    )


    # --------------------------------------------------------
    # 9. Time Period
    # --------------------------------------------------------

    def classify_period(hour):

        if 0 <= hour < 6:
            return "Night"

        elif 6 <= hour < 12:
            return "Morning"

        elif 12 <= hour < 18:
            return "Afternoon"

        else:
            return "Evening"


    df["Time_Period"] = df["Hour"].apply(
        classify_period
    )


    # --------------------------------------------------------
    # 10. Peak Consumption Indicator
    # --------------------------------------------------------

    peak_threshold = df[
        "Electricity_Consumed"
    ].quantile(0.90)

    df["Peak_Threshold"] = peak_threshold

    df["Peak_Flag"] = (
        df["Electricity_Consumed"] >= peak_threshold
    ).astype(int)


    print(
        f"\n90th percentile peak threshold: "
        f"{peak_threshold:.6f}"
    )


    # --------------------------------------------------------
    # 11. Final Dataset Information
    # --------------------------------------------------------

    print("\nFinal dataset shape:")

    print(f"Rows    : {df.shape[0]}")

    print(f"Columns : {df.shape[1]}")


    return df


# ============================================================
# SAVE DATA
# ============================================================

def save_data(df):

    PROCESSED_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nCleaned dataset saved successfully.")

    print(f"Location: {OUTPUT_FILE}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ADRA - DATA PREPROCESSING")
    print("=" * 70)

    df = load_data()

    df = preprocess_data(df)

    save_data(df)

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()