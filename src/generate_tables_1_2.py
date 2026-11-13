import pandas as pd
from pathlib import Path


# ============================================================
# FILE PATHS
# ============================================================

RAW_FILE = Path(
    "data/raw/smart_meter_data (1).csv"
)

TRAIN_FILE = Path(
    "data/processed/train.csv"
)

TEST_FILE = Path(
    "data/processed/test.csv"
)

OUTPUT_FOLDER = Path(
    "results/tables"
)


# ============================================================
# LOAD DATA
# ============================================================

raw_df = pd.read_csv(RAW_FILE)

train_df = pd.read_csv(TRAIN_FILE)

test_df = pd.read_csv(TEST_FILE)


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TABLE 1 - DATASET STATISTICS
# ============================================================

normal_count = (
    raw_df["Anomaly_Label"]
    .value_counts()
    .get("Normal", 0)
)

abnormal_count = (
    raw_df["Anomaly_Label"]
    .value_counts()
    .get("Abnormal", 0)
)


table_1 = pd.DataFrame({

    "Parameter": [

        "Total Records",
        "Training Records",
        "Testing Records",
        "Number of Columns",
        "Numerical Features",
        "Time Interval",
        "Missing Values",
        "Duplicate Records",
        "Normal Records",
        "Abnormal Records"

    ],

    "Value": [

        len(raw_df),

        len(train_df),

        len(test_df),

        len(raw_df.columns),

        5,

        "30 minutes",

        int(raw_df.isnull().sum().sum()),

        int(raw_df.duplicated().sum()),

        normal_count,

        abnormal_count

    ]

})


table_1.to_csv(

    OUTPUT_FOLDER /
    "table_1_dataset_statistics.csv",

    index=False

)


# ============================================================
# TABLE 2 - DESCRIPTIVE STATISTICS
# ============================================================

numerical_columns = [

    "Electricity_Consumed",
    "Temperature",
    "Humidity",
    "Wind_Speed",
    "Avg_Past_Consumption"

]


table_2 = raw_df[
    numerical_columns
].describe().T


table_2 = table_2[
    [
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max"
    ]
]


table_2.columns = [

    "Mean",
    "Std",
    "Min",
    "25%",
    "Median",
    "75%",
    "Max"

]


table_2.index.name = "Variable"


table_2.to_csv(

    OUTPUT_FOLDER /
    "table_2_descriptive_statistics.csv"

)


# ============================================================
# DISPLAY
# ============================================================

print("=" * 80)

print("TABLE 1 - DATASET STATISTICS")

print("=" * 80)

print(
    table_1.to_string(index=False)
)


print("\n")

print("=" * 80)

print("TABLE 2 - DESCRIPTIVE STATISTICS")

print("=" * 80)

print(
    table_2.round(6).to_string()
)


print("\n")

print("=" * 80)

print("TABLES 1 AND 2 CREATED SUCCESSFULLY")

print("=" * 80)

print(
    "\nSaved:"
)

print(
    OUTPUT_FOLDER /
    "table_1_dataset_statistics.csv"
)

print(
    OUTPUT_FOLDER /
    "table_2_descriptive_statistics.csv"
)