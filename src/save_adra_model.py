import pandas as pd
import pickle
from pathlib import Path


# ============================================================
# FILE PATHS
# ============================================================

ADRA_FILE = Path(
    "results/tables/table_5_adra_ranking.csv"
)

PARAMETER_FILE = Path(
    "results/tables/table_3_distribution_parameters.csv"
)

MODEL_FOLDER = Path(
    "results/models"
)


# ============================================================
# CREATE MODEL FOLDER
# ============================================================

MODEL_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD ADRA RESULTS
# ============================================================

adra_df = pd.read_csv(
    ADRA_FILE
)


# ============================================================
# FIND BEST DISTRIBUTION
# ============================================================

adra_df = adra_df.sort_values(
    "ADRA_Rank"
)

best_row = adra_df.iloc[0]

best_distribution = (
    best_row["Distribution"]
)

adra_score = (
    best_row["ADRA_Score"]
)

adra_rank = (
    best_row["ADRA_Rank"]
)


# ============================================================
# LOAD DISTRIBUTION PARAMETERS
# ============================================================

parameter_df = pd.read_csv(
    PARAMETER_FILE
)


best_parameters_row = parameter_df[
    parameter_df["Distribution"]
    == best_distribution
]


if best_parameters_row.empty:

    raise ValueError(
        "Parameters for the selected "
        "distribution were not found."
    )


# Find parameter columns

parameter_columns = [

    column

    for column in parameter_df.columns

    if column.startswith("Parameter_")

]


parameters = {}

for column in parameter_columns:

    value = (
        best_parameters_row
        .iloc[0][column]
    )

    if pd.notna(value):

        parameters[column] = float(value)


# ============================================================
# CREATE MODEL OBJECT
# ============================================================

adra_model = {

    "algorithm":
        "Adaptive Distribution Ranking Algorithm (ADRA)",

    "project_title":
        "Statistical Modeling of Smart Grid Energy Consumption Patterns Using Probability Distribution Metrics",

    "selected_distribution":
        best_distribution,

    "adra_score":
        float(adra_score),

    "adra_rank":
        int(adra_rank),

    "parameters":
        parameters

}


# ============================================================
# SAVE MODEL
# ============================================================

model_file = (
    MODEL_FOLDER /
    "adra_model.pkl"
)


with open(
    model_file,
    "wb"
) as file:

    pickle.dump(
        adra_model,
        file
    )


# ============================================================
# DISPLAY
# ============================================================

print("=" * 80)

print(
    "ADRA MODEL SAVING"
)

print("=" * 80)

print(
    f"\nAlgorithm           : "
    f"{adra_model['algorithm']}"
)

print(
    f"Selected Distribution: "
    f"{best_distribution}"
)

print(
    f"ADRA Score           : "
    f"{adra_score:.6f}"
)

print(
    f"ADRA Rank            : "
    f"{adra_rank}"
)

print(
    f"Parameters           : "
    f"{parameters}"
)

print(
    f"\nModel saved to:"
)

print(
    model_file
)

print("\n")

print("=" * 80)

print(
    "ADRA MODEL SAVED SUCCESSFULLY"
)

print("=" * 80)