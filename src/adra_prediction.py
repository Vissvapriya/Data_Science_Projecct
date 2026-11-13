import pandas as pd
import numpy as np
import pickle

from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

TEST_FILE = Path(
    "data/processed/test.csv"
)

MODEL_FILE = Path(
    "results/models/adra_model.pkl"
)

TABLE_FOLDER = Path(
    "results/tables"
)

GRAPH_FOLDER = Path(
    "results/graphs"
)


# ============================================================
# DISTRIBUTIONS
# ============================================================

DISTRIBUTIONS = {

    "Normal": stats.norm,

    "Lognormal": stats.lognorm,

    "Gamma": stats.gamma,

    "Weibull": stats.weibull_min,

    "Exponential": stats.expon,

    "GEV": stats.genextreme,

    "Student-t": stats.t

}


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

TABLE_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

GRAPH_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD ADRA MODEL
# ============================================================

with open(
    MODEL_FILE,
    "rb"
) as file:

    adra_model = pickle.load(file)


best_distribution = (
    adra_model[
        "selected_distribution"
    ]
)

parameters_dict = (
    adra_model[
        "parameters"
    ]
)


# ============================================================
# EXTRACT PARAMETERS
# ============================================================

parameters = list(
    parameters_dict.values()
)


distribution = (
    DISTRIBUTIONS[
        best_distribution
    ]
)


# ============================================================
# LOAD TEST DATA
# ============================================================

test_df = pd.read_csv(
    TEST_FILE
)


energy = (
    test_df[
        "Electricity_Consumed"
    ]
    .astype(float)
)


# ============================================================
# CALCULATE PROBABILITY
# ============================================================

pdf_values = distribution.pdf(
    energy,
    *parameters
)


cdf_values = distribution.cdf(
    energy,
    *parameters
)


# ============================================================
# Z-SCORE
# ============================================================

if best_distribution == "Normal":

    mean = parameters[0]

    std = parameters[1]

    z_scores = (
        (energy - mean)
        / std
    )

else:

    mean = energy.mean()

    std = energy.std()

    z_scores = (
        (energy - mean)
        / std
    )


# ============================================================
# ENERGY CLASSIFICATION
# ============================================================

classification = []

for z in z_scores:

    if z >= 2:

        classification.append(
            "Very High"
        )

    elif z >= 1:

        classification.append(
            "High"
        )

    elif z <= -2:

        classification.append(
            "Very Low"
        )

    elif z <= -1:

        classification.append(
            "Low"
        )

    else:

        classification.append(
            "Normal"
        )


# ============================================================
# CREATE RESULT DATAFRAME
# ============================================================

result_df = pd.DataFrame({

    "Timestamp":
        test_df["Timestamp"],

    "Electricity_Consumed":
        energy,

    "Probability_Density":
        pdf_values,

    "Cumulative_Probability":
        cdf_values,

    "Z_Score":
        z_scores,

    "Energy_Classification":
        classification,

    "Actual_Anomaly_Label":
        test_df[
            "Anomaly_Label"
        ]

})


# ============================================================
# SAVE PREDICTION RESULTS
# ============================================================

prediction_file = (
    TABLE_FOLDER /
    "adra_prediction_results.csv"
)


result_df.to_csv(
    prediction_file,
    index=False
)


# ============================================================
# GRAPH 1
# TEST DISTRIBUTION + ADRA MODEL
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.hist(
    energy,
    bins=40,
    density=True,
    alpha=0.6,
    label="Test Energy Consumption"
)


x = np.linspace(
    energy.min(),
    energy.max(),
    500
)


y = distribution.pdf(
    x,
    *parameters
)


plt.plot(
    x,
    y,
    linewidth=2,
    label=f"ADRA Selected: {best_distribution}"
)


plt.xlabel(
    "Electricity Consumed"
)

plt.ylabel(
    "Probability Density"
)

plt.title(
    "ADRA Probability Distribution Analysis"
)

plt.legend()

plt.tight_layout()


graph_file = (
    GRAPH_FOLDER /
    "adra_probability_analysis.png"
)


plt.savefig(
    graph_file,
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 2
# ENERGY CLASSIFICATION
# ============================================================

classification_counts = (
    result_df[
        "Energy_Classification"
    ]
    .value_counts()
)


plt.figure(
    figsize=(9, 6)
)

classification_counts.plot(
    kind="bar"
)

plt.xlabel(
    "Energy Consumption Category"
)

plt.ylabel(
    "Number of Observations"
)

plt.title(
    "ADRA Energy Consumption Classification"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()


graph_file = (
    GRAPH_FOLDER /
    "adra_energy_classification.png"
)


plt.savefig(
    graph_file,
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 3
# TEST ENERGY DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    energy.values,
    linewidth=1
)

plt.xlabel(
    "Test Observation"
)

plt.ylabel(
    "Electricity Consumed"
)

plt.title(
    "Test-Set Energy Consumption Pattern"
)

plt.tight_layout()


graph_file = (
    GRAPH_FOLDER /
    "adra_test_distribution.png"
)


plt.savefig(
    graph_file,
    dpi=300
)

plt.close()


# ============================================================
# SUMMARY
# ============================================================

print("=" * 90)

print(
    "ADRA ENERGY CONSUMPTION ANALYSIS"
)

print("=" * 90)

print(
    f"\nSelected Distribution : "
    f"{best_distribution}"
)

print(
    f"ADRA Score            : "
    f"{adra_model['adra_score']:.6f}"
)

print(
    f"Test Records          : "
    f"{len(result_df)}"
)


print("\nEnergy Classification:")

print(
    classification_counts
    .to_string()
)


print("\nFiles generated:")

print(
    prediction_file
)

print(
    GRAPH_FOLDER /
    "adra_probability_analysis.png"
)

print(
    GRAPH_FOLDER /
    "adra_energy_classification.png"
)

print(
    GRAPH_FOLDER /
    "adra_test_distribution.png"
)


print("\n")

print("=" * 90)

print(
    "ADRA ENERGY ANALYSIS COMPLETED"
)

print("=" * 90)
