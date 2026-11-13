import pandas as pd
import numpy as np

from pathlib import Path
from scipy import stats


# ============================================================
# PATHS
# ============================================================

TRAIN_FILE = Path(
    "data/processed/train.csv"
)

TEST_FILE = Path(
    "data/processed/test.csv"
)

GOF_FILE = Path(
    "results/tables/table_4_goodness_of_fit.csv"
)

ADRA_FILE = Path(
    "results/tables/table_5_adra_ranking.csv"
)

OUTPUT_FOLDER = Path(
    "results/tables"
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
# LOAD DATA
# ============================================================

train_df = pd.read_csv(
    TRAIN_FILE
)

test_df = pd.read_csv(
    TEST_FILE
)


train_data = (
    train_df["Electricity_Consumed"]
    .dropna()
    .values
)

test_data = (
    test_df["Electricity_Consumed"]
    .dropna()
    .values
)


# ============================================================
# LOAD DISTRIBUTION PARAMETERS
# ============================================================

def get_distribution_parameters(
    distribution_name
):

    parameter_file = Path(
        "results/tables/"
        "table_3_distribution_parameters.csv"
    )

    df = pd.read_csv(
        parameter_file
    )

    row = df[
        df["Distribution"]
        == distribution_name
    ]

    if row.empty:

        raise ValueError(
            f"Parameters not found for "
            f"{distribution_name}"
        )

    parameter_columns = [

        column

        for column in df.columns

        if column.startswith("Parameter_")

    ]

    parameters = []

    for column in parameter_columns:

        value = row.iloc[0][column]

        if pd.notna(value):

            parameters.append(
                float(value)
            )

    return parameters


# ============================================================
# TEST EVALUATION
# ============================================================

def evaluate_distribution(
    distribution_name
):

    distribution = (
        DISTRIBUTIONS[
            distribution_name
        ]
    )

    parameters = (
        get_distribution_parameters(
            distribution_name
        )
    )

    # --------------------------------------------------------
    # KS TEST
    # --------------------------------------------------------

    ks_statistic, ks_pvalue = (
        stats.kstest(
            test_data,
            distribution.cdf,
            args=parameters
        )
    )

    # --------------------------------------------------------
    # EMPIRICAL CDF
    # --------------------------------------------------------

    sorted_data = np.sort(
        test_data
    )

    n = len(test_data)

    empirical_cdf = (
        np.arange(1, n + 1)
        / n
    )

    fitted_cdf = distribution.cdf(
        sorted_data,
        *parameters
    )

    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    rmse = np.sqrt(
        np.mean(
            (
                empirical_cdf
                - fitted_cdf
            ) ** 2
        )
    )

    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    mae = np.mean(
        np.abs(
            empirical_cdf
            - fitted_cdf
        )
    )

    return {

        "Test_KS_Statistic":
            ks_statistic,

        "Test_KS_P_Value":
            ks_pvalue,

        "Test_RMSE":
            rmse,

        "Test_MAE":
            mae

    }


# ============================================================
# FIND BEST DISTRIBUTION BY AIC
# ============================================================

gof_df = pd.read_csv(
    GOF_FILE
)

aic_distribution = (
    gof_df
    .sort_values("AIC")
    .iloc[0]["Distribution"]
)


# ============================================================
# FIND BEST DISTRIBUTION BY BIC
# ============================================================

bic_distribution = (
    gof_df
    .sort_values("BIC")
    .iloc[0]["Distribution"]
)


# ============================================================
# FIND BEST DISTRIBUTION BY KS
# ============================================================

ks_distribution = (
    gof_df
    .sort_values("KS_Statistic")
    .iloc[0]["Distribution"]
)


# ============================================================
# FIND BEST DISTRIBUTION BY ADRA
# ============================================================

adra_df = pd.read_csv(
    ADRA_FILE
)

adra_distribution = (
    adra_df
    .sort_values("ADRA_Rank")
    .iloc[0]["Distribution"]
)


# ============================================================
# METHOD SELECTION
# ============================================================

methods = {

    "AIC": aic_distribution,

    "BIC": bic_distribution,

    "KS": ks_distribution,

    "ADRA": adra_distribution

}


print("=" * 100)

print(
    "ADRA - BASELINE COMPARISON"
)

print("=" * 100)


print("\nSelected distributions:")

for method, distribution in methods.items():

    print(
        f"{method:10s} -> "
        f"{distribution}"
    )


# ============================================================
# EVALUATE EACH METHOD
# ============================================================

results = []


for method, distribution in methods.items():

    metrics = evaluate_distribution(
        distribution
    )

    results.append({

        "Method":
            method,

        "Selected_Distribution":
            distribution,

        "Test_KS_Statistic":
            metrics[
                "Test_KS_Statistic"
            ],

        "Test_KS_P_Value":
            metrics[
                "Test_KS_P_Value"
            ],

        "Test_RMSE":
            metrics[
                "Test_RMSE"
            ],

        "Test_MAE":
            metrics[
                "Test_MAE"
            ]

    })


# ============================================================
# CREATE DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# CREATE OVERALL SCORE
# ============================================================

results_df["Overall_Score"] = (

    results_df["Test_KS_Statistic"]

    + results_df["Test_RMSE"]

    + results_df["Test_MAE"]

)


# Lower score is better

results_df["Rank"] = (
    results_df[
        "Overall_Score"
    ]
    .rank(
        method="min"
    )
    .astype(int)
)


results_df = (
    results_df
    .sort_values("Rank")
)


# ============================================================
# SAVE TABLE 7
# ============================================================

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


output_file = (
    OUTPUT_FOLDER /
    "table_7_baseline_comparison.csv"
)


results_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")

print("=" * 120)

print(
    "TABLE 7 - BASELINE COMPARISON"
)

print("=" * 120)

print(
    results_df.round(6)
    .to_string(index=False)
)


print("\n")

print("=" * 120)

print(
    "TABLE 7 SAVED SUCCESSFULLY"
)

print("=" * 120)

print(
    output_file
)