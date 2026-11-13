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

PARAMETER_FILE = Path(
    "results/tables/table_3_distribution_parameters.csv"
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
# LOAD ADRA RESULT
# ============================================================

def get_best_distribution():

    adra_df = pd.read_csv(
        ADRA_FILE
    )

    adra_df = adra_df.sort_values(
        "ADRA_Rank"
    )

    best_distribution = (
        adra_df.iloc[0]["Distribution"]
    )

    print(
        f"\nADRA selected distribution: "
        f"{best_distribution}"
    )

    return best_distribution


# ============================================================
# GET TRAINING PARAMETERS
# ============================================================

def get_parameters(distribution_name):

    parameters_df = pd.read_csv(
        PARAMETER_FILE
    )

    row = parameters_df[
        parameters_df["Distribution"]
        == distribution_name
    ]

    if row.empty:

        raise ValueError(
            "Parameters not found for "
            f"{distribution_name}"
        )

    parameter_columns = [

        column

        for column in parameters_df.columns

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
# RMSE
# ============================================================

def calculate_rmse(
    observed,
    expected
):

    return np.sqrt(
        np.mean(
            (observed - expected) ** 2
        )
    )


# ============================================================
# MAE
# ============================================================

def calculate_mae(
    observed,
    expected
):

    return np.mean(
        np.abs(
            observed - expected
        )
    )


# ============================================================
# TEST EVALUATION
# ============================================================

def evaluate_test_data(
    distribution_name,
    parameters,
    test_data
):

    distribution = (
        DISTRIBUTIONS[
            distribution_name
        ]
    )

    # --------------------------------------------------------
    # Log likelihood
    # --------------------------------------------------------

    log_pdf = distribution.logpdf(
        test_data,
        *parameters
    )

    valid_log_pdf = log_pdf[
        np.isfinite(log_pdf)
    ]

    log_likelihood = np.sum(
        valid_log_pdf
    )

    # --------------------------------------------------------
    # Number of parameters
    # --------------------------------------------------------

    k = len(parameters)

    n = len(test_data)

    # --------------------------------------------------------
    # AIC
    # --------------------------------------------------------

    aic = (
        2 * k
        - 2 * log_likelihood
    )

    # --------------------------------------------------------
    # BIC
    # --------------------------------------------------------

    bic = (
        k * np.log(n)
        - 2 * log_likelihood
    )

    # --------------------------------------------------------
    # KS test
    # --------------------------------------------------------

    ks_statistic, ks_pvalue = (
        stats.kstest(
            test_data,
            distribution.cdf,
            args=parameters
        )
    )

    # --------------------------------------------------------
    # Empirical CDF
    # --------------------------------------------------------

    sorted_data = np.sort(
        test_data
    )

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

    rmse = calculate_rmse(
        empirical_cdf,
        fitted_cdf
    )

    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    mae = calculate_mae(
        empirical_cdf,
        fitted_cdf
    )

    return {

        "Distribution":
            distribution_name,

        "Test_Records":
            n,

        "Test_Log_Likelihood":
            log_likelihood,

        "Test_AIC":
            aic,

        "Test_BIC":
            bic,

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
# MAIN
# ============================================================

def main():

    print("=" * 90)

    print(
        "ADRA - TEST SET EVALUATION"
    )

    print("=" * 90)

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    test_df = pd.read_csv(
        TEST_FILE
    )

    test_data = (
        test_df[
            "Electricity_Consumed"
        ]
        .dropna()
        .values
    )

    print(
        f"\nTest observations: "
        f"{len(test_data)}"
    )

    # --------------------------------------------------------
    # Find ADRA selected distribution
    # --------------------------------------------------------

    best_distribution = (
        get_best_distribution()
    )

    # --------------------------------------------------------
    # Get parameters learned from TRAINING data
    # --------------------------------------------------------

    parameters = get_parameters(
        best_distribution
    )

    print(
        f"Training parameters: "
        f"{parameters}"
    )

    # --------------------------------------------------------
    # Evaluate on test data
    # --------------------------------------------------------

    result = evaluate_test_data(

        best_distribution,

        parameters,

        test_data

    )

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    result_df = pd.DataFrame(
        [result]
    )

    # --------------------------------------------------------
    # Save Table 6
    # --------------------------------------------------------

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_FOLDER /
        "table_6_test_evaluation.csv"
    )

    result_df.to_csv(
        output_file,
        index=False
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\n")

    print("=" * 100)

    print(
        "TABLE 6 - TEST SET EVALUATION"
    )

    print("=" * 100)

    print(
        result_df.round(6)
        .to_string(index=False)
    )

    print("\n")

    print(
        f"Table 6 saved to:"
    )

    print(output_file)

    print("\n")

    print("=" * 100)

    print(
        "TEST SET EVALUATION COMPLETED"
    )

    print("=" * 100)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()