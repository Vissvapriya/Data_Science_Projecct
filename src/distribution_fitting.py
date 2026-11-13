import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy import stats


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path("data/processed/train.csv")

RESULTS_FOLDER = Path("results")

TABLE_FOLDER = RESULTS_FOLDER / "tables"

GRAPH_FOLDER = RESULTS_FOLDER / "graphs"


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

def load_data():

    print("\nLoading training data...")

    df = pd.read_csv(INPUT_FILE)

    data = df["Electricity_Consumed"].dropna().values

    print(f"Number of observations: {len(data)}")

    return data


# ============================================================
# CALCULATE LOG-LIKELIHOOD
# ============================================================

def calculate_log_likelihood(distribution, data, parameters):

    log_pdf = distribution.logpdf(
        data,
        *parameters
    )

    # Remove invalid values
    log_pdf = log_pdf[
        np.isfinite(log_pdf)
    ]

    return np.sum(log_pdf)


# ============================================================
# AIC
# ============================================================

def calculate_aic(log_likelihood, number_of_parameters):

    return (
        2 * number_of_parameters
        - 2 * log_likelihood
    )


# ============================================================
# BIC
# ============================================================

def calculate_bic(
    log_likelihood,
    number_of_parameters,
    sample_size
):

    return (
        number_of_parameters
        * np.log(sample_size)
        - 2 * log_likelihood
    )


# ============================================================
# RMSE
# ============================================================

def calculate_rmse(observed, expected):

    return np.sqrt(
        np.mean(
            (observed - expected) ** 2
        )
    )


# ============================================================
# MAE
# ============================================================

def calculate_mae(observed, expected):

    return np.mean(
        np.abs(
            observed - expected
        )
    )


# ============================================================
# FIT DISTRIBUTIONS
# ============================================================

def fit_distributions(data):

    results = []

    parameter_results = []

    x = np.sort(data)

    empirical_cdf = np.arange(
        1,
        len(data) + 1
    ) / len(data)

    for name, distribution in DISTRIBUTIONS.items():

        print(f"\nFitting {name} distribution...")

        try:

            # ------------------------------------------------
            # Estimate parameters
            # ------------------------------------------------

            parameters = distribution.fit(data)

            # ------------------------------------------------
            # Number of parameters
            # ------------------------------------------------

            k = len(parameters)

            # ------------------------------------------------
            # Log likelihood
            # ------------------------------------------------

            log_likelihood = calculate_log_likelihood(
                distribution,
                data,
                parameters
            )

            # ------------------------------------------------
            # AIC
            # ------------------------------------------------

            aic = calculate_aic(
                log_likelihood,
                k
            )

            # ------------------------------------------------
            # BIC
            # ------------------------------------------------

            bic = calculate_bic(
                log_likelihood,
                k,
                len(data)
            )

            # ------------------------------------------------
            # KS Test
            # ------------------------------------------------

            ks_statistic, ks_pvalue = stats.kstest(
                data,
                distribution.name,
                args=parameters
            )

            # ------------------------------------------------
            # Expected CDF
            # ------------------------------------------------

            fitted_cdf = distribution.cdf(
                x,
                *parameters
            )

            # ------------------------------------------------
            # RMSE
            # ------------------------------------------------

            rmse = calculate_rmse(
                empirical_cdf,
                fitted_cdf
            )

            # ------------------------------------------------
            # MAE
            # ------------------------------------------------

            mae = calculate_mae(
                empirical_cdf,
                fitted_cdf
            )

            # ------------------------------------------------
            # Store general results
            # ------------------------------------------------

            results.append({

                "Distribution": name,

                "Log_Likelihood":
                    log_likelihood,

                "AIC":
                    aic,

                "BIC":
                    bic,

                "KS_Statistic":
                    ks_statistic,

                "KS_P_Value":
                    ks_pvalue,

                "RMSE":
                    rmse,

                "MAE":
                    mae

            })

            # ------------------------------------------------
            # Store parameters
            # ------------------------------------------------

            parameter_row = {
                "Distribution": name
            }

            for i, parameter in enumerate(
                parameters,
                start=1
            ):

                parameter_row[
                    f"Parameter_{i}"
                ] = parameter

            parameter_results.append(
                parameter_row
            )

            print(
                f"  AIC = {aic:.4f}"
            )

            print(
                f"  BIC = {bic:.4f}"
            )

            print(
                f"  KS  = {ks_statistic:.4f}"
            )

            print(
                f"  p   = {ks_pvalue:.4f}"
            )

        except Exception as error:

            print(
                f"  ERROR: {error}"
            )


    results_df = pd.DataFrame(results)

    parameters_df = pd.DataFrame(
        parameter_results
    )

    return results_df, parameters_df


# ============================================================
# SAVE TABLES
# ============================================================

def save_tables(results_df, parameters_df):

    TABLE_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    # Table 3
    parameters_df.to_csv(
        TABLE_FOLDER /
        "table_3_distribution_parameters.csv",
        index=False
    )

    # Table 4
    results_df.to_csv(
        TABLE_FOLDER /
        "table_4_goodness_of_fit.csv",
        index=False
    )

    print("\nTables saved:")

    print(
        TABLE_FOLDER /
        "table_3_distribution_parameters.csv"
    )

    print(
        TABLE_FOLDER /
        "table_4_goodness_of_fit.csv"
    )


# ============================================================
# DISTRIBUTION COMPARISON GRAPH
# ============================================================

def create_distribution_graph(
    data,
    results_df,
    parameters_df
):

    GRAPH_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    x = np.linspace(
        data.min(),
        data.max(),
        500
    )

    plt.figure(
        figsize=(12, 7)
    )

    # Histogram
    plt.hist(
        data,
        bins=40,
        density=True,
        alpha=0.5,
        edgecolor="black",
        label="Observed Data"
    )

    # Fitted distributions
    for _, row in parameters_df.iterrows():

        name = row["Distribution"]

        distribution = DISTRIBUTIONS[name]

        parameters = [
            row[column]
            for column in parameters_df.columns
            if column.startswith("Parameter_")
            and pd.notna(row[column])
        ]

        pdf = distribution.pdf(
            x,
            *parameters
        )

        plt.plot(
            x,
            pdf,
            linewidth=2,
            label=name
        )

    plt.xlabel(
        "Electricity Consumption"
    )

    plt.ylabel(
        "Probability Density"
    )

    plt.title(
        "Probability Distribution Fitting "
        "for Electricity Consumption"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    output_file = (
        GRAPH_FOLDER /
        "probability_distribution_comparison.png"
    )

    plt.savefig(
        output_file,
        dpi=300
    )

    plt.close()

    print(
        f"\nSaved graph: {output_file}"
    )


# ============================================================
# PRINT RESULTS
# ============================================================

def display_results(results_df):

    print("\n")
    print("=" * 90)
    print("PROBABILITY DISTRIBUTION RESULTS")
    print("=" * 90)

    display_columns = [
        "Distribution",
        "AIC",
        "BIC",
        "KS_Statistic",
        "KS_P_Value",
        "RMSE",
        "MAE"
    ]

    print(
        results_df[
            display_columns
        ].round(6).to_string(index=False)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print(
        "ADRA - PROBABILITY DISTRIBUTION FITTING"
    )
    print("=" * 90)

    # Load data
    data = load_data()

    # Fit distributions
    results_df, parameters_df = (
        fit_distributions(data)
    )

    # Save tables
    save_tables(
        results_df,
        parameters_df
    )

    # Display results
    display_results(
        results_df
    )

    # Create graph
    create_distribution_graph(
        data,
        results_df,
        parameters_df
    )

    print("\n")
    print("=" * 90)
    print(
        "PROBABILITY DISTRIBUTION FITTING COMPLETED"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()