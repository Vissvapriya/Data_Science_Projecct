import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path(
    "results/tables/table_4_goodness_of_fit.csv"
)

OUTPUT_FILE = Path(
    "results/tables/table_5_adra_ranking.csv"
)


# ============================================================
# LOAD RESULTS
# ============================================================

def load_results():

    print("=" * 80)
    print("ADRA - ADAPTIVE DISTRIBUTION RANKING ALGORITHM")
    print("=" * 80)

    print("\nLoading distribution evaluation results...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Distributions found: {len(df)}")

    return df


# ============================================================
# MIN-MAX NORMALIZATION
# ============================================================

def normalize_lower_better(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(
            np.ones(len(series)),
            index=series.index
        )

    return (
        maximum - series
    ) / (
        maximum - minimum
    )


def normalize_higher_better(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(
            np.ones(len(series)),
            index=series.index
        )

    return (
        series - minimum
    ) / (
        maximum - minimum
    )


# ============================================================
# ADAPTIVE WEIGHT CALCULATION
# ============================================================

def calculate_adaptive_weights(df):

    print("\nCalculating adaptive weights...")

    # --------------------------------------------------------
    # Reliability of each metric
    # --------------------------------------------------------

    # Lower variation means more stable metric.
    # We calculate coefficient of variation.

    metrics = [
        "AIC",
        "BIC",
        "KS_Statistic",
        "KS_P_Value",
        "RMSE",
        "MAE"
    ]

    variability = {}

    for metric in metrics:

        mean_value = abs(df[metric].mean())

        std_value = df[metric].std()

        if mean_value == 0:

            variability[metric] = 1

        else:

            variability[metric] = (
                std_value / mean_value
            )

    # --------------------------------------------------------
    # Convert variability to reliability
    # --------------------------------------------------------

    reliability = {}

    for metric in metrics:

        reliability[metric] = (
            1 /
            (1 + variability[metric])
        )

    # --------------------------------------------------------
    # Normalize weights
    # --------------------------------------------------------

    total_reliability = sum(
        reliability.values()
    )

    weights = {}

    for metric in metrics:

        weights[metric] = (
            reliability[metric]
            / total_reliability
        )

    return weights


# ============================================================
# ADRA SCORE
# ============================================================

def calculate_adra(df, weights):

    print("\nCalculating ADRA scores...")

    # --------------------------------------------------------
    # Normalize metrics
    # --------------------------------------------------------

    df["Score_AIC"] = normalize_lower_better(
        df["AIC"]
    )

    df["Score_BIC"] = normalize_lower_better(
        df["BIC"]
    )

    df["Score_KS"] = normalize_lower_better(
        df["KS_Statistic"]
    )

    df["Score_P_Value"] = normalize_higher_better(
        df["KS_P_Value"]
    )

    df["Score_RMSE"] = normalize_lower_better(
        df["RMSE"]
    )

    df["Score_MAE"] = normalize_lower_better(
        df["MAE"]
    )

    # --------------------------------------------------------
    # Weighted ADRA score
    # --------------------------------------------------------

    df["ADRA_Score"] = (

        weights["AIC"]
        * df["Score_AIC"]

        +

        weights["BIC"]
        * df["Score_BIC"]

        +

        weights["KS_Statistic"]
        * df["Score_KS"]

        +

        weights["KS_P_Value"]
        * df["Score_P_Value"]

        +

        weights["RMSE"]
        * df["Score_RMSE"]

        +

        weights["MAE"]
        * df["Score_MAE"]

    )

    # --------------------------------------------------------
    # Rank distributions
    # --------------------------------------------------------

    df["ADRA_Rank"] = (
        df["ADRA_Score"]
        .rank(
            ascending=False,
            method="min"
        )
        .astype(int)
    )

    # Sort by rank

    df = df.sort_values(
        "ADRA_Rank"
    )

    return df


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(df):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nADRA ranking saved to:"
    )

    print(OUTPUT_FILE)


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(df, weights):

    print("\n")
    print("=" * 100)
    print("ADRA ADAPTIVE WEIGHTS")
    print("=" * 100)

    for metric, weight in weights.items():

        print(
            f"{metric:<20} : "
            f"{weight:.6f}"
        )

    print("\n")
    print("=" * 100)
    print("ADRA DISTRIBUTION RANKING")
    print("=" * 100)

    columns = [
        "Distribution",
        "AIC",
        "BIC",
        "KS_Statistic",
        "KS_P_Value",
        "RMSE",
        "MAE",
        "ADRA_Score",
        "ADRA_Rank"
    ]

    print(
        df[columns]
        .round(6)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Best distribution
    # --------------------------------------------------------

    best = df.iloc[0]

    print("\n")
    print("=" * 100)
    print("BEST-FIT DISTRIBUTION")
    print("=" * 100)

    print(
        f"\nDistribution : "
        f"{best['Distribution']}"
    )

    print(
        f"ADRA Score  : "
        f"{best['ADRA_Score']:.6f}"
    )

    print(
        f"Rank        : "
        f"{int(best['ADRA_Rank'])}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_results()

    # Calculate adaptive weights
    weights = calculate_adaptive_weights(
        df
    )

    # Calculate ADRA
    df = calculate_adra(
        df,
        weights
    )

    # Save
    save_results(df)

    # Display
    display_results(
        df,
        weights
    )

    print("\n")
    print("=" * 100)
    print("ADRA COMPLETED SUCCESSFULLY")
    print("=" * 100)


if __name__ == "__main__":
    main()