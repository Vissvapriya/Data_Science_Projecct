import pandas as pd
from pathlib import Path


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = Path("data/processed/cleaned_data.csv")

TRAIN_FILE = Path("data/processed/train.csv")

TEST_FILE = Path("data/processed/test.csv")


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_dataset():

    print("=" * 70)
    print("ADRA - TRAIN / TEST DATA SPLITTING")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load cleaned dataset
    # --------------------------------------------------------

    print("\nLoading cleaned dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Total records: {len(df)}")

    # --------------------------------------------------------
    # 2. Sort chronologically
    # --------------------------------------------------------

    print("\nSorting data chronologically...")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    df = df.sort_values("Timestamp")

    df = df.reset_index(drop=True)

    # --------------------------------------------------------
    # 3. 80/20 chronological split
    # --------------------------------------------------------

    split_index = int(len(df) * 0.80)

    train_df = df.iloc[:split_index].copy()

    test_df = df.iloc[split_index:].copy()

    # --------------------------------------------------------
    # 4. Save train and test data
    # --------------------------------------------------------

    train_df.to_csv(TRAIN_FILE, index=False)

    test_df.to_csv(TEST_FILE, index=False)

    # --------------------------------------------------------
    # 5. Display results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SPLIT RESULTS")
    print("=" * 70)

    print(f"\nTotal records : {len(df)}")
    print(f"Training data : {len(train_df)}")
    print(f"Testing data  : {len(test_df)}")

    print("\nTraining percentage: "
          f"{len(train_df) / len(df) * 100:.2f}%")

    print("Testing percentage : "
          f"{len(test_df) / len(df) * 100:.2f}%")

    # --------------------------------------------------------
    # 6. Display time ranges
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TIME RANGE")
    print("=" * 70)

    print("\nTraining data:")

    print(f"Start: {train_df['Timestamp'].min()}")

    print(f"End  : {train_df['Timestamp'].max()}")

    print("\nTesting data:")

    print(f"Start: {test_df['Timestamp'].min()}")

    print(f"End  : {test_df['Timestamp'].max()}")

    # --------------------------------------------------------
    # 7. Final confirmation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(f"\nTrain file: {TRAIN_FILE}")

    print(f"Test file : {TEST_FILE}")

    print("\nTrain/Test splitting completed successfully.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    split_dataset()