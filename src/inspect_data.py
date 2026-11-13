import pandas as pd
from pathlib import Path


def inspect_dataset():

    print("=" * 70)
    print("SMART GRID DATASET INSPECTION")
    print("=" * 70)

    # CSV location
    file_path = Path("data/raw/smart_meter_data (1).csv")

    # Check whether file exists
    if not file_path.exists():
        print("\nERROR: Dataset not found!")
        print(f"Expected location: {file_path}")
        return

    print(f"\nDataset: {file_path}")

    # Load dataset
    df = pd.read_csv(file_path)

    # --------------------------------------------------
    # 1. Dataset Size
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("1. DATASET SIZE")
    print("=" * 70)

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    # --------------------------------------------------
    # 2. Column Names
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("2. COLUMN NAMES")
    print("=" * 70)

    for number, column in enumerate(df.columns, start=1):
        print(f"{number}. {column}")

    # --------------------------------------------------
    # 3. First 5 Records
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("3. FIRST 5 RECORDS")
    print("=" * 70)

    print(df.head().to_string())

    # --------------------------------------------------
    # 4. Last 5 Records
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("4. LAST 5 RECORDS")
    print("=" * 70)

    print(df.tail().to_string())

    # --------------------------------------------------
    # 5. Data Types
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("5. DATA TYPES")
    print("=" * 70)

    print(df.dtypes)

    # --------------------------------------------------
    # 6. Missing Values
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("6. MISSING VALUES")
    print("=" * 70)

    print(df.isnull().sum())

    # --------------------------------------------------
    # 7. Duplicate Records
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("7. DUPLICATE RECORDS")
    print("=" * 70)

    print(f"Duplicate rows: {df.duplicated().sum()}")

    # --------------------------------------------------
    # 8. Numerical Statistics
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("8. NUMERICAL STATISTICS")
    print("=" * 70)

    print(df.describe().to_string())

    # --------------------------------------------------
    # 9. Unique Values
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("9. UNIQUE VALUES")
    print("=" * 70)

    for column in df.columns:
        print(f"{column}: {df[column].nunique()} unique values")

    # --------------------------------------------------
    # 10. Memory Information
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("10. MEMORY USAGE")
    print("=" * 70)

    print(df.info())

    print("\n" + "=" * 70)
    print("DATASET INSPECTION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    inspect_dataset()