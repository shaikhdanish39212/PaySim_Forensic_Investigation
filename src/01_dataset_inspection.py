# PaySim Forensic Investigation
# Step 1: Dataset Inspection

import os
import json
import pandas as pd


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

DATA_PATH = os.path.join(
    "data",
    "raw",
    "PS_20174392719_1491204439457_log.csv"
)

OUTPUT_DIR = os.path.join("outputs", "reports")
REPORT_PATH = os.path.join(OUTPUT_DIR, "dataset_report.json")


# ---------------------------------------------------------
# CREATE OUTPUT DIRECTORY
# ---------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# CHECK FILE
# ---------------------------------------------------------

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}\n\n"
        "Place the original PaySim CSV inside data/raw/"
    )


print("=" * 60)
print("PAYSIM DATASET INSPECTION")
print("=" * 60)

print(f"\nDataset: {DATA_PATH}")


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")


# ---------------------------------------------------------
# BASIC INFORMATION
# ---------------------------------------------------------

rows, columns = df.shape

print("\n" + "=" * 60)
print("1. BASIC DATASET INFORMATION")
print("=" * 60)

print(f"Rows    : {rows:,}")
print(f"Columns : {columns:,}")


# ---------------------------------------------------------
# COLUMN INFORMATION
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("2. COLUMNS")
print("=" * 60)

for i, column in enumerate(df.columns, start=1):
    print(f"{i:2}. {column}")


# ---------------------------------------------------------
# DATA TYPES
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("3. DATA TYPES")
print("=" * 60)

print(df.dtypes)


# ---------------------------------------------------------
# MISSING VALUES
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("4. MISSING VALUES")
print("=" * 60)

missing = df.isnull().sum()

print(missing)

total_missing = int(missing.sum())

print(f"\nTotal missing values: {total_missing:,}")


# ---------------------------------------------------------
# DUPLICATES
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("5. DUPLICATE ROWS")
print("=" * 60)

duplicate_count = int(df.duplicated().sum())

print(f"Duplicate rows: {duplicate_count:,}")


# ---------------------------------------------------------
# FRAUD DISTRIBUTION
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("6. FRAUD DISTRIBUTION")
print("=" * 60)

fraud_counts = df["isFraud"].value_counts().sort_index()

fraud_percentages = (
    df["isFraud"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

for value in fraud_counts.index:
    label = "Genuine" if value == 0 else "Fraud"

    print(
        f"{label:<10}: "
        f"{fraud_counts[value]:,} "
        f"({fraud_percentages[value]:.4f}%)"
    )


# ---------------------------------------------------------
# TRANSACTION TYPES
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("7. TRANSACTION TYPES")
print("=" * 60)

transaction_types = df["type"].value_counts()

print(transaction_types)


# ---------------------------------------------------------
# FRAUD BY TRANSACTION TYPE
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("8. FRAUD BY TRANSACTION TYPE")
print("=" * 60)

fraud_by_type = (
    df.groupby("type")["isFraud"]
    .agg(
        total_transactions="count",
        fraud_transactions="sum"
    )
    .sort_values("fraud_transactions", ascending=False)
)

fraud_by_type["fraud_percentage"] = (
    fraud_by_type["fraud_transactions"]
    / fraud_by_type["total_transactions"]
    * 100
)

print(fraud_by_type)


# ---------------------------------------------------------
# UNIQUE ACCOUNTS
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("9. UNIQUE ACCOUNTS")
print("=" * 60)

unique_origins = df["nameOrig"].nunique()
unique_destinations = df["nameDest"].nunique()

print(f"Unique source accounts      : {unique_origins:,}")
print(f"Unique destination accounts : {unique_destinations:,}")


# ---------------------------------------------------------
# TIME RANGE
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("10. TEMPORAL INFORMATION")
print("=" * 60)

min_step = int(df["step"].min())
max_step = int(df["step"].max())
unique_steps = int(df["step"].nunique())

print(f"Minimum step : {min_step}")
print(f"Maximum step : {max_step}")
print(f"Unique steps : {unique_steps}")


# ---------------------------------------------------------
# AMOUNT INFORMATION
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("11. TRANSACTION AMOUNT")
print("=" * 60)

print(f"Minimum amount : {df['amount'].min():,.2f}")
print(f"Maximum amount : {df['amount'].max():,.2f}")
print(f"Mean amount    : {df['amount'].mean():,.2f}")
print(f"Median amount  : {df['amount'].median():,.2f}")


# ---------------------------------------------------------
# FLAGGED FRAUD
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("12. isFlaggedFraud")
print("=" * 60)

flagged_distribution = df["isFlaggedFraud"].value_counts().sort_index()

print(flagged_distribution)


# ---------------------------------------------------------
# BALANCE COLUMNS
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("13. BALANCE COLUMNS")
print("=" * 60)

balance_columns = [
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]

print(df[balance_columns].describe())


# ---------------------------------------------------------
# DATASET REPORT
# ---------------------------------------------------------

report = {
    "dataset": "PaySim",
    "rows": int(rows),
    "columns": int(columns),
    "column_names": list(df.columns),
    "missing_values": {
        column: int(value)
        for column, value in missing.items()
    },
    "total_missing_values": total_missing,
    "duplicate_rows": duplicate_count,
    "fraud_distribution": {
        str(int(key)): int(value)
        for key, value in fraud_counts.items()
    },
    "transaction_types": {
        str(key): int(value)
        for key, value in transaction_types.items()
    },
    "unique_source_accounts": int(unique_origins),
    "unique_destination_accounts": int(unique_destinations),
    "step_range": {
        "minimum": min_step,
        "maximum": max_step,
        "unique_steps": unique_steps
    },
    "amount": {
        "minimum": float(df["amount"].min()),
        "maximum": float(df["amount"].max()),
        "mean": float(df["amount"].mean()),
        "median": float(df["amount"].median())
    },
    "is_flagged_fraud": {
        str(int(key)): int(value)
        for key, value in flagged_distribution.items()
    }
}


# ---------------------------------------------------------
# SAVE REPORT
# ---------------------------------------------------------

with open(REPORT_PATH, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)


# ---------------------------------------------------------
# FINISH
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("INSPECTION COMPLETED")
print("=" * 60)

print(f"\nReport saved to:")
print(REPORT_PATH)