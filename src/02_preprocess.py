# PaySim Forensic Investigation
# Step 2: Preprocessing
#
# Purpose:
# 1. Load the original PaySim dataset
# 2. Preserve the original row reference
# 3. Create deterministic transaction IDs
# 4. Ensure chronological ordering
# 5. Save a processed copy
#
# IMPORTANT:
# The original raw CSV is never modified.

import os
import pandas as pd


# =========================================================
# PATHS
# =========================================================

DATA_PATH = os.path.join(
    "data",
    "raw",
    "PS_20174392719_1491204439457_log.csv"
)

OUTPUT_DIR = os.path.join(
    "data",
    "processed"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "paysim_preprocessed.csv"
)


# =========================================================
# CREATE OUTPUT DIRECTORY
# =========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# CHECK INPUT FILE
# =========================================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )


print("=" * 70)
print("PAYSIM PREPROCESSING")
print("=" * 70)

print(f"\nInput dataset:")
print(DATA_PATH)

print("\nLoading dataset...")


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")


# =========================================================
# VERIFY EXPECTED COLUMNS
# =========================================================

expected_columns = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud"
]

missing_columns = [
    column
    for column in expected_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )


# =========================================================
# ORIGINAL ROW REFERENCE
# =========================================================
#
# CSV data rows start after the header.
#
# We use a zero-based internal row reference.
#
# original_row_number = 1 means the first transaction
# in the original CSV data.
#
# This mapping will later support forensic traceability.
# =========================================================

df["original_row_number"] = range(1, len(df) + 1)


# =========================================================
# DETERMINISTIC TRANSACTION ID
# =========================================================
#
# Example:
#
# TX00000001
# TX00000002
# TX00000003
#
# The ID is assigned according to the original dataset
# order and is therefore deterministic.
# =========================================================

df["transaction_id"] = [
    f"TX{i:08d}"
    for i in range(1, len(df) + 1)
]


# =========================================================
# CHECK TRANSACTION ID UNIQUENESS
# =========================================================

unique_transaction_ids = df["transaction_id"].nunique()

if unique_transaction_ids != len(df):
    raise ValueError(
        "Transaction IDs are not unique."
    )

print(
    f"\nTransaction IDs created: "
    f"{unique_transaction_ids:,}"
)


# =========================================================
# CHECK ORIGINAL ROW MAPPING
# =========================================================

if df["original_row_number"].is_unique is not True:
    raise ValueError(
        "Original row numbers are not unique."
    )

print(
    f"Original row references created: "
    f"{len(df):,}"
)


# =========================================================
# SORT CHRONOLOGICALLY
# =========================================================
#
# PaySim uses 'step' as simulation time.
#
# We sort by:
#   1. step
#   2. original_row_number
#
# The second field provides deterministic ordering for
# transactions occurring at the same step.
# =========================================================

df = df.sort_values(
    by=["step", "original_row_number"],
    ascending=[True, True]
).reset_index(drop=True)


# =========================================================
# VERIFY TEMPORAL ORDER
# =========================================================

if not df["step"].is_monotonic_increasing:
    raise ValueError(
        "Temporal ordering failed."
    )

print("\nTemporal ordering verified.")


# =========================================================
# VERIFY DATA INTEGRITY
# =========================================================

if df["isFraud"].isnull().any():
    raise ValueError(
        "isFraud contains missing values."
    )

if df["amount"].isnull().any():
    raise ValueError(
        "amount contains missing values."
    )

if df["nameOrig"].isnull().any():
    raise ValueError(
        "nameOrig contains missing values."
    )

if df["nameDest"].isnull().any():
    raise ValueError(
        "nameDest contains missing values."
    )


# =========================================================
# COLUMN ORDER
# =========================================================
#
# Keep transaction identifiers first so that every
# downstream file has an explicit evidence reference.
# =========================================================

final_columns = [
    "transaction_id",
    "original_row_number",
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud"
]

df = df[final_columns]


# =========================================================
# SAVE PROCESSED DATASET
# =========================================================

print("\nSaving processed dataset...")

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("Processed dataset saved successfully.")


# =========================================================
# FINAL VERIFICATION
# =========================================================

print("\n" + "=" * 70)
print("PREPROCESSING SUMMARY")
print("=" * 70)

print(f"Rows              : {len(df):,}")
print(f"Columns           : {len(df.columns):,}")
print(f"First step        : {df['step'].min()}")
print(f"Last step         : {df['step'].max()}")
print(
    f"Fraud transactions: "
    f"{df['isFraud'].sum():,}"
)

print(
    f"\nOutput file:\n"
    f"{OUTPUT_PATH}"
)

print("\nOriginal raw dataset was NOT modified.")

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETED")
print("=" * 70)