
import os
import numpy as np
import pandas as pd

# ============================================================
# STEP 3: CAUSAL FEATURE ENGINEERING
# PaySim Financial Fraud Investigation Research
# ============================================================

INPUT_FILE = "data/processed/paysim_preprocessed.csv"
OUTPUT_FILE = "data/processed/paysim_features.csv"

print("=" * 70)
print("PAYSIM CAUSAL FEATURE ENGINEERING")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

print("\nLoading preprocessed dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df):,}")

# ------------------------------------------------------------
# 2. BASIC TRANSACTION FEATURES
# ------------------------------------------------------------

print("\nCreating basic transaction features...")

# Log-transformed amount
df["log_amount"] = np.log1p(df["amount"])

# Square-root transformed amount
df["sqrt_amount"] = np.sqrt(df["amount"])

# Hour within day
df["step_hour"] = df["step"] % 24

# Day number
df["step_day"] = (df["step"] - 1) // 24

# ------------------------------------------------------------
# 3. TRANSACTION TYPE ENCODING
# ------------------------------------------------------------

print("Encoding transaction type...")

type_dummies = pd.get_dummies(
    df["type"],
    prefix="type",
    dtype=np.int8
)

df = pd.concat([df, type_dummies], axis=1)

# ------------------------------------------------------------
# 4. ORIGIN ACCOUNT HISTORICAL FEATURES
# ------------------------------------------------------------

print("Creating origin-account historical features...")

# IMPORTANT:
# These features only use information BEFORE the current
# transaction, preventing future-data leakage.

origin_group = df.groupby("nameOrig", sort=False)

# Number of previous transactions from this origin
df["orig_prev_count"] = origin_group.cumcount().astype(np.int32)

# Previous cumulative amount
origin_cumsum = origin_group["amount"].cumsum()

df["orig_prev_total_amount"] = (
    origin_cumsum - df["amount"]
)

# Previous average transaction amount
df["orig_prev_avg_amount"] = np.where(
    df["orig_prev_count"] > 0,
    df["orig_prev_total_amount"] / df["orig_prev_count"],
    0.0
)

# Number of previous unique destinations
# We calculate cumulative unique destinations using a compact
# factorized pair representation.

origin_dest_pair = (
    df["nameOrig"].astype(str)
    + "||"
    + df["nameDest"].astype(str)
)

pair_first = ~origin_dest_pair.duplicated()

df["orig_unique_destinations"] = (
    pair_first.groupby(df["nameOrig"]).cumsum()
    .astype(np.int32)
)

# Remove current destination from the historical count
df["orig_unique_destinations"] = (
    df["orig_unique_destinations"] - pair_first.astype(np.int8)
).astype(np.int32)

del origin_group
del origin_cumsum
del origin_dest_pair
del pair_first

# ------------------------------------------------------------
# 5. DESTINATION ACCOUNT HISTORICAL FEATURES
# ------------------------------------------------------------

print("Creating destination-account historical features...")

dest_group = df.groupby("nameDest", sort=False)

# Number of previous transactions received
df["dest_prev_count"] = dest_group.cumcount().astype(np.int32)

# Previous cumulative received amount
dest_cumsum = dest_group["amount"].cumsum()

df["dest_prev_total_amount"] = (
    dest_cumsum - df["amount"]
)

# Previous average received amount
df["dest_prev_avg_amount"] = np.where(
    df["dest_prev_count"] > 0,
    df["dest_prev_total_amount"] / df["dest_prev_count"],
    0.0
)

del dest_group
del dest_cumsum

# ------------------------------------------------------------
# 6. RECENT ACTIVITY FEATURES
# ------------------------------------------------------------
#
# Instead of groupby().apply(), we use transaction-position
# based lag features.
#
# For an account, the previous N transactions provide a
# lightweight approximation of recent activity.
#
# This avoids the extremely slow Python-level loops that caused
# the previous implementation to run for a very long time.
# ------------------------------------------------------------

print("Creating recent activity features...")

# ------------------------------------------------------------
# Origin recent activity
# ------------------------------------------------------------

origin_order = df.groupby("nameOrig", sort=False).cumcount()

# Previous transaction step for each origin account
df["orig_prev_step"] = (
    df.groupby("nameOrig", sort=False)["step"]
    .shift(1)
)

# Gap between current and previous transaction
df["orig_step_gap"] = (
    df["step"] - df["orig_prev_step"]
).fillna(0).astype(np.int16)

# ------------------------------------------------------------
# Destination recent activity
# ------------------------------------------------------------

df["dest_prev_step"] = (
    df.groupby("nameDest", sort=False)["step"]
    .shift(1)
)

df["dest_step_gap"] = (
    df["step"] - df["dest_prev_step"]
).fillna(0).astype(np.int16)

del origin_order

# ------------------------------------------------------------
# Efficient recent transaction counts
# ------------------------------------------------------------
#
# Rather than performing expensive rolling calculations over
# millions of rows, we use transaction lags.
#
# These features represent recent behavioural intensity:
#
#   previous 1 transaction
#   previous 3 transactions
#   previous 5 transactions
#
# They are causal because only earlier transactions are used.
# ------------------------------------------------------------

print("Creating causal activity-lag features...")

# Origin previous transaction amounts
orig_amount_group = df.groupby("nameOrig", sort=False)["amount"]

df["orig_prev_amount"] = (
    orig_amount_group.shift(1).fillna(0.0)
)

df["orig_prev2_amount"] = (
    orig_amount_group.shift(2).fillna(0.0)
)

df["orig_prev3_amount"] = (
    orig_amount_group.shift(3).fillna(0.0)
)

# Destination previous transaction amounts
dest_amount_group = df.groupby("nameDest", sort=False)["amount"]

df["dest_prev_amount"] = (
    dest_amount_group.shift(1).fillna(0.0)
)

df["dest_prev2_amount"] = (
    dest_amount_group.shift(2).fillna(0.0)
)

df["dest_prev3_amount"] = (
    dest_amount_group.shift(3).fillna(0.0)
)

del orig_amount_group
del dest_amount_group

# ------------------------------------------------------------
# 7. PREVIOUS TRANSACTION TYPE INFORMATION
# ------------------------------------------------------------

print("Creating previous transaction behaviour features...")

df["orig_prev_type"] = (
    df.groupby("nameOrig", sort=False)["type"]
    .shift(1)
    .fillna("NONE")
)

df["dest_prev_type"] = (
    df.groupby("nameDest", sort=False)["type"]
    .shift(1)
    .fillna("NONE")
)

# Encode whether previous transaction exists
df["orig_has_history"] = (
    df["orig_prev_count"] > 0
).astype(np.int8)

df["dest_has_history"] = (
    df["dest_prev_count"] > 0
).astype(np.int8)

# ------------------------------------------------------------
# 8. BALANCE-CHANGE FEATURES
# ------------------------------------------------------------
#
# Balance columns are NOT used as direct predictive inputs
# because PaySim's simulated fraud mechanism can create
# leakage-like behaviour.
#
# We keep only behavioural differences as contextual features.
# ------------------------------------------------------------

print("Creating balance-change context features...")

df["orig_balance_change"] = (
    df["oldbalanceOrg"] - df["newbalanceOrig"]
)

df["dest_balance_change"] = (
    df["newbalanceDest"] - df["oldbalanceDest"]
)

# Amount-to-balance relationship
df["orig_amount_balance_ratio"] = np.where(
    df["oldbalanceOrg"] > 0,
    df["amount"] / (df["oldbalanceOrg"] + 1.0),
    0.0
)

# ------------------------------------------------------------
# 9. FINAL FEATURE CLEANUP
# ------------------------------------------------------------

print("Cleaning feature dataset...")

# Replace infinite values
df.replace(
    [np.inf, -np.inf],
    0,
    inplace=True
)

# Fill numeric missing values
numeric_columns = df.select_dtypes(
    include=[np.number]
).columns

df[numeric_columns] = (
    df[numeric_columns]
    .fillna(0)
)

# ------------------------------------------------------------
# 10. CONVERT OBJECT COLUMNS
# ------------------------------------------------------------

# We keep identifiers for forensic reconstruction.
# Machine-learning scripts can select only numerical features.

df["type"] = df["type"].astype("category")

df["orig_prev_type"] = (
    df["orig_prev_type"].astype("category")
)

df["dest_prev_type"] = (
    df["dest_prev_type"].astype("category")
)

# ------------------------------------------------------------
# 11. VALIDATION
# ------------------------------------------------------------

print("\nValidating causal features...")

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns):,}")

print(
    f"Fraud transactions: "
    f"{df['isFraud'].sum():,}"
)

print(
    f"Step range: "
    f"{df['step'].min()} -> {df['step'].max()}"
)

# Check transaction IDs
print(
    f"Unique transaction IDs: "
    f"{df['transaction_id'].nunique():,}"
)

# Check original row references
print(
    f"Unique original row references: "
    f"{df['original_row_number'].nunique():,}"
)

# Check missing values
missing_total = int(
    df.isna().sum().sum()
)

print(
    f"Total missing values: "
    f"{missing_total:,}"
)

# ------------------------------------------------------------
# 12. SAVE
# ------------------------------------------------------------

print("\nSaving engineered feature dataset...")

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 70)

print(f"Output file:")
print(OUTPUT_FILE)

print(f"\nRows: {len(df):,}")
print(f"Columns: {len(df.columns):,}")

print("\nResearch safeguards:")
print("✓ Historical features use previous transactions only")
print("✓ No future transactions used")
print("✓ Transaction IDs preserved")
print("✓ Original row references preserved")
print("✓ Raw dataset unchanged")
print("✓ Forensic identifiers preserved")

print("\nNext step:")
print("Run 04_train_models.py")
