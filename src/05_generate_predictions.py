
import os
import json
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix
)

# ============================================================
# STEP 5: GENERATE TRANSACTION-LEVEL SUSPICION SCORES
#
# PaySim Financial Fraud Investigation Research
# ============================================================
#
# Purpose:
#   1. Load the leakage-audited Random Forest
#   2. Generate validation probabilities
#   3. Select operating threshold using VALIDATION ONLY
#   4. Freeze the threshold
#   5. Generate TEST probabilities
#   6. Create transaction-level suspicion records
#
# IMPORTANT:
#   The test set is NOT used to choose the threshold.
# ============================================================

INPUT_FILE = "data/processed/paysim_features.csv"

MODEL_FILE = (
    "outputs/models/"
    "random_forest_audited.joblib"
)

OUTPUT_DIR = "outputs/predictions"

REPORT_DIR = "outputs/reports"

OUTPUT_FILE = (
    OUTPUT_DIR +
    "/paysim_transaction_predictions.csv"
)

THRESHOLD_REPORT = (
    REPORT_DIR +
    "/step5_threshold_report.json"
)

# ------------------------------------------------------------
# TEMPORAL SPLIT
# ------------------------------------------------------------

TRAIN_END_STEP = 323
VALIDATION_END_STEP = 399

# ------------------------------------------------------------
# THRESHOLD SEARCH
# ------------------------------------------------------------

THRESHOLD_MIN = 0.01
THRESHOLD_MAX = 0.99
THRESHOLD_STEP = 0.01

RANDOM_STATE = 42

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)

print("=" * 70)
print("STEP 5: TRANSACTION-LEVEL SUSPICION SCORES")
print("=" * 70)

start_time = time.time()

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

print("\nLoading engineered feature dataset...")

df = pd.read_csv(
    INPUT_FILE
)

print(
    f"Rows loaded: {len(df):,}"
)

print(
    f"Columns loaded: {len(df.columns):,}"
)

# ------------------------------------------------------------
# 2. LOAD MODEL
# ------------------------------------------------------------

print("\nLoading audited Random Forest...")

model = joblib.load(
    MODEL_FILE
)

print(
    "Model loaded successfully."
)

# ------------------------------------------------------------
# 3. DEFINE SAME FEATURES USED DURING TRAINING
# ------------------------------------------------------------

EXCLUDED_FEATURES = {
    # Forensic identifiers
    "transaction_id",
    "original_row_number",

    # Account identifiers
    "nameOrig",
    "nameDest",

    # Raw balance fields
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",

    # Categorical fields
    "type",
    "orig_prev_type",
    "dest_prev_type",

    # Target
    "isFraud",

    # PaySim flag
    "isFlaggedFraud",

    # Balance-derived features
    "orig_balance_change",
    "dest_balance_change",
    "orig_amount_balance_ratio"
}

feature_columns = [
    col
    for col in df.columns
    if col not in EXCLUDED_FEATURES
    and pd.api.types.is_numeric_dtype(
        df[col]
    )
]

print(
    f"\nPredictive features: "
    f"{len(feature_columns)}"
)

# ------------------------------------------------------------
# 4. TEMPORAL SPLIT
# ------------------------------------------------------------

print("\nCreating temporal validation/test sets...")

validation_df = df[
    (df["step"] > TRAIN_END_STEP)
    & (df["step"] <= VALIDATION_END_STEP)
].copy()

test_df = df[
    df["step"] > VALIDATION_END_STEP
].copy()

print(
    f"Validation rows: "
    f"{len(validation_df):,}"
)

print(
    f"Test rows: "
    f"{len(test_df):,}"
)

print(
    f"Validation fraud: "
    f"{validation_df['isFraud'].sum():,}"
)

print(
    f"Test fraud: "
    f"{test_df['isFraud'].sum():,}"
)

# ------------------------------------------------------------
# 5. VALIDATION PROBABILITIES
# ------------------------------------------------------------

print(
    "\nGenerating validation suspicion scores..."
)

X_validation = validation_df[
    feature_columns
].astype(np.float32)

y_validation = validation_df[
    "isFraud"
].astype(np.int8)

validation_probability = (
    model.predict_proba(
        X_validation
    )[:, 1]
)

print(
    "Validation probabilities generated."
)

# ------------------------------------------------------------
# 6. SELECT THRESHOLD USING VALIDATION ONLY
# ------------------------------------------------------------
#
# We search thresholds from 0.01 to 0.99.
#
# Selection criterion:
#   Maximum F1-score on validation data.
#
# F1 balances precision and recall.
#
# IMPORTANT:
#   Test data is completely excluded from this decision.
# ------------------------------------------------------------

print(
    "\nSelecting operating threshold "
    "using validation data only..."
)

thresholds = np.arange(
    THRESHOLD_MIN,
    THRESHOLD_MAX + THRESHOLD_STEP / 2,
    THRESHOLD_STEP
)

threshold_results = []

for threshold in thresholds:

    validation_prediction = (
        validation_probability >= threshold
    ).astype(np.int8)

    precision = precision_score(
        y_validation,
        validation_prediction,
        zero_division=0
    )

    recall = recall_score(
        y_validation,
        validation_prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_validation,
        validation_prediction,
        zero_division=0
    )

    threshold_results.append(
        {
            "threshold": float(
                threshold
            ),
            "precision": float(
                precision
            ),
            "recall": float(
                recall
            ),
            "f1": float(
                f1
            )
        }
    )

threshold_df = pd.DataFrame(
    threshold_results
)

# Maximum validation F1
best_index = (
    threshold_df["f1"]
    .idxmax()
)

best_threshold = float(
    threshold_df.loc[
        best_index,
        "threshold"
    ]
)

best_validation_precision = float(
    threshold_df.loc[
        best_index,
        "precision"
    ]
)

best_validation_recall = float(
    threshold_df.loc[
        best_index,
        "recall"
    ]
)

best_validation_f1 = float(
    threshold_df.loc[
        best_index,
        "f1"
    ]
)

print("\nSelected threshold:")
print(
    f"Threshold : "
    f"{best_threshold:.2f}"
)

print(
    f"Precision : "
    f"{best_validation_precision:.4f}"
)

print(
    f"Recall    : "
    f"{best_validation_recall:.4f}"
)

print(
    f"F1        : "
    f"{best_validation_f1:.4f}"
)

# ------------------------------------------------------------
# 7. SAVE THRESHOLD SEARCH
# ------------------------------------------------------------

threshold_search_file = (
    REPORT_DIR +
    "/step5_threshold_search.csv"
)

threshold_df.to_csv(
    threshold_search_file,
    index=False
)

# ------------------------------------------------------------
# 8. GENERATE TEST PROBABILITIES
# ------------------------------------------------------------

print(
    "\nGenerating test suspicion scores..."
)

X_test = test_df[
    feature_columns
].astype(np.float32)

y_test = test_df[
    "isFraud"
].astype(np.int8)

test_probability = (
    model.predict_proba(
        X_test
    )[:, 1]
)

print(
    "Test probabilities generated."
)

# ------------------------------------------------------------
# 9. APPLY FROZEN THRESHOLD
# ------------------------------------------------------------
#
# IMPORTANT:
# best_threshold was selected without looking at test labels.
#
# Now it is frozen and applied to test transactions.
# ------------------------------------------------------------

test_prediction = (
    test_probability >= best_threshold
).astype(np.int8)

# ------------------------------------------------------------
# 10. FINAL TEST EVALUATION
# ------------------------------------------------------------

print(
    "\nEvaluating frozen threshold "
    "on test data..."
)

test_precision = precision_score(
    y_test,
    test_prediction,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_prediction,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_prediction,
    zero_division=0
)

test_pr_auc = average_precision_score(
    y_test,
    test_probability
)

test_roc_auc = roc_auc_score(
    y_test,
    test_probability
)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_prediction,
    labels=[0, 1]
).ravel()

print("\nFinal test results:")
print(
    f"Threshold : {best_threshold:.2f}"
)

print(
    f"Precision : {test_precision:.4f}"
)

print(
    f"Recall    : {test_recall:.4f}"
)

print(
    f"F1        : {test_f1:.4f}"
)

print(
    f"PR-AUC    : {test_pr_auc:.4f}"
)

print(
    f"ROC-AUC   : {test_roc_auc:.4f}"
)

print(
    f"TN        : {tn:,}"
)

print(
    f"FP        : {fp:,}"
)

print(
    f"FN        : {fn:,}"
)

print(
    f"TP        : {tp:,}"
)

# ------------------------------------------------------------
# 11. CREATE TRANSACTION-LEVEL OUTPUT
# ------------------------------------------------------------
#
# This is the bridge between:
#
#       FRAUD DETECTION
#             ↓
#       FORENSIC INVESTIGATION
#
# Each suspicious transaction keeps its original identifiers
# so we can later retrieve supporting evidence.
# ------------------------------------------------------------

print(
    "\nCreating transaction-level "
    "prediction records..."
)

prediction_df = test_df[
    [
        "transaction_id",
        "original_row_number",
        "step",
        "type",
        "amount",
        "nameOrig",
        "nameDest",
        "isFraud"
    ]
].copy()

prediction_df[
    "suspicion_score"
] = test_probability

prediction_df[
    "predicted_suspicious"
] = test_prediction

# ------------------------------------------------------------
# 12. SUSPICION LEVEL
# ------------------------------------------------------------
#
# This is descriptive only.
#
# It is NOT a legal/criminal classification.
# ------------------------------------------------------------

def classify_suspicion(score, threshold):

    if score >= threshold:
        return "SUSPICIOUS"

    return "NOT_SUSPICIOUS"


prediction_df[
    "suspicion_level"
] = [
    classify_suspicion(
        score,
        best_threshold
    )
    for score in prediction_df[
        "suspicion_score"
    ]
]

# ------------------------------------------------------------
# 13. SORT BY SUSPICION SCORE
# ------------------------------------------------------------

prediction_df = prediction_df.sort_values(
    "suspicion_score",
    ascending=False
).reset_index(
    drop=True
)

# ------------------------------------------------------------
# 14. SAVE PREDICTIONS
# ------------------------------------------------------------

print(
    "\nSaving transaction-level predictions..."
)

prediction_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"Prediction file saved:"
)

print(
    OUTPUT_FILE
)

# ------------------------------------------------------------
# 15. SAVE THRESHOLD REPORT
# ------------------------------------------------------------

threshold_report = {

    "experiment":
        "Step 5 - Suspicion Score Generation",

    "model":
        "Random Forest - Leakage Audited",

    "threshold_selection": {

        "method":
            "Maximum validation F1",

        "validation_only":
            True,

        "threshold_min":
            THRESHOLD_MIN,

        "threshold_max":
            THRESHOLD_MAX,

        "threshold_step":
            THRESHOLD_STEP,

        "selected_threshold":
            best_threshold
    },

    "validation_metrics_at_selected_threshold": {

        "precision":
            best_validation_precision,

        "recall":
            best_validation_recall,

        "f1":
            best_validation_f1
    },

    "test_metrics": {

        "precision":
            float(test_precision),

        "recall":
            float(test_recall),

        "f1":
            float(test_f1),

        "pr_auc":
            float(test_pr_auc),

        "roc_auc":
            float(test_roc_auc),

        "true_negative":
            int(tn),

        "false_positive":
            int(fp),

        "false_negative":
            int(fn),

        "true_positive":
            int(tp)
    },

    "output": {

        "file":
            OUTPUT_FILE,

        "rows":
            int(len(prediction_df)),

        "suspicious_transactions":
            int(
                prediction_df[
                    "predicted_suspicious"
                ].sum()
            )
    },

    "research_safeguards": [

        "Threshold selected using validation data only",

        "Test labels not used for threshold selection",

        "Transaction identifiers preserved",

        "Original row references preserved",

        "Suspicion scores preserved",

        "No claim of criminal guilt",

        "Output supports later forensic case reconstruction"
    ]
}

with open(
    THRESHOLD_REPORT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        threshold_report,
        f,
        indent=4
    )

# ------------------------------------------------------------
# 16. DISPLAY SUSPICIOUS TRANSACTION SUMMARY
# ------------------------------------------------------------

suspicious_count = int(
    prediction_df[
        "predicted_suspicious"
    ].sum()
)

print("\n" + "=" * 70)
print("STEP 5 COMPLETE")
print("=" * 70)

print(
    f"\nTotal test transactions:"
    f" {len(prediction_df):,}"
)

print(
    f"Predicted suspicious:"
    f" {suspicious_count:,}"
)

print(
    f"Predicted non-suspicious:"
    f" {len(prediction_df) - suspicious_count:,}"
)

print(
    f"\nFrozen threshold:"
    f" {best_threshold:.2f}"
)

print("\nTop 10 suspicious transactions:")

print(
    prediction_df[
        [
            "transaction_id",
            "step",
            "type",
            "amount",
            "nameOrig",
            "nameDest",
            "suspicion_score",
            "predicted_suspicious"
        ]
    ].head(10).to_string(
        index=False
    )
)

print("\nFiles created:")

print(
    f"Prediction file:"
    f" {OUTPUT_FILE}"
)

print(
    f"Threshold report:"
    f" {THRESHOLD_REPORT}"
)

print(
    f"Threshold search:"
    f" {threshold_search_file}"
)

print(
    f"\nExecution time:"
    f" {time.time() - start_time:.2f} seconds"
)

print("\nNext step:")
print(
    "Build contextual evidence and "
    "candidate forensic cases."
)
