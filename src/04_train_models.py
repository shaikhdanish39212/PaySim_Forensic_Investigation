import os
import json
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# ============================================================
# STEP 4: MODEL TRAINING
# LEAKAGE-AUDITED VERSION
#
# PaySim Financial Fraud Investigation Research
# ============================================================

INPUT_FILE = "data/processed/paysim_features.csv"

MODEL_DIR = "outputs/models"
REPORT_DIR = "outputs/reports"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

RANDOM_STATE = 42

# ------------------------------------------------------------
# TEMPORAL SPLIT
# ------------------------------------------------------------

TRAIN_END_STEP = 323
VALIDATION_END_STEP = 399

# ------------------------------------------------------------
# TRAINING SIZE CONTROL
# ------------------------------------------------------------

MAX_GENUINE_TRAIN = 300_000

print("=" * 70)
print("PAYSIM FRAUD MODEL TRAINING")
print("LEAKAGE-AUDITED FEATURE SET")
print("=" * 70)

start_time = time.time()

# ------------------------------------------------------------
# 1. LOAD FEATURE DATA
# ------------------------------------------------------------

print("\nLoading engineered feature dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df):,}")
print(f"Columns loaded: {len(df.columns):,}")

# ------------------------------------------------------------
# 2. TEMPORAL TRAIN / VALIDATION / TEST SPLIT
# ------------------------------------------------------------

print("\nCreating temporal train/validation/test split...")

train_df = df[
    df["step"] <= TRAIN_END_STEP
].copy()

validation_df = df[
    (df["step"] > TRAIN_END_STEP)
    & (df["step"] <= VALIDATION_END_STEP)
].copy()

test_df = df[
    df["step"] > VALIDATION_END_STEP
].copy()

print("\nTemporal split:")
print(
    f"Training steps    : 1 - {TRAIN_END_STEP}"
)
print(
    f"Validation steps  : "
    f"{TRAIN_END_STEP + 1} - {VALIDATION_END_STEP}"
)
print(
    f"Test steps        : "
    f"{VALIDATION_END_STEP + 1} - {df['step'].max()}"
)

print("\nRows:")
print(f"Training   : {len(train_df):,}")
print(f"Validation : {len(validation_df):,}")
print(f"Test       : {len(test_df):,}")

print("\nFraud distribution:")
print(
    f"Training fraud   : "
    f"{train_df['isFraud'].sum():,}"
)
print(
    f"Validation fraud : "
    f"{validation_df['isFraud'].sum():,}"
)
print(
    f"Test fraud       : "
    f"{test_df['isFraud'].sum():,}"
)

# ------------------------------------------------------------
# 3. DEFINE MODEL FEATURES
# ------------------------------------------------------------
#
# IMPORTANT:
#
# The following are excluded from prediction:
#
# - Account identifiers
# - Raw balance fields
# - Balance-derived features
# - Target label
# - PaySim's isFlaggedFraud field
#
# The balance-derived features were excluded because the
# previous experiment produced unusually near-perfect results
# and these features represented a large proportion of the
# Random Forest feature importance.
#
# They remain available in the feature dataset for later
# contextual/forensic analysis.
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
    and pd.api.types.is_numeric_dtype(df[col])
]

print("\nLeakage-audited model features:")
print(
    f"Number of predictive features: "
    f"{len(feature_columns)}"
)

for feature in feature_columns:
    print(f"  - {feature}")

print("\nExcluded balance-derived features:")
print("  - orig_balance_change")
print("  - dest_balance_change")
print("  - orig_amount_balance_ratio")

# ------------------------------------------------------------
# 4. PREPARE TRAINING DATA
# ------------------------------------------------------------

print("\nPreparing training data...")

fraud_train = train_df[
    train_df["isFraud"] == 1
].copy()

genuine_train = train_df[
    train_df["isFraud"] == 0
].copy()

print(
    f"Available genuine training transactions: "
    f"{len(genuine_train):,}"
)

print(
    f"Available fraud training transactions: "
    f"{len(fraud_train):,}"
)

if len(genuine_train) > MAX_GENUINE_TRAIN:

    print(
        f"\nSampling {MAX_GENUINE_TRAIN:,} genuine "
        f"transactions for model development..."
    )

    genuine_train = genuine_train.sample(
        n=MAX_GENUINE_TRAIN,
        random_state=RANDOM_STATE
    )

train_model_df = pd.concat(
    [
        genuine_train,
        fraud_train
    ],
    ignore_index=True
)

# Shuffle only the training sample.
train_model_df = train_model_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

print(
    f"\nFinal model-training rows: "
    f"{len(train_model_df):,}"
)

print(
    f"Genuine: "
    f"{(train_model_df['isFraud'] == 0).sum():,}"
)

print(
    f"Fraud: "
    f"{(train_model_df['isFraud'] == 1).sum():,}"
)

# ------------------------------------------------------------
# 5. CREATE X / y
# ------------------------------------------------------------

X_train = train_model_df[
    feature_columns
].astype(np.float32)

y_train = train_model_df[
    "isFraud"
].astype(np.int8)

X_validation = validation_df[
    feature_columns
].astype(np.float32)

y_validation = validation_df[
    "isFraud"
].astype(np.int8)

X_test = test_df[
    feature_columns
].astype(np.float32)

y_test = test_df[
    "isFraud"
].astype(np.int8)

print("\nMatrix shapes:")
print(f"X_train      : {X_train.shape}")
print(f"X_validation : {X_validation.shape}")
print(f"X_test       : {X_test.shape}")

# ------------------------------------------------------------
# 6. EVALUATION FUNCTION
# ------------------------------------------------------------

def evaluate_model(
    model_name,
    y_true,
    probabilities,
    threshold=0.5
):
    """
    Evaluate binary fraud predictions.
    """

    predictions = (
        probabilities >= threshold
    ).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    metrics = {
        "model": model_name,
        "threshold": float(threshold),

        "accuracy": float(
            accuracy_score(
                y_true,
                predictions
            )
        ),

        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities
            )
        ),

        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities
            )
        ),

        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp)
    }

    return metrics


# ------------------------------------------------------------
# 7. LOGISTIC REGRESSION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL 1: LOGISTIC REGRESSION")
print("=" * 70)

print("\nTraining Logistic Regression...")

lr_start = time.time()

logistic_model = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=300,
                class_weight="balanced",
                solver="lbfgs",
                random_state=RANDOM_STATE
            )
        )
    ]
)

logistic_model.fit(
    X_train,
    y_train
)

print(
    f"Training time: "
    f"{time.time() - lr_start:.2f} seconds"
)

# ------------------------------------------------------------
# 8. LOGISTIC REGRESSION VALIDATION
# ------------------------------------------------------------

print(
    "\nGenerating Logistic Regression "
    "validation probabilities..."
)

lr_validation_prob = (
    logistic_model.predict_proba(
        X_validation
    )[:, 1]
)

lr_validation_metrics = evaluate_model(
    "Logistic Regression",
    y_validation,
    lr_validation_prob,
    threshold=0.5
)

print("\nValidation results:")

for key, value in lr_validation_metrics.items():
    print(f"{key}: {value}")

# ------------------------------------------------------------
# 9. LOGISTIC REGRESSION TEST
# ------------------------------------------------------------

print(
    "\nEvaluating Logistic Regression "
    "on test set..."
)

lr_test_prob = (
    logistic_model.predict_proba(
        X_test
    )[:, 1]
)

lr_test_metrics = evaluate_model(
    "Logistic Regression",
    y_test,
    lr_test_prob,
    threshold=0.5
)

print("\nTest results:")

for key, value in lr_test_metrics.items():
    print(f"{key}: {value}")

joblib.dump(
    logistic_model,
    os.path.join(
        MODEL_DIR,
        "logistic_regression_audited.joblib"
    )
)

# ------------------------------------------------------------
# 10. RANDOM FOREST
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL 2: RANDOM FOREST")
print("=" * 70)

print("\nTraining Random Forest...")

rf_start = time.time()

random_forest = RandomForestClassifier(
    n_estimators=50,
    max_depth=12,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

random_forest.fit(
    X_train,
    y_train
)

print(
    f"Training time: "
    f"{time.time() - rf_start:.2f} seconds"
)

# ------------------------------------------------------------
# 11. RANDOM FOREST VALIDATION
# ------------------------------------------------------------

print(
    "\nGenerating Random Forest "
    "validation probabilities..."
)

rf_validation_prob = (
    random_forest.predict_proba(
        X_validation
    )[:, 1]
)

rf_validation_metrics = evaluate_model(
    "Random Forest",
    y_validation,
    rf_validation_prob,
    threshold=0.5
)

print("\nValidation results:")

for key, value in rf_validation_metrics.items():
    print(f"{key}: {value}")

# ------------------------------------------------------------
# 12. RANDOM FOREST TEST
# ------------------------------------------------------------

print(
    "\nEvaluating Random Forest "
    "on test set..."
)

rf_test_prob = (
    random_forest.predict_proba(
        X_test
    )[:, 1]
)

rf_test_metrics = evaluate_model(
    "Random Forest",
    y_test,
    rf_test_prob,
    threshold=0.5
)

print("\nTest results:")

for key, value in rf_test_metrics.items():
    print(f"{key}: {value}")

joblib.dump(
    random_forest,
    os.path.join(
        MODEL_DIR,
        "random_forest_audited.joblib"
    )
)

# ------------------------------------------------------------
# 13. RANDOM FOREST FEATURE IMPORTANCE
# ------------------------------------------------------------

print(
    "\nCalculating Random Forest "
    "feature importance..."
)

importance_df = pd.DataFrame(
    {
        "feature": feature_columns,
        "importance": (
            random_forest.feature_importances_
        )
    }
).sort_values(
    "importance",
    ascending=False
)

importance_file = os.path.join(
    REPORT_DIR,
    "random_forest_audited_feature_importance.csv"
)

importance_df.to_csv(
    importance_file,
    index=False
)

print("\nTop 15 features:")

print(
    importance_df.head(15).to_string(
        index=False
    )
)

# ------------------------------------------------------------
# 14. MODEL COMPARISON
# ------------------------------------------------------------

comparison = pd.DataFrame(
    [
        lr_test_metrics,
        rf_test_metrics
    ]
)

comparison_file = os.path.join(
    REPORT_DIR,
    "model_comparison_audited.csv"
)

comparison.to_csv(
    comparison_file,
    index=False
)

# ------------------------------------------------------------
# 15. COMPLETE JSON REPORT
# ------------------------------------------------------------

report = {
    "experiment": (
        "Step 4 - Leakage-Audited "
        "Fraud Model Training"
    ),

    "dataset": {
        "input_file": INPUT_FILE,
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "total_fraud": int(
            df["isFraud"].sum()
        )
    },

    "temporal_split": {
        "train_end_step": TRAIN_END_STEP,
        "validation_end_step": VALIDATION_END_STEP,
        "test_start_step": (
            VALIDATION_END_STEP + 1
        )
    },

    "rows": {
        "train_available": int(
            len(train_df)
        ),
        "train_model": int(
            len(train_model_df)
        ),
        "validation": int(
            len(validation_df)
        ),
        "test": int(
            len(test_df)
        )
    },

    "training_sampling": {
        "all_training_fraud_used": True,
        "maximum_genuine_training": (
            MAX_GENUINE_TRAIN
        ),
        "random_state": RANDOM_STATE
    },

    "leakage_audit": {
        "raw_balance_features_excluded": True,
        "balance_derived_features_excluded": True,
        "excluded_balance_derived_features": [
            "orig_balance_change",
            "dest_balance_change",
            "orig_amount_balance_ratio"
        ],
        "account_identifiers_excluded": True,
        "future_transactions_excluded": True,
        "target_excluded_from_features": True
    },

    "feature_configuration": {
        "number_of_features": len(
            feature_columns
        ),
        "features": feature_columns,
        "excluded_features": sorted(
            list(EXCLUDED_FEATURES)
        )
    },

    "models": {
        "logistic_regression": {
            "validation": (
                lr_validation_metrics
            ),
            "test": (
                lr_test_metrics
            )
        },

        "random_forest": {
            "validation": (
                rf_validation_metrics
            ),
            "test": (
                rf_test_metrics
            )
        }
    }
}

json_file = os.path.join(
    REPORT_DIR,
    "model_training_audited_report.json"
)

with open(
    json_file,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        report,
        f,
        indent=4
    )

# ------------------------------------------------------------
# 16. FINAL SUMMARY
# ------------------------------------------------------------

total_time = time.time() - start_time

print("\n" + "=" * 70)
print("LEAKAGE-AUDITED MODEL TRAINING COMPLETE")
print("=" * 70)

print(
    f"\nTotal execution time: "
    f"{total_time / 60:.2f} minutes"
)

print("\nModels saved:")

print(
    os.path.join(
        MODEL_DIR,
        "logistic_regression_audited.joblib"
    )
)

print(
    os.path.join(
        MODEL_DIR,
        "random_forest_audited.joblib"
    )
)

print("\nReports saved:")
print(comparison_file)
print(importance_file)
print(json_file)

print("\nResearch safeguards:")
print("✓ Temporal train/validation/test split")
print("✓ No random temporal split")
print("✓ Future transactions excluded")
print("✓ Account IDs excluded")
print("✓ Raw balance fields excluded")
print("✓ Balance-derived features excluded")
print("✓ isFlaggedFraud excluded")
print("✓ Target excluded from predictive features")
print("✓ Class imbalance handled")
print("✓ Complete validation set")
print("✓ Complete test set")
print("✓ Reproducible random state")
print("✓ PR-AUC reported")
print("✓ ROC-AUC reported")
print("✓ Precision, Recall and F1 reported")

print("\nNext step:")
print("Review audited results before generating predictions.")