import json
import os
import re
import pandas as pd


# ============================================================
# STEP 9: FINAL FORENSIC EVALUATION
# ============================================================

PREDICTION_FILE = "outputs/predictions/paysim_transaction_predictions.csv"
CASE_FILE = "outputs/cases/candidate_forensic_cases.csv"
EVIDENCE_FILE = "outputs/cases/case_evidence.csv"
SEED_FILE = "outputs/cases/case_seeds.csv"

WINDOW_REPORT = "outputs/reports/window_sensitivity_results.csv"

OUTPUT_DIR = "outputs/reports"

FINAL_RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "final_forensic_evaluation_results.csv"
)

FINAL_REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "step9_final_evaluation_report.json"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_divide(a, b):
    if b == 0:
        return 0.0
    return a / b


def load_csv(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} not found: {path}"
        )

    df = pd.read_csv(path)

    print(f"{name}: {len(df):,} records")

    return df


def parse_transaction_ids(value):
    """
    Convert a case-level transaction ID field into a list.

    Handles:
        TX000001
        TX000001,TX000002
        TX000001;TX000002
        ['TX000001', 'TX000002']
    """

    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    text = text.strip("[]()")

    text = text.replace("'", "")
    text = text.replace('"', "")

    parts = re.split(
        r"[,;|]",
        text
    )

    ids = []

    for part in parts:

        transaction_id = part.strip()

        if transaction_id:
            ids.append(transaction_id)

    return ids


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 9: FINAL FORENSIC EVALUATION")
print("=" * 70)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 1. LOAD FILES
# ============================================================

print("\nLoading final evaluation files...")

predictions = load_csv(
    PREDICTION_FILE,
    "Transaction predictions"
)

cases = load_csv(
    CASE_FILE,
    "Candidate forensic cases"
)

evidence = load_csv(
    EVIDENCE_FILE,
    "Case evidence"
)

seeds = load_csv(
    SEED_FILE,
    "Case seeds"
)

window_results = load_csv(
    WINDOW_REPORT,
    "Window sensitivity results"
)


# ============================================================
# 2. VALIDATE PREDICTION FILE
# ============================================================

required_prediction_columns = [
    "transaction_id",
    "suspicion_score",
    "predicted_suspicious",
    "suspicion_level"
]

missing_prediction_columns = [
    column
    for column in required_prediction_columns
    if column not in predictions.columns
]

if missing_prediction_columns:
    raise ValueError(
        "Missing prediction columns: "
        + ", ".join(missing_prediction_columns)
    )


# ============================================================
# 3. VALIDATE CASE FILE
# ============================================================

required_case_columns = [
    "case_id",
    "seed_count",
    "evidence_transaction_count",
    "context_transaction_count",
    "seed_transaction_ids",
    "evidence_transaction_ids",
    "finding"
]

missing_case_columns = [
    column
    for column in required_case_columns
    if column not in cases.columns
]

if missing_case_columns:
    raise ValueError(
        "Missing case columns: "
        + ", ".join(missing_case_columns)
    )


# ============================================================
# 4. VALIDATE EVIDENCE FILE
# ============================================================

required_evidence_columns = [
    "case_id",
    "transaction_id",
    "evidence_role",
    "seed_transaction_id"
]

missing_evidence_columns = [
    column
    for column in required_evidence_columns
    if column not in evidence.columns
]

if missing_evidence_columns:
    raise ValueError(
        "Missing evidence columns: "
        + ", ".join(missing_evidence_columns)
    )


# ============================================================
# 5. BASIC DATA VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("DATA VALIDATION")
print("-" * 70)

prediction_unique = (
    predictions["transaction_id"].nunique()
)

seed_unique = (
    seeds["transaction_id"].nunique()
)

evidence_unique = (
    evidence["transaction_id"].nunique()
)

case_unique = (
    cases["case_id"].nunique()
)

print(
    f"Prediction records: "
    f"{len(predictions):,}"
)

print(
    f"Unique prediction transaction IDs: "
    f"{prediction_unique:,}"
)

print(
    f"Seed records: "
    f"{len(seeds):,}"
)

print(
    f"Unique seed transaction IDs: "
    f"{seed_unique:,}"
)

print(
    f"Evidence records: "
    f"{len(evidence):,}"
)

print(
    f"Unique evidence transaction IDs: "
    f"{evidence_unique:,}"
)

print(
    f"Case records: "
    f"{len(cases):,}"
)

print(
    f"Unique case IDs: "
    f"{case_unique:,}"
)


# ============================================================
# 6. FROZEN FINAL CONFIGURATION
# ============================================================

FINAL_WINDOW = 24
FINAL_THRESHOLD = 0.95

suspicious_predictions = predictions[
    predictions["predicted_suspicious"] == 1
].copy()

suspicious_seed_count = len(
    suspicious_predictions
)

print("\n" + "-" * 70)
print("FROZEN CONFIGURATION")
print("-" * 70)

print(
    f"Suspicious seeds: "
    f"{suspicious_seed_count:,}"
)

print(
    f"Suspicion threshold: "
    f"{FINAL_THRESHOLD}"
)

print(
    f"Temporal window: "
    f"+/- {FINAL_WINDOW} PaySim steps"
)


# ============================================================
# 7. THRESHOLD CONSISTENCY
# ============================================================

print("\nValidating suspicion threshold...")

min_suspicious_score = (
    suspicious_predictions["suspicion_score"].min()
)

print(
    f"Minimum suspicious score among selected seeds: "
    f"{min_suspicious_score:.6f}"
)

if min_suspicious_score >= FINAL_THRESHOLD:
    threshold_status = "PASSED"
else:
    threshold_status = "FAILED"

print(
    f"Threshold consistency check: "
    f"{threshold_status}"
)


# ============================================================
# 8. WINDOW SELECTION VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("WINDOW SELECTION VALIDATION")
print("-" * 70)

selected_window_row = window_results[
    window_results["window_steps"] == FINAL_WINDOW
]

if selected_window_row.empty:
    raise ValueError(
        "Final window of +/-24 steps was not found "
        "in window sensitivity results."
    )

selected_window_row = selected_window_row.iloc[0]

selected_cases_with_context = int(
    selected_window_row["cases_with_context"]
)

stored_window_rate = float(
    selected_window_row["context_enrichment_rate"]
)

# Step 8 stores this as a decimal.
# Example:
# 0.022917 = 2.2917%

if stored_window_rate <= 1:
    selected_enrichment_percent = (
        stored_window_rate * 100
    )
else:
    selected_enrichment_percent = (
        stored_window_rate
    )

print(
    f"Final window: "
    f"+/- {FINAL_WINDOW} steps"
)

print(
    f"Cases with contextual evidence: "
    f"{selected_cases_with_context}"
)

print(
    f"Context enrichment rate: "
    f"{selected_enrichment_percent:.2f}%"
)


# ============================================================
# 9. CASE COMPLETENESS
# ============================================================

print("\n" + "-" * 70)
print("CASE COMPLETENESS")
print("-" * 70)

candidate_cases = len(cases)

complete_cases = 0

for _, row in cases.iterrows():

    seed_ids = parse_transaction_ids(
        row["seed_transaction_ids"]
    )

    evidence_ids = parse_transaction_ids(
        row["evidence_transaction_ids"]
    )

    finding = row["finding"]

    case_valid = True

    # Case ID
    if pd.isna(row["case_id"]):
        case_valid = False

    # At least one seed
    if len(seed_ids) == 0:
        case_valid = False

    # At least one evidence transaction
    if len(evidence_ids) == 0:
        case_valid = False

    # Finding must exist
    if pd.isna(finding):
        case_valid = False

    elif (
        isinstance(finding, str)
        and finding.strip() == ""
    ):
        case_valid = False

    if case_valid:
        complete_cases += 1


case_completeness_rate = safe_divide(
    complete_cases,
    candidate_cases
) * 100

print(
    f"Candidate cases: "
    f"{candidate_cases:,}"
)

print(
    f"Structurally complete cases: "
    f"{complete_cases:,}"
)

print(
    f"Case completeness rate: "
    f"{case_completeness_rate:.2f}%"
)


# ============================================================
# 10. EVIDENCE ROLE ANALYSIS
# ============================================================

print("\n" + "-" * 70)
print("EVIDENCE ROLE ANALYSIS")
print("-" * 70)

evidence[
    "evidence_role_normalized"
] = (
    evidence[
        "evidence_role"
    ]
    .astype(str)
    .str.strip()
    .str.lower()
)

role_counts = (
    evidence[
        "evidence_role_normalized"
    ].value_counts()
)

print("Evidence roles:")

for role, count in role_counts.items():

    print(
        f"  {role}: {count:,}"
    )


# IMPORTANT:
# Actual evidence roles in this project are:
#
# suspicious_seed
# contextual_evidence
#
# NOT:
# seed
# context

seed_evidence = evidence[
    evidence[
        "evidence_role_normalized"
    ] == "suspicious_seed"
].copy()

context_evidence = evidence[
    evidence[
        "evidence_role_normalized"
    ] == "contextual_evidence"
].copy()


print(
    f"\nSeed evidence records: "
    f"{len(seed_evidence):,}"
)

print(
    f"Contextual evidence records: "
    f"{len(context_evidence):,}"
)


# ============================================================
# 11. EVIDENCE ROLE SANITY CHECK
# ============================================================

EXPECTED_SEED_EVIDENCE = 480
EXPECTED_CONTEXT_EVIDENCE = 11

if len(seed_evidence) != EXPECTED_SEED_EVIDENCE:

    raise ValueError(
        "Unexpected suspicious_seed evidence count: "
        f"{len(seed_evidence)}. "
        f"Expected {EXPECTED_SEED_EVIDENCE}."
    )

if len(context_evidence) != EXPECTED_CONTEXT_EVIDENCE:

    raise ValueError(
        "Unexpected contextual_evidence count: "
        f"{len(context_evidence)}. "
        f"Expected {EXPECTED_CONTEXT_EVIDENCE}."
    )


# ============================================================
# 12. CONTEXTUAL CASE COVERAGE
# ============================================================

print("\n" + "-" * 70)
print("CONTEXTUAL EVIDENCE COVERAGE")
print("-" * 70)

contextual_case_ids = set(
    context_evidence[
        "case_id"
    ]
    .dropna()
    .astype(str)
)

case_ids = set(
    cases[
        "case_id"
    ]
    .dropna()
    .astype(str)
)

contextual_case_ids = (
    contextual_case_ids
    .intersection(case_ids)
)

cases_with_context = len(
    contextual_case_ids
)

context_enrichment_rate = safe_divide(
    cases_with_context,
    candidate_cases
) * 100

print(
    f"Cases with contextual evidence: "
    f"{cases_with_context:,}"
)

print(
    f"Context enrichment rate: "
    f"{context_enrichment_rate:.2f}%"
)


# ============================================================
# 13. EVIDENCE COVERAGE
# ============================================================

print("\n" + "-" * 70)
print("EVIDENCE COVERAGE")
print("-" * 70)

evidence_records = len(
    evidence
)

context_evidence_records = len(
    context_evidence
)

context_evidence_ratio = safe_divide(
    context_evidence_records,
    evidence_records
) * 100

print(
    f"Total evidence records: "
    f"{evidence_records:,}"
)

print(
    f"Suspicious-seed evidence records: "
    f"{len(seed_evidence):,}"
)

print(
    f"Contextual evidence records: "
    f"{context_evidence_records:,}"
)

print(
    f"Context evidence ratio: "
    f"{context_evidence_ratio:.2f}%"
)


# ============================================================
# 14. AVERAGE EVIDENCE PER CASE
# ============================================================

average_evidence_per_case = safe_divide(
    evidence_records,
    candidate_cases
)

average_context_per_case = safe_divide(
    context_evidence_records,
    candidate_cases
)

print(
    f"Average evidence records/case: "
    f"{average_evidence_per_case:.4f}"
)

print(
    f"Average contextual evidence/case: "
    f"{average_context_per_case:.4f}"
)


# ============================================================
# 15. SEED TRACEABILITY
# ============================================================

print("\n" + "-" * 70)
print("TRACEABILITY EVALUATION")
print("-" * 70)

case_seed_reference_set = set()

for value in cases[
    "seed_transaction_ids"
]:

    parsed_ids = parse_transaction_ids(
        value
    )

    for transaction_id in parsed_ids:

        case_seed_reference_set.add(
            transaction_id
        )

seed_ids = set(
    seeds[
        "transaction_id"
    ]
    .dropna()
    .astype(str)
)

traceable_seed_ids = (
    seed_ids
    .intersection(
        case_seed_reference_set
    )
)

seed_traceability = safe_divide(
    len(traceable_seed_ids),
    len(seed_ids)
) * 100

print(
    f"Seed IDs referenced by cases: "
    f"{len(case_seed_reference_set):,}"
)

print(
    f"Traceable seed IDs: "
    f"{len(traceable_seed_ids):,}"
)

print(
    f"Seed traceability: "
    f"{seed_traceability:.2f}%"
)


# ============================================================
# 16. EVIDENCE TRACEABILITY
# ============================================================

prediction_ids = set(
    predictions[
        "transaction_id"
    ]
    .dropna()
    .astype(str)
)

evidence_ids = set(
    evidence[
        "transaction_id"
    ]
    .dropna()
    .astype(str)
)

traceable_evidence_ids = (
    evidence_ids
    .intersection(
        prediction_ids
    )
)

evidence_traceability = safe_divide(
    len(traceable_evidence_ids),
    len(evidence_ids)
) * 100

print(
    f"Evidence traceability: "
    f"{evidence_traceability:.2f}%"
)


# ============================================================
# 17. FINDING TRACEABILITY
# ============================================================

finding_traceable_cases = 0

for _, row in cases.iterrows():

    has_case_id = not pd.isna(
        row["case_id"]
    )

    seed_ids = parse_transaction_ids(
        row["seed_transaction_ids"]
    )

    has_seed_reference = (
        len(seed_ids) > 0
    )

    finding = row["finding"]

    has_finding = (
        not pd.isna(finding)
        and
        (
            not isinstance(finding, str)
            or finding.strip() != ""
        )
    )

    if (
        has_case_id
        and has_seed_reference
        and has_finding
    ):
        finding_traceable_cases += 1


finding_traceability = safe_divide(
    finding_traceable_cases,
    candidate_cases
) * 100

print(
    f"Finding traceability: "
    f"{finding_traceability:.2f}%"
)


# ============================================================
# 18. MULTI-SEED CASE ANALYSIS
# ============================================================

print("\n" + "-" * 70)
print("MULTI-SEED CASE ANALYSIS")
print("-" * 70)

multi_seed_cases = 0

for _, row in cases.iterrows():

    seed_ids = parse_transaction_ids(
        row["seed_transaction_ids"]
    )

    if len(set(seed_ids)) > 1:
        multi_seed_cases += 1


multi_seed_rate = safe_divide(
    multi_seed_cases,
    candidate_cases
) * 100

print(
    f"Multi-seed cases: "
    f"{multi_seed_cases:,}"
)

print(
    f"Multi-seed case rate: "
    f"{multi_seed_rate:.2f}%"
)


# ============================================================
# 19. STEP 8 -> STEP 9 CONSISTENCY
# ============================================================

print("\n" + "-" * 70)
print("STEP 8 CONSISTENCY CHECK")
print("-" * 70)

expected_cases = int(
    selected_window_row[
        "candidate_cases"
    ]
)

expected_context_cases = int(
    selected_window_row[
        "cases_with_context"
    ]
)

expected_context_transactions = int(
    selected_window_row[
        "context_transactions"
    ]
)

expected_multi_seed = int(
    selected_window_row[
        "multi_seed_cases"
    ]
)

print(
    f"Step 8 candidate cases: "
    f"{expected_cases}"
)

print(
    f"Step 9 candidate cases: "
    f"{candidate_cases}"
)

print(
    f"Step 8 context cases: "
    f"{expected_context_cases}"
)

print(
    f"Step 9 context cases: "
    f"{cases_with_context}"
)

print(
    f"Step 8 context transactions: "
    f"{expected_context_transactions}"
)

print(
    f"Step 9 context transactions: "
    f"{context_evidence_records}"
)

print(
    f"Step 8 multi-seed cases: "
    f"{expected_multi_seed}"
)

print(
    f"Step 9 multi-seed cases: "
    f"{multi_seed_cases}"
)


case_consistency = (
    expected_cases
    == candidate_cases
)

context_case_consistency = (
    expected_context_cases
    == cases_with_context
)

context_transaction_consistency = (
    expected_context_transactions
    == context_evidence_records
)

multi_seed_consistency = (
    expected_multi_seed
    == multi_seed_cases
)


if (
    case_consistency
    and context_case_consistency
    and context_transaction_consistency
    and multi_seed_consistency
):

    consistency_status = "PASSED"

else:

    consistency_status = "FAILED"


print(
    f"\nStep 8 -> Step 9 consistency: "
    f"{consistency_status}"
)


# ============================================================
# 20. FINAL RESULTS TABLE
# ============================================================

results = [

    {
        "metric": "Suspicious seeds",
        "value": suspicious_seed_count,
        "unit": "count"
    },

    {
        "metric": "Candidate forensic cases",
        "value": candidate_cases,
        "unit": "count"
    },

    {
        "metric": "Structurally complete cases",
        "value": complete_cases,
        "unit": "count"
    },

    {
        "metric": "Case completeness rate",
        "value": round(
            case_completeness_rate,
            4
        ),
        "unit": "percent"
    },

    {
        "metric": "Cases with contextual evidence",
        "value": cases_with_context,
        "unit": "count"
    },

    {
        "metric": "Context enrichment rate",
        "value": round(
            context_enrichment_rate,
            4
        ),
        "unit": "percent"
    },

    {
        "metric": "Total evidence records",
        "value": evidence_records,
        "unit": "count"
    },

    {
        "metric": "Suspicious-seed evidence records",
        "value": len(seed_evidence),
        "unit": "count"
    },

    {
        "metric": "Contextual evidence records",
        "value": context_evidence_records,
        "unit": "count"
    },

    {
        "metric": "Context evidence ratio",
        "value": round(
            context_evidence_ratio,
            4
        ),
        "unit": "percent"
    },

    {
        "metric": "Average evidence per case",
        "value": round(
            average_evidence_per_case,
            4
        ),
        "unit": "records/case"
    },

    {
        "metric": "Average contextual evidence per case",
        "value": round(
            average_context_per_case,
            4
        ),
        "unit": "records/case"
    },

    {
        "metric": "Seed traceability",
        "value": round(
            seed_traceability,
            4
        ),
        "unit": "percent"
    },

    {
        "metric": "Evidence traceability",
        "value": round(
            evidence_traceability,
            4
        ),
        "unit": "percent"
    },

    {
        "metric": "Finding traceability",
        "value": round(
            finding_traceability,
            4
        ),
        "unit": "percent"
    },

    {
        "metric": "Multi-seed cases",
        "value": multi_seed_cases,
        "unit": "count"
    },

    {
        "metric": "Multi-seed case rate",
        "value": round(
            multi_seed_rate,
            4
        ),
        "unit": "percent"
    },

    {
        "metric": "Final reconstruction window",
        "value": FINAL_WINDOW,
        "unit": "PaySim steps (+/-)"
    },

    {
        "metric": "Final suspicion threshold",
        "value": FINAL_THRESHOLD,
        "unit": "probability"
    }
]


results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    FINAL_RESULTS_FILE,
    index=False
)


# ============================================================
# 21. FINAL JSON REPORT
# ============================================================

report = {

    "step": 9,

    "title":
        "Final Forensic Evaluation",

    "configuration": {

        "model":
            "Audited Random Forest",

        "suspicion_threshold":
            FINAL_THRESHOLD,

        "temporal_window_steps":
            FINAL_WINDOW,

        "account_relationship_rule":
            "Same source account OR same destination account "
            "OR cross origin/destination relationship",

        "model_retrained":
            False,

        "threshold_changed":
            False
    },

    "dataset_evaluation": {

        "prediction_records":
            int(len(predictions)),

        "unique_prediction_transaction_ids":
            int(prediction_unique),

        "suspicious_seeds":
            int(suspicious_seed_count),

        "candidate_cases":
            int(candidate_cases),

        "evidence_records":
            int(evidence_records)
    },

    "case_evaluation": {

        "structurally_complete_cases":
            int(complete_cases),

        "case_completeness_rate_percent":
            round(
                case_completeness_rate,
                4
            ),

        "cases_with_context":
            int(cases_with_context),

        "context_enrichment_rate_percent":
            round(
                context_enrichment_rate,
                4
            ),

        "suspicious_seed_evidence_records":
            int(len(seed_evidence)),

        "contextual_evidence_records":
            int(context_evidence_records),

        "context_evidence_ratio_percent":
            round(
                context_evidence_ratio,
                4
            ),

        "average_evidence_per_case":
            round(
                average_evidence_per_case,
                4
            ),

        "average_contextual_evidence_per_case":
            round(
                average_context_per_case,
                4
            )
    },

    "traceability": {

        "seed_traceability_percent":
            round(
                seed_traceability,
                4
            ),

        "evidence_traceability_percent":
            round(
                evidence_traceability,
                4
            ),

        "finding_traceability_percent":
            round(
                finding_traceability,
                4
            )
    },

    "multi_seed_analysis": {

        "multi_seed_cases":
            int(multi_seed_cases),

        "multi_seed_case_rate_percent":
            round(
                multi_seed_rate,
                4
            )
    },

    "window_selection": {

        "selected_window_steps":
            FINAL_WINDOW,

        "cases_with_context_at_selected_window":
            selected_cases_with_context,

        "context_enrichment_at_selected_window_percent":
            selected_enrichment_percent
    },

    "consistency_check": {

        "step8_step9_consistency":
            consistency_status,

        "case_count_consistent":
            case_consistency,

        "context_case_count_consistent":
            context_case_consistency,

        "context_transaction_count_consistent":
            context_transaction_consistency,

        "multi_seed_count_consistent":
            multi_seed_consistency
    },

    "interpretation": {

        "important_limitation":
            "PaySim provides transactional evidence only and "
            "does not contain external digital forensic artifacts "
            "such as device, IP, email, authentication or CCTV data.",

        "case_completeness_interpretation":
            "Structural completeness indicates that generated "
            "case records contain the required case, seed, "
            "evidence and finding references. It does not establish "
            "completeness of real-world forensic evidence.",

        "context_interpretation":
            "Contextual evidence coverage remains limited even "
            "after using the selected temporal window.",

        "seed_interpretation":
            "A suspicious seed represents an analytically flagged "
            "transaction and should not be interpreted as proof "
            "of criminal activity."
    }
}


with open(
    FINAL_REPORT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        report,
        file,
        indent=4
    )


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STEP 9 COMPLETE"
)

print(
    "=" * 70
)

print(
    "\nFinal forensic evaluation:"
)

print(
    f"  Suspicious seeds:              "
    f"{suspicious_seed_count:,}"
)

print(
    f"  Candidate cases:               "
    f"{candidate_cases:,}"
)

print(
    f"  Complete cases:                "
    f"{complete_cases:,}"
)

print(
    f"  Case completeness:             "
    f"{case_completeness_rate:.2f}%"
)

print(
    f"  Cases with context:            "
    f"{cases_with_context:,}"
)

print(
    f"  Context enrichment:            "
    f"{context_enrichment_rate:.2f}%"
)

print(
    f"  Evidence records:              "
    f"{evidence_records:,}"
)

print(
    f"  Seed evidence:                 "
    f"{len(seed_evidence):,}"
)

print(
    f"  Context evidence:              "
    f"{context_evidence_records:,}"
)

print(
    f"  Context evidence ratio:        "
    f"{context_evidence_ratio:.2f}%"
)

print(
    f"  Avg evidence/case:             "
    f"{average_evidence_per_case:.4f}"
)

print(
    f"  Avg context/case:              "
    f"{average_context_per_case:.4f}"
)

print(
    f"  Seed traceability:             "
    f"{seed_traceability:.2f}%"
)

print(
    f"  Evidence traceability:         "
    f"{evidence_traceability:.2f}%"
)

print(
    f"  Finding traceability:          "
    f"{finding_traceability:.2f}%"
)

print(
    f"  Multi-seed cases:              "
    f"{multi_seed_cases:,}"
)

print(
    f"  Final window:                  "
    f"+/- {FINAL_WINDOW} steps"
)

print(
    f"  Final threshold:               "
    f"{FINAL_THRESHOLD}"
)

print(
    f"  Step 8 -> Step 9 consistency: "
    f"{consistency_status}"
)

print(
    "\nFiles created:"
)

print(
    f"  {FINAL_RESULTS_FILE}"
)

print(
    f"  {FINAL_REPORT_FILE}"
)

print(
    "\nResearch interpretation:"
)

print(
    "  Structural case completeness and source traceability "
    "are evaluated independently from contextual evidence coverage."
)

print(
    "  Contextual evidence is counted only when evidence_role "
    "is exactly 'contextual_evidence'."
)

print(
    "  Suspicious seed evidence is counted separately from "
    "contextual evidence."
)

print(
    "  A suspicious transaction is an analytical seed, not "
    "proof of criminal activity."
)

print(
    "\n" + "=" * 70
)