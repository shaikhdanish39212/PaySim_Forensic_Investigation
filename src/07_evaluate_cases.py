import os
import json
import pandas as pd


# ============================================================
# STEP 7: EVIDENCE COVERAGE, CASE COMPLETENESS & TRACEABILITY
#
# PaySim Financial Fraud Investigation Research
# ============================================================
#
# Purpose:
#
#   Evaluate whether reconstructed candidate cases contain:
#
#   1. Valid suspicious seed references
#   2. Valid source transaction references
#   3. Contextual evidence
#   4. Required case fields
#   5. Traceable investigative findings
#
# IMPORTANT:
#
# These are experimental forensic-workflow metrics.
# They do NOT establish criminal guilt.
#
# "Case completeness" here means structural completeness
# according to our defined case representation.
# ============================================================


# ------------------------------------------------------------
# INPUT FILES
# ------------------------------------------------------------

CASE_SUMMARY_FILE = (
    "outputs/cases/"
    "candidate_forensic_cases.csv"
)

CASE_EVIDENCE_FILE = (
    "outputs/cases/"
    "case_evidence.csv"
)

CASE_SEEDS_FILE = (
    "outputs/cases/"
    "case_seeds.csv"
)

SOURCE_FILE = (
    "data/processed/"
    "paysim_preprocessed.csv"
)


# ------------------------------------------------------------
# OUTPUT FILES
# ------------------------------------------------------------

REPORT_DIR = "outputs/reports"

TRACEABILITY_FILE = (
    REPORT_DIR +
    "/case_traceability_results.csv"
)

CASE_COMPLETENESS_FILE = (
    REPORT_DIR +
    "/case_completeness_results.csv"
)

REPORT_FILE = (
    REPORT_DIR +
    "/step7_forensic_evaluation_report.json"
)


os.makedirs(
    REPORT_DIR,
    exist_ok=True
)


print("=" * 70)
print("STEP 7: FORENSIC CASE EVALUATION")
print("=" * 70)


# ============================================================
# 1. LOAD CASE OUTPUTS
# ============================================================

print("\nLoading reconstructed cases...")

case_summary = pd.read_csv(
    CASE_SUMMARY_FILE
)

case_evidence = pd.read_csv(
    CASE_EVIDENCE_FILE
)

case_seeds = pd.read_csv(
    CASE_SEEDS_FILE
)

print(
    f"Candidate cases: "
    f"{len(case_summary):,}"
)

print(
    f"Evidence records: "
    f"{len(case_evidence):,}"
)

print(
    f"Seed records: "
    f"{len(case_seeds):,}"
)


# ============================================================
# 2. LOAD ORIGINAL PREPROCESSED SOURCE
# ============================================================
#
# We use the source transaction table to verify that every
# evidence reference actually exists.
# ============================================================

print(
    "\nLoading source transaction references..."
)

source_reference = pd.read_csv(
    SOURCE_FILE,
    usecols=[
        "transaction_id",
        "original_row_number",
        "step",
        "nameOrig",
        "nameDest"
    ]
)

print(
    f"Source transactions: "
    f"{len(source_reference):,}"
)


# ============================================================
# 3. SOURCE REFERENCE VALIDATION
# ============================================================

print(
    "\nValidating evidence transaction references..."
)

source_ids = set(
    source_reference[
        "transaction_id"
    ]
)

source_row_numbers = set(
    source_reference[
        "original_row_number"
    ]
)

case_evidence[
    "valid_transaction_id"
] = (
    case_evidence[
        "transaction_id"
    ].isin(
        source_ids
    )
)

case_evidence[
    "valid_original_row"
] = (
    case_evidence[
        "original_row_number"
    ].isin(
        source_row_numbers
    )
)

case_evidence[
    "traceable"
] = (
    case_evidence[
        "valid_transaction_id"
    ]
    &
    case_evidence[
        "valid_original_row"
    ]
)

# ------------------------------------------------------------
# Check that transaction ID and original row actually point
# to the same source record.
# ------------------------------------------------------------

source_lookup = source_reference[
    [
        "transaction_id",
        "original_row_number"
    ]
].drop_duplicates(
    subset=[
        "transaction_id"
    ]
)

case_evidence = case_evidence.merge(
    source_lookup,
    on="transaction_id",
    how="left",
    suffixes=(
        "",
        "_source"
    )
)

case_evidence[
    "source_row_matches"
] = (
    case_evidence[
        "original_row_number"
    ]
    ==
    case_evidence[
        "original_row_number_source"
    ]
)

case_evidence[
    "fully_traceable"
] = (
    case_evidence[
        "traceable"
    ]
    &
    case_evidence[
        "source_row_matches"
    ]
)


# ============================================================
# 4. SEED TRACEABILITY
# ============================================================

print(
    "\nEvaluating suspicious seed traceability..."
)

seed_ids = set(
    case_seeds[
        "transaction_id"
    ]
)

evidence_ids = set(
    case_evidence[
        "transaction_id"
    ]
)

traceable_seed_ids = (
    seed_ids
    &
    evidence_ids
)

seed_traceability_rate = (
    len(traceable_seed_ids)
    /
    len(seed_ids)
    if len(seed_ids) > 0
    else 0
)


# ============================================================
# 5. EVIDENCE TRACEABILITY
# ============================================================

total_evidence_records = len(
    case_evidence
)

traceable_evidence_records = int(
    case_evidence[
        "fully_traceable"
    ].sum()
)

evidence_traceability_rate = (
    traceable_evidence_records
    /
    total_evidence_records
    if total_evidence_records > 0
    else 0
)


# ============================================================
# 6. CASE-LEVEL TRACEABILITY
# ============================================================

print(
    "\nEvaluating case-level traceability..."
)

case_traceability_rows = []

for case_id, group in (
    case_evidence.groupby(
        "case_id",
        sort=True
    )
):

    total_records = len(group)

    traceable_records = int(
        group[
            "fully_traceable"
        ].sum()
    )

    has_seed = bool(
        (
            group[
                "evidence_role"
            ]
            ==
            "SUSPICIOUS_SEED"
        ).any()
    )

    has_context = bool(
        (
            group[
                "evidence_role"
            ]
            ==
            "CONTEXTUAL_EVIDENCE"
        ).any()
    )

    finding_exists = bool(
        case_summary[
            case_summary[
                "case_id"
            ]
            ==
            case_id
        ]["finding"]
        .fillna("")
        .str.strip()
        .ne("")
        .any()
    )

    case_fully_traceable = (
        total_records > 0
        and
        traceable_records
        ==
        total_records
        and
        has_seed
        and
        finding_exists
    )

    case_traceability_rows.append(
        {
            "case_id":
                case_id,

            "total_evidence_records":
                total_records,

            "traceable_evidence_records":
                traceable_records,

            "evidence_traceability_rate":
                (
                    traceable_records
                    /
                    total_records
                ),

            "has_suspicious_seed":
                has_seed,

            "has_contextual_evidence":
                has_context,

            "finding_exists":
                finding_exists,

            "case_fully_traceable":
                case_fully_traceable
        }
    )


case_traceability_df = pd.DataFrame(
    case_traceability_rows
)


# ============================================================
# 7. FINDING TRACEABILITY
# ============================================================
#
# Every generated case contains a finding.
#
# A finding is considered traceable when:
#
#   1. It belongs to a valid case
#   2. The case contains at least one valid source transaction
#   3. The finding is therefore linked to explicit evidence IDs
# ============================================================

print(
    "\nEvaluating finding traceability..."
)

case_ids_with_valid_evidence = set(
    case_traceability_df[
        case_traceability_df[
            "traceable_evidence_records"
        ]
        > 0
    ]["case_id"]
)

case_ids_with_findings = set(
    case_summary[
        case_summary[
            "finding"
        ]
        .fillna("")
        .str.strip()
        .ne("")
    ]["case_id"]
)

traceable_finding_cases = (
    case_ids_with_valid_evidence
    &
    case_ids_with_findings
)

total_findings = len(
    case_ids_with_findings
)

traceable_findings = len(
    traceable_finding_cases
)

finding_traceability_rate = (
    traceable_findings
    /
    total_findings
    if total_findings > 0
    else 0
)


# ============================================================
# 8. CONTEXT ENRICHMENT RATE
# ============================================================

print(
    "\nCalculating contextual evidence enrichment..."
)

total_cases = len(
    case_summary
)

cases_with_context = int(
    (
        case_summary[
            "context_transaction_count"
        ]
        > 0
    ).sum()
)

context_enrichment_rate = (
    cases_with_context
    /
    total_cases
    if total_cases > 0
    else 0
)


# ============================================================
# 9. EVIDENCE COMPOSITION
# ============================================================

seed_evidence_count = int(
    (
        case_evidence[
            "evidence_role"
        ]
        ==
        "SUSPICIOUS_SEED"
    ).sum()
)

context_evidence_count = int(
    (
        case_evidence[
            "evidence_role"
        ]
        ==
        "CONTEXTUAL_EVIDENCE"
    ).sum()
)

total_evidence_count = len(
    case_evidence
)

context_evidence_ratio = (
    context_evidence_count
    /
    total_evidence_count
    if total_evidence_count > 0
    else 0
)


# ============================================================
# 10. STRUCTURAL CASE COMPLETENESS
# ============================================================
#
# Required fields:
#
#   Case ID
#   Suspicious seed
#   Evidence transaction
#   Account information
#   Timeline
#   Finding
# ============================================================

print(
    "\nEvaluating structural case completeness..."
)

case_completeness_rows = []

for row in case_summary.itertuples(
    index=False
):

    case_id = row.case_id

    evidence_group = case_evidence[
        case_evidence[
            "case_id"
        ]
        ==
        case_id
    ]

    has_case_id = (
        pd.notna(case_id)
        and
        str(case_id).strip()
        != ""
    )

    has_seed = bool(
        (
            evidence_group[
                "evidence_role"
            ]
            ==
            "SUSPICIOUS_SEED"
        ).any()
    )

    has_evidence = (
        len(evidence_group) > 0
    )

    has_accounts = bool(
        (
            evidence_group[
                "nameOrig"
            ].notna()
            |
            evidence_group[
                "nameDest"
            ].notna()
        ).any()
    )

    has_timeline = bool(
        pd.notna(row.start_step)
        and
        pd.notna(row.end_step)
    )

    has_finding = bool(
        pd.notna(row.finding)
        and
        str(row.finding).strip()
        != ""
    )

    structurally_complete = all(
        [
            has_case_id,
            has_seed,
            has_evidence,
            has_accounts,
            has_timeline,
            has_finding
        ]
    )

    case_completeness_rows.append(
        {
            "case_id":
                case_id,

            "has_case_id":
                has_case_id,

            "has_suspicious_seed":
                has_seed,

            "has_evidence":
                has_evidence,

            "has_accounts":
                has_accounts,

            "has_timeline":
                has_timeline,

            "has_finding":
                has_finding,

            "structurally_complete":
                structurally_complete
        }
    )


case_completeness_df = pd.DataFrame(
    case_completeness_rows
)

complete_cases = int(
    case_completeness_df[
        "structurally_complete"
    ].sum()
)

case_completeness_rate = (
    complete_cases
    /
    total_cases
    if total_cases > 0
    else 0
)


# ============================================================
# 11. EVIDENCE DENSITY
# ============================================================

average_evidence_per_case = (
    total_evidence_count
    /
    total_cases
    if total_cases > 0
    else 0
)

average_context_per_case = (
    context_evidence_count
    /
    total_cases
    if total_cases > 0
    else 0
)


# ============================================================
# 12. SAVE DETAILED RESULTS
# ============================================================

print(
    "\nSaving detailed evaluation results..."
)

case_traceability_df.to_csv(
    TRACEABILITY_FILE,
    index=False
)

case_completeness_df.to_csv(
    CASE_COMPLETENESS_FILE,
    index=False
)


# ============================================================
# 13. CREATE SUMMARY REPORT
# ============================================================

report = {

    "experiment":
        "Step 7 - Evidence Coverage, "
        "Case Completeness and Traceability",

    "input": {

        "candidate_cases":
            CASE_SUMMARY_FILE,

        "case_evidence":
            CASE_EVIDENCE_FILE,

        "case_seeds":
            CASE_SEEDS_FILE,

        "source_transactions":
            SOURCE_FILE
    },

    "case_statistics": {

        "total_candidate_cases":
            int(total_cases),

        "cases_with_context":
            int(cases_with_context),

        "cases_without_context":
            int(
                total_cases
                -
                cases_with_context
            ),

        "structurally_complete_cases":
            int(complete_cases)
    },

    "evidence_statistics": {

        "total_evidence_records":
            int(total_evidence_count),

        "suspicious_seed_records":
            int(seed_evidence_count),

        "contextual_evidence_records":
            int(context_evidence_count),

        "average_evidence_per_case":
            float(average_evidence_per_case),

        "average_context_per_case":
            float(average_context_per_case)
    },

    "traceability_metrics": {

        "seed_traceability_rate":
            float(seed_traceability_rate),

        "evidence_traceability_rate":
            float(evidence_traceability_rate),

        "finding_traceability_rate":
            float(finding_traceability_rate)
    },

    "reconstruction_metrics": {

        "context_enrichment_rate":
            float(context_enrichment_rate),

        "context_evidence_ratio":
            float(context_evidence_ratio),

        "case_completeness_rate":
            float(case_completeness_rate)
    },

    "metric_definitions": {

        "seed_traceability":
            "Percentage of suspicious seed transactions "
            "that have a corresponding evidence record.",

        "evidence_traceability":
            "Percentage of evidence records whose "
            "transaction ID and original row reference "
            "both map back to the source transaction table.",

        "finding_traceability":
            "Percentage of case findings linked to at "
            "least one valid source evidence transaction.",

        "context_enrichment":
            "Percentage of candidate cases containing "
            "at least one contextual evidence transaction.",

        "case_completeness":
            "Percentage of cases containing the required "
            "structural case fields.",

        "context_evidence_ratio":
            "Contextual evidence records divided by all "
            "evidence records."
    },

    "important_limitation":
        "Evidence coverage cannot be interpreted as "
        "complete real-world forensic evidence coverage "
        "because PaySim contains transactional data only "
        "and has no external artifacts such as device, "
        "IP, authentication, email or CCTV evidence."
}


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=4
    )


# ============================================================
# 14. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("STEP 7 COMPLETE")
print("=" * 70)

print("\nCASE STATISTICS")

print(
    f"Total candidate cases:"
    f" {total_cases:,}"
)

print(
    f"Cases with contextual evidence:"
    f" {cases_with_context:,}"
)

print(
    f"Structurally complete cases:"
    f" {complete_cases:,}"
)

print("\nEVIDENCE STATISTICS")

print(
    f"Total evidence records:"
    f" {total_evidence_count:,}"
)

print(
    f"Suspicious seed records:"
    f" {seed_evidence_count:,}"
)

print(
    f"Contextual evidence records:"
    f" {context_evidence_count:,}"
)

print(
    f"Average evidence per case:"
    f" {average_evidence_per_case:.4f}"
)

print("\nTRACEABILITY")

print(
    f"Seed traceability rate:"
    f" {seed_traceability_rate:.4f}"
    f" ({seed_traceability_rate * 100:.2f}%)"
)

print(
    f"Evidence traceability rate:"
    f" {evidence_traceability_rate:.4f}"
    f" ({evidence_traceability_rate * 100:.2f}%)"
)

print(
    f"Finding traceability rate:"
    f" {finding_traceability_rate:.4f}"
    f" ({finding_traceability_rate * 100:.2f}%)"
)

print("\nRECONSTRUCTION")

print(
    f"Context enrichment rate:"
    f" {context_enrichment_rate:.4f}"
    f" ({context_enrichment_rate * 100:.2f}%)"
)

print(
    f"Context evidence ratio:"
    f" {context_evidence_ratio:.4f}"
    f" ({context_evidence_ratio * 100:.2f}%)"
)

print(
    f"Case completeness rate:"
    f" {case_completeness_rate:.4f}"
    f" ({case_completeness_rate * 100:.2f}%)"
)

print("\nFiles created:")

print(
    TRACEABILITY_FILE
)

print(
    CASE_COMPLETENESS_FILE
)

print(
    REPORT_FILE
)

print("\nResearch limitation:")
print(
    "PaySim cannot measure completeness of real-world "
    "digital forensic evidence because it lacks external "
    "forensic artifacts."
)

print("\nNext step:")
print(
    "Analyze reconstruction-window sensitivity "
    "(1, 6, 12 and 24 PaySim steps)."
)