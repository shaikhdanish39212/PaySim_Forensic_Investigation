import os
import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd


# ============================================================
# STEP 6: CONTEXTUAL EVIDENCE & FORENSIC CASE RECONSTRUCTION
#
# PaySim Financial Fraud Investigation Research
# ============================================================
#
# IMPORTANT:
#
# A machine-learning flagged transaction is treated as a
# SUSPICIOUS SEED, not as proven fraud.
#
# Contextual evidence is retrieved from transactions that:
#
#   1. involve the same source/destination accounts, AND
#   2. occur within a configurable temporal window.
#
# Initial window:
#
#       +/- 24 PaySim steps
#
# Since PaySim's step represents one simulated hour, this is
# approximately a +/- 24-hour contextual window.
#
# The resulting objects are called:
#
#       CANDIDATE FORENSIC CASES
#
# not confirmed criminal cases.
# ============================================================


# ------------------------------------------------------------
# INPUTS
# ------------------------------------------------------------

PREDICTION_FILE = (
    "outputs/predictions/"
    "paysim_transaction_predictions.csv"
)

RAW_TRANSACTION_FILE = (
    "data/processed/"
    "paysim_preprocessed.csv"
)

# ------------------------------------------------------------
# OUTPUTS
# ------------------------------------------------------------

OUTPUT_DIR = "outputs/cases"
REPORT_DIR = "outputs/reports"

CASE_SUMMARY_FILE = (
    OUTPUT_DIR +
    "/candidate_forensic_cases.csv"
)

CASE_EVIDENCE_FILE = (
    OUTPUT_DIR +
    "/case_evidence.csv"
)

CASE_SEEDS_FILE = (
    OUTPUT_DIR +
    "/case_seeds.csv"
)

REPORT_FILE = (
    REPORT_DIR +
    "/step6_case_reconstruction_report.json"
)

# ------------------------------------------------------------
# RESEARCH CONFIGURATION
# ------------------------------------------------------------

CONTEXT_WINDOW = 24

MIN_CASE_TRANSACTIONS = 2

# Maximum number of context rows retained for a single seed.
#
# This is only a safety guard against unusually high-activity
# accounts. It does NOT change the conceptual methodology.
MAX_CONTEXT_PER_SEED = 5000

# ------------------------------------------------------------
# REQUIRED COLUMNS
# ------------------------------------------------------------

TRANSACTION_COLUMNS = [
    "transaction_id",
    "original_row_number",
    "step",
    "type",
    "amount",
    "nameOrig",
    "nameDest",
    "isFraud"
]


# ------------------------------------------------------------
# CREATE OUTPUT DIRECTORIES
# ------------------------------------------------------------

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)


print("=" * 70)
print("STEP 6: CONTEXTUAL EVIDENCE & FORENSIC CASE RECONSTRUCTION")
print("=" * 70)

start_time = time.time()


# ============================================================
# 1. LOAD SUSPICIOUS TRANSACTION PREDICTIONS
# ============================================================

print("\nLoading transaction-level predictions...")

predictions = pd.read_csv(
    PREDICTION_FILE
)

print(
    f"Prediction rows: "
    f"{len(predictions):,}"
)

# Only transactions classified as suspicious
seeds = predictions[
    predictions["predicted_suspicious"] == 1
].copy()

print(
    f"Suspicious seeds: "
    f"{len(seeds):,}"
)

if len(seeds) == 0:
    raise RuntimeError(
        "No suspicious transactions were found. "
        "Case reconstruction cannot continue."
    )


# ============================================================
# 2. PREPARE SEED INFORMATION
# ============================================================

seed_records = []

for row in seeds.itertuples(index=False):

    seed_records.append(
        {
            "seed_transaction_id":
                row.transaction_id,

            "seed_original_row":
                int(row.original_row_number),

            "seed_step":
                int(row.step),

            "seed_type":
                row.type,

            "seed_amount":
                float(row.amount),

            "seed_origin":
                row.nameOrig,

            "seed_destination":
                row.nameDest,

            "seed_suspicion_score":
                float(row.suspicion_score),

            "seed_isFraud":
                int(row.isFraud)
        }
    )


# ============================================================
# 3. CREATE ACCOUNT -> SEED INDEX
# ============================================================
#
# This allows us to efficiently identify which suspicious
# seeds are associated with an account.
# ============================================================

account_to_seeds = defaultdict(set)

for seed_index, seed in enumerate(seed_records):

    account_to_seeds[
        seed["seed_origin"]
    ].add(seed_index)

    account_to_seeds[
        seed["seed_destination"]
    ].add(seed_index)


seed_accounts = set(
    account_to_seeds.keys()
)

print(
    f"Unique accounts associated with seeds: "
    f"{len(seed_accounts):,}"
)


# ============================================================
# 4. DETERMINE GLOBAL TIME RANGE
# ============================================================

minimum_seed_step = min(
    seed["seed_step"]
    for seed in seed_records
)

maximum_seed_step = max(
    seed["seed_step"]
    for seed in seed_records
)

global_min_step = max(
    1,
    minimum_seed_step - CONTEXT_WINDOW
)

global_max_step = (
    maximum_seed_step + CONTEXT_WINDOW
)

print("\nContext configuration:")
print(
    f"Window: +/- {CONTEXT_WINDOW} steps"
)

print(
    f"Seed step range: "
    f"{minimum_seed_step} -> "
    f"{maximum_seed_step}"
)

print(
    f"Global context range: "
    f"{global_min_step} -> "
    f"{global_max_step}"
)


# ============================================================
# 5. STREAM ORIGINAL TRANSACTION DATA
# ============================================================
#
# We do NOT load the complete 6.36M-row dataset into memory
# unnecessarily.
#
# Instead, the CSV is processed in chunks and only transactions
# involving seed-related accounts are retained.
# ============================================================

print(
    "\nScanning PaySim transactions for contextual evidence..."
)

USECOLS = [
    "transaction_id",
    "original_row_number",
    "step",
    "type",
    "amount",
    "nameOrig",
    "nameDest",
    "isFraud"
]

CHUNK_SIZE = 250_000

context_chunks = []

rows_scanned = 0
rows_matching_accounts = 0

for chunk_number, chunk in enumerate(
    pd.read_csv(
        RAW_TRANSACTION_FILE,
        usecols=USECOLS,
        chunksize=CHUNK_SIZE
    ),
    start=1
):

    rows_scanned += len(chunk)

    # First restrict by global temporal range
    chunk = chunk[
        (chunk["step"] >= global_min_step)
        &
        (chunk["step"] <= global_max_step)
    ]

    if len(chunk) == 0:
        continue

    # Keep transactions involving at least one seed account
    account_mask = (
        chunk["nameOrig"].isin(seed_accounts)
        |
        chunk["nameDest"].isin(seed_accounts)
    )

    matched = chunk[
        account_mask
    ].copy()

    if len(matched) > 0:

        context_chunks.append(
            matched
        )

        rows_matching_accounts += len(
            matched
        )

    if chunk_number % 5 == 0:

        print(
            f"  Processed ~"
            f"{rows_scanned:,} rows..."
        )


if len(context_chunks) == 0:

    raise RuntimeError(
        "No contextual transactions were found."
    )


context_df = pd.concat(
    context_chunks,
    ignore_index=True
)

del context_chunks


print(
    f"\nContext candidate transactions: "
    f"{len(context_df):,}"
)


# ============================================================
# 6. BUILD CONTEXT FOR EACH SEED
# ============================================================
#
# Relationship rule:
#
# A transaction R is contextual evidence for seed S if:
#
#   (R.origin == S.origin)
# OR
#   (R.destination == S.destination)
# OR
#   (R.origin == S.destination)
# OR
#   (R.destination == S.origin)
#
# AND
#
#   abs(R.step - S.step) <= CONTEXT_WINDOW
#
# This captures direct account relationships around the
# suspicious transaction.
# ============================================================

print(
    "\nBuilding seed-level contextual evidence..."
)

seed_evidence = {}

for seed_index, seed in enumerate(
    seed_records
):

    seed_origin = seed[
        "seed_origin"
    ]

    seed_destination = seed[
        "seed_destination"
    ]

    seed_step = seed[
        "seed_step"
    ]

    minimum_step = max(
        1,
        seed_step - CONTEXT_WINDOW
    )

    maximum_step = (
        seed_step + CONTEXT_WINDOW
    )

    # Restrict candidate rows to seed time window
    candidates = context_df[
        (context_df["step"] >= minimum_step)
        &
        (context_df["step"] <= maximum_step)
    ]

    # Direct account relationship
    related_mask = (
        (candidates["nameOrig"] == seed_origin)
        |
        (candidates["nameDest"] == seed_origin)
        |
        (candidates["nameOrig"] == seed_destination)
        |
        (candidates["nameDest"] == seed_destination)
    )

    related = candidates[
        related_mask
    ].copy()

    # Make sure seed itself is present
    if seed[
        "seed_transaction_id"
    ] not in set(
        related["transaction_id"]
    ):

        seed_row = context_df[
            context_df["transaction_id"]
            ==
            seed["seed_transaction_id"]
        ]

        if len(seed_row) > 0:

            related = pd.concat(
                [
                    related,
                    seed_row
                ],
                ignore_index=True
            )

    # Remove duplicates
    related = related.drop_duplicates(
        subset=[
            "transaction_id"
        ]
    )

    # Safety limit for extremely active accounts
    if len(related) > MAX_CONTEXT_PER_SEED:

        # Keep the seed and the closest transactions
        related["distance_from_seed"] = (
            related["step"] - seed_step
        ).abs()

        seed_rows = related[
            related["transaction_id"]
            ==
            seed["seed_transaction_id"]
        ]

        other_rows = related[
            related["transaction_id"]
            !=
            seed["seed_transaction_id"]
        ].sort_values(
            "distance_from_seed"
        )

        remaining_slots = (
            MAX_CONTEXT_PER_SEED
            - len(seed_rows)
        )

        related = pd.concat(
            [
                seed_rows,
                other_rows.head(
                    remaining_slots
                )
            ],
            ignore_index=True
        )

        related = related.drop(
            columns=[
                "distance_from_seed"
            ],
            errors="ignore"
        )

    seed_evidence[
        seed_index
    ] = related

    if (
        seed_index + 1
    ) % 50 == 0:

        print(
            f"  Processed "
            f"{seed_index + 1}/"
            f"{len(seed_records)} seeds..."
        )


# ============================================================
# 7. BUILD EVIDENCE ACCOUNT SETS
# ============================================================

print(
    "\nBuilding account relationships..."
)

seed_case_accounts = {}

for seed_index, evidence in seed_evidence.items():

    accounts = set()

    accounts.update(
        evidence["nameOrig"]
        .dropna()
        .tolist()
    )

    accounts.update(
        evidence["nameDest"]
        .dropna()
        .tolist()
    )

    seed_case_accounts[
        seed_index
    ] = accounts


# ============================================================
# 8. UNION-FIND FOR CONNECTED SEEDS
# ============================================================
#
# If two suspicious seeds participate in overlapping
# contextual account networks, they can belong to the same
# candidate investigative case.
# ============================================================

print(
    "\nGrouping connected suspicious seeds..."
)

parent = list(
    range(len(seed_records))
)

rank = [
    0
] * len(seed_records)


def find_root(x):

    while parent[x] != x:

        parent[x] = parent[
            parent[x]
        ]

        x = parent[x]

    return x


def union_sets(a, b):

    root_a = find_root(a)
    root_b = find_root(b)

    if root_a == root_b:
        return

    if rank[root_a] < rank[root_b]:

        parent[root_a] = root_b

    elif rank[root_a] > rank[root_b]:

        parent[root_b] = root_a

    else:

        parent[root_b] = root_a

        rank[root_a] += 1


# Account -> seed index mapping
evidence_account_to_seeds = defaultdict(
    list
)

for seed_index, accounts in (
    seed_case_accounts.items()
):

    for account in accounts:

        evidence_account_to_seeds[
            account
        ].append(seed_index)


# Connect seeds sharing contextual accounts
for account, indices in (
    evidence_account_to_seeds.items()
):

    if len(indices) <= 1:
        continue

    first = indices[0]

    for other in indices[1:]:

        union_sets(
            first,
            other
        )


# ============================================================
# 9. CREATE CASE GROUPS
# ============================================================

case_groups = defaultdict(
    list
)

for seed_index in range(
    len(seed_records)
):

    root = find_root(
        seed_index
    )

    case_groups[
        root
    ].append(
        seed_index
    )

print(
    f"Candidate case groups: "
    f"{len(case_groups):,}"
)


# ============================================================
# 10. CREATE CASE IDs
# ============================================================

case_id_map = {}

for case_number, (
    root,
    indices
) in enumerate(
    sorted(
        case_groups.items(),
        key=lambda item: min(
            item[1]
        )
    ),
    start=1
):

    case_id = (
        f"CF{case_number:05d}"
    )

    for seed_index in indices:

        case_id_map[
            seed_index
        ] = case_id


# ============================================================
# 11. BUILD CASE EVIDENCE TABLE
# ============================================================

print(
    "\nCreating case evidence table..."
)

evidence_rows = []

seen_case_transactions = set()

for seed_index, evidence in (
    seed_evidence.items()
):

    case_id = case_id_map[
        seed_index
    ]

    seed_transaction_id = (
        seed_records[
            seed_index
        ][
            "seed_transaction_id"
        ]
    )

    for row in evidence.itertuples(
        index=False
    ):

        transaction_id = (
            row.transaction_id
        )

        unique_key = (
            case_id,
            transaction_id
        )

        # Avoid duplicate evidence records
        # inside the same case.
        if unique_key in seen_case_transactions:

            continue

        seen_case_transactions.add(
            unique_key
        )

        if (
            transaction_id
            ==
            seed_transaction_id
        ):

            evidence_role = (
                "SUSPICIOUS_SEED"
            )

        else:

            evidence_role = (
                "CONTEXTUAL_EVIDENCE"
            )

        evidence_rows.append(
            {
                "case_id":
                    case_id,

                "transaction_id":
                    transaction_id,

                "original_row_number":
                    int(
                        row.original_row_number
                    ),

                "evidence_role":
                    evidence_role,

                "step":
                    int(row.step),

                "type":
                    row.type,

                "amount":
                    float(row.amount),

                "nameOrig":
                    row.nameOrig,

                "nameDest":
                    row.nameDest,

                "isFraud":
                    int(row.isFraud),

                "seed_transaction_id":
                    seed_transaction_id
            }
        )


case_evidence_df = pd.DataFrame(
    evidence_rows
)

# Sort
case_evidence_df = (
    case_evidence_df
    .sort_values(
        [
            "case_id",
            "step",
            "transaction_id"
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# 12. BUILD CASE SUMMARY
# ============================================================

print(
    "\nCreating candidate forensic case summaries..."
)

case_summary_rows = []
case_seed_rows = []


for case_id, group in (
    case_evidence_df.groupby(
        "case_id",
        sort=True
    )
):

    seed_group = group[
        group["evidence_role"]
        ==
        "SUSPICIOUS_SEED"
    ]

    seed_transaction_ids = (
        seed_group[
            "transaction_id"
        ]
        .drop_duplicates()
        .tolist()
    )

    all_transaction_ids = (
        group[
            "transaction_id"
        ]
        .drop_duplicates()
        .tolist()
    )

    accounts = set()

    accounts.update(
        group["nameOrig"]
        .dropna()
        .tolist()
    )

    accounts.update(
        group["nameDest"]
        .dropna()
        .tolist()
    )

    start_step = int(
        group["step"].min()
    )

    end_step = int(
        group["step"].max()
    )

    transaction_count = (
        group[
            "transaction_id"
        ].nunique()
    )

    seed_count = (
        len(seed_transaction_ids)
    )

    context_count = (
        transaction_count
        -
        seed_count
    )

    unique_account_count = (
        len(accounts)
    )

    total_amount = float(
        group[
            "amount"
        ].sum()
    )

    maximum_suspicion = 0.0

    # Match seed scores from prediction file
    case_seed_predictions = predictions[
        predictions[
            "transaction_id"
        ].isin(
            seed_transaction_ids
        )
    ]

    if len(case_seed_predictions) > 0:

        maximum_suspicion = float(
            case_seed_predictions[
                "suspicion_score"
            ].max()
        )

    # --------------------------------------------------------
    # Descriptive investigative finding
    # --------------------------------------------------------

    if transaction_count >= 2:

        finding = (
            "The available transaction evidence "
            "indicates a suspicious transactional "
            "pattern requiring further investigation."
        )

    else:

        finding = (
            "A suspicious transaction was identified; "
            "additional evidence is required for "
            "further investigation."
        )

    # --------------------------------------------------------
    # Case summary
    # --------------------------------------------------------

    case_summary_rows.append(
        {
            "case_id":
                case_id,

            "seed_count":
                seed_count,

            "evidence_transaction_count":
                transaction_count,

            "context_transaction_count":
                context_count,

            "unique_account_count":
                unique_account_count,

            "start_step":
                start_step,

            "end_step":
                end_step,

            "time_span_steps":
                end_step - start_step,

            "total_transaction_amount":
                total_amount,

            "maximum_seed_suspicion_score":
                maximum_suspicion,

            "seed_transaction_ids":
                ";".join(
                    seed_transaction_ids
                ),

            "evidence_transaction_ids":
                ";".join(
                    all_transaction_ids
                ),

            "finding":
                finding
        }
    )

    # --------------------------------------------------------
    # Case seed table
    # --------------------------------------------------------

    for seed_transaction_id in (
        seed_transaction_ids
    ):

        seed_row = predictions[
            predictions[
                "transaction_id"
            ]
            ==
            seed_transaction_id
        ]

        if len(seed_row) == 0:
            continue

        row = seed_row.iloc[0]

        case_seed_rows.append(
            {
                "case_id":
                    case_id,

                "transaction_id":
                    seed_transaction_id,

                "original_row_number":
                    int(
                        row[
                            "original_row_number"
                        ]
                    ),

                "step":
                    int(
                        row["step"]
                    ),

                "type":
                    row["type"],

                "amount":
                    float(
                        row["amount"]
                    ),

                "nameOrig":
                    row["nameOrig"],

                "nameDest":
                    row["nameDest"],

                "suspicion_score":
                    float(
                        row[
                            "suspicion_score"
                        ]
                    ),

                "isFraud":
                    int(
                        row["isFraud"]
                    )
            }
        )


case_summary_df = pd.DataFrame(
    case_summary_rows
)

case_seed_df = pd.DataFrame(
    case_seed_rows
)


# ============================================================
# 13. CASE QUALITY SUMMARY
# ============================================================

candidate_case_count = (
    len(case_summary_df)
)

cases_with_context = int(
    (
        case_summary_df[
            "context_transaction_count"
        ]
        > 0
    ).sum()
)

cases_with_multiple_seeds = int(
    (
        case_summary_df[
            "seed_count"
        ]
        > 1
    ).sum()
)

total_seed_count = (
    len(seed_records)
)

total_evidence_transactions = (
    len(
        case_evidence_df[
            "transaction_id"
        ].drop_duplicates()
    )
)

# ============================================================
# 14. SAVE OUTPUTS
# ============================================================

print(
    "\nSaving candidate forensic cases..."
)

case_summary_df.to_csv(
    CASE_SUMMARY_FILE,
    index=False
)

case_evidence_df.to_csv(
    CASE_EVIDENCE_FILE,
    index=False
)

case_seed_df.to_csv(
    CASE_SEEDS_FILE,
    index=False
)

# ============================================================
# 15. SAVE REPORT
# ============================================================

report = {

    "experiment":
        "Step 6 - Contextual Evidence "
        "and Candidate Forensic Case Reconstruction",

    "input": {

        "prediction_file":
            PREDICTION_FILE,

        "transaction_file":
            RAW_TRANSACTION_FILE
    },

    "configuration": {

        "context_window_steps":
            CONTEXT_WINDOW,

        "context_window_description":
            "Approximately +/- 24 simulated hours "
            "because PaySim step represents one hour",

        "minimum_case_transactions":
            MIN_CASE_TRANSACTIONS,

        "maximum_context_per_seed":
            MAX_CONTEXT_PER_SEED
    },

    "seeds": {

        "total_suspicious_seeds":
            total_seed_count,

        "seed_step_min":
            minimum_seed_step,

        "seed_step_max":
            maximum_seed_step
    },

    "cases": {

        "candidate_case_count":
            candidate_case_count,

        "cases_with_context":
            cases_with_context,

        "cases_with_multiple_seeds":
            cases_with_multiple_seeds,

        "unique_evidence_transactions":
            total_evidence_transactions
    },

    "relationship_rule": {

        "same_origin":
            True,

        "same_destination":
            True,

        "origin_destination_cross_relation":
            True,

        "temporal_window":
            CONTEXT_WINDOW
    },

    "terminology": {

        "case_type":
            "Candidate forensic case",

        "seed_definition":
            "ML-flagged suspicious transaction",

        "finding_definition":
            "Suspicious transactional pattern "
            "requiring further investigation",

        "guilt_claim":
            False
    },

    "traceability_preparation": {

        "transaction_ids_preserved":
            True,

        "original_row_numbers_preserved":
            True,

        "source_transaction_references_preserved":
            True
    },

    "outputs": {

        "case_summary":
            CASE_SUMMARY_FILE,

        "case_evidence":
            CASE_EVIDENCE_FILE,

        "case_seeds":
            CASE_SEEDS_FILE
    }
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
# 16. DISPLAY SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 6 COMPLETE")
print("=" * 70)

print(
    f"\nSuspicious seeds:"
    f" {total_seed_count:,}"
)

print(
    f"Candidate forensic cases:"
    f" {candidate_case_count:,}"
)

print(
    f"Cases containing contextual evidence:"
    f" {cases_with_context:,}"
)

print(
    f"Cases with multiple suspicious seeds:"
    f" {cases_with_multiple_seeds:,}"
)

print(
    f"Unique evidence transactions:"
    f" {total_evidence_transactions:,}"
)

print("\nFiles created:")

print(
    f"Case summary:"
    f" {CASE_SUMMARY_FILE}"
)

print(
    f"Case evidence:"
    f" {CASE_EVIDENCE_FILE}"
)

print(
    f"Case seeds:"
    f" {CASE_SEEDS_FILE}"
)

print(
    f"Report:"
    f" {REPORT_FILE}"
)

print(
    f"\nExecution time:"
    f" {time.time() - start_time:.2f} seconds"
)

print("\nResearch safeguards:")
print(
    "✓ Suspicious transactions treated as seeds"
)
print(
    "✓ Contextual transactions retained as evidence"
)
print(
    "✓ Source transaction IDs preserved"
)
print(
    "✓ Original row references preserved"
)
print(
    "✓ Temporal relationship explicitly recorded"
)
print(
    "✓ Account relationships explicitly recorded"
)
print(
    "✓ Candidate cases, not proven criminal cases"
)
print(
    "✓ No claim of guilt"
)

print(
    "\nNext step:"
)
print(
    "Evaluate evidence coverage, case completeness "
    "and traceability."
)