import os
import json
import time

import pandas as pd


# ============================================================
# STEP 8: RECONSTRUCTION WINDOW SENSITIVITY ANALYSIS
#
# PaySim Financial Fraud Investigation Research
# ============================================================
#
# PURPOSE
#
# Evaluate how the temporal context window affects:
#
#   - Context enrichment
#   - Evidence volume
#   - Case reconstruction
#   - Multi-seed case formation
#
# Tested windows:
#
#   1 step
#   6 steps
#   12 steps
#   24 steps
#
# IMPORTANT:
#
# The following remain FIXED:
#
#   - Same suspicious seeds
#   - Same Random Forest model
#   - Same prediction file
#   - Same account relationship rule
#
# ONLY THE TEMPORAL WINDOW CHANGES.
#
# This makes the experiment a controlled sensitivity analysis.
# ============================================================


# ------------------------------------------------------------
# INPUTS
# ------------------------------------------------------------

PREDICTION_FILE = (
    "outputs/predictions/"
    "paysim_transaction_predictions.csv"
)

SOURCE_FILE = (
    "data/processed/"
    "paysim_preprocessed.csv"
)


# ------------------------------------------------------------
# OUTPUTS
# ------------------------------------------------------------

REPORT_DIR = "outputs/reports"

RESULT_FILE = (
    REPORT_DIR +
    "/window_sensitivity_results.csv"
)

DETAIL_FILE = (
    REPORT_DIR +
    "/window_sensitivity_case_details.csv"
)

REPORT_JSON = (
    REPORT_DIR +
    "/step8_window_sensitivity_report.json"
)


os.makedirs(
    REPORT_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# EXPERIMENT WINDOWS
# ------------------------------------------------------------

WINDOWS = [
    1,
    6,
    12,
    24
]

CHUNK_SIZE = 250_000


print("=" * 70)
print("STEP 8: RECONSTRUCTION WINDOW SENSITIVITY")
print("=" * 70)

start_time = time.time()


# ============================================================
# 1. LOAD SUSPICIOUS SEEDS
# ============================================================

print("\nLoading transaction predictions...")

predictions = pd.read_csv(
    PREDICTION_FILE
)

seeds = predictions[
    predictions[
        "predicted_suspicious"
    ] == 1
].copy()

print(
    f"Total prediction records: "
    f"{len(predictions):,}"
)

print(
    f"Suspicious seeds: "
    f"{len(seeds):,}"
)


if len(seeds) == 0:

    raise RuntimeError(
        "No suspicious seeds found."
    )


# ------------------------------------------------------------
# Seed columns
# ------------------------------------------------------------

seed_columns = [
    "transaction_id",
    "step",
    "nameOrig",
    "nameDest",
    "suspicion_score"
]

seeds = seeds[
    seed_columns
].copy()


# ============================================================
# 2. CREATE SEED ACCOUNT SET
# ============================================================

seed_accounts = set()

seed_accounts.update(
    seeds[
        "nameOrig"
    ]
    .dropna()
    .tolist()
)

seed_accounts.update(
    seeds[
        "nameDest"
    ]
    .dropna()
    .tolist()
)

print(
    f"Unique seed-associated accounts: "
    f"{len(seed_accounts):,}"
)


# ============================================================
# 3. DETERMINE GLOBAL RANGE
# ============================================================

minimum_seed_step = int(
    seeds["step"].min()
)

maximum_seed_step = int(
    seeds["step"].max()
)

maximum_window = max(
    WINDOWS
)

global_min_step = max(
    1,
    minimum_seed_step - maximum_window
)

global_max_step = (
    maximum_seed_step + maximum_window
)

print("\nGlobal source scan range:")

print(
    f"Step {global_min_step} "
    f"-> {global_max_step}"
)


# ============================================================
# 4. LOAD ONLY RELEVANT SOURCE TRANSACTIONS
# ============================================================
#
# We scan the original preprocessed dataset once.
#
# We retain transactions that:
#
#   - fall inside the global time range
#   - involve at least one seed-associated account
#
# Then all four experiments reuse this same contextual pool.
# ============================================================

print(
    "\nScanning source transactions..."
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

context_chunks = []

rows_scanned = 0

for chunk_number, chunk in enumerate(
    pd.read_csv(
        SOURCE_FILE,
        usecols=USECOLS,
        chunksize=CHUNK_SIZE
    ),
    start=1
):

    rows_scanned += len(chunk)

    # Global temporal filter
    chunk = chunk[
        (chunk["step"] >= global_min_step)
        &
        (chunk["step"] <= global_max_step)
    ]

    if len(chunk) == 0:
        continue

    # Account relationship filter
    account_mask = (
        chunk["nameOrig"].isin(
            seed_accounts
        )
        |
        chunk["nameDest"].isin(
            seed_accounts
        )
    )

    matched = chunk[
        account_mask
    ].copy()

    if len(matched) > 0:

        context_chunks.append(
            matched
        )

    if chunk_number % 5 == 0:

        print(
            f"  Processed "
            f"{rows_scanned:,} rows..."
        )


if len(context_chunks) == 0:

    raise RuntimeError(
        "No contextual transactions found."
    )


context_df = pd.concat(
    context_chunks,
    ignore_index=True
)

del context_chunks


context_df = context_df.drop_duplicates(
    subset=[
        "transaction_id"
    ]
)


print(
    f"\nContext transaction pool: "
    f"{len(context_df):,}"
)


# ============================================================
# 5. RUN EACH WINDOW
# ============================================================

results = []
detail_rows = []


for window in WINDOWS:

    print("\n" + "-" * 70)

    print(
        f"Testing temporal window: "
        f"+/- {window} PaySim steps"
    )

    # --------------------------------------------------------
    # Case construction
    # --------------------------------------------------------

    seed_case_accounts = {}
    seed_context_counts = {}

    # Each seed starts as an independent case.
    #
    # We will merge cases if their contextual evidence
    # connects their account sets.
    #
    # This is equivalent to the relationship logic used in
    # Step 6, but repeated for each window.
    # --------------------------------------------------------

    parent = list(
        range(len(seeds))
    )

    rank = [
        0
    ] * len(seeds)

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


    # --------------------------------------------------------
    # Build contextual account sets
    # --------------------------------------------------------

    for seed_index, seed in enumerate(
        seeds.itertuples(
            index=False
        )
    ):

        seed_step = int(
            seed.step
        )

        seed_origin = (
            seed.nameOrig
        )

        seed_destination = (
            seed.nameDest
        )

        minimum_step = max(
            1,
            seed_step - window
        )

        maximum_step = (
            seed_step + window
        )

        candidates = context_df[
            (context_df["step"] >= minimum_step)
            &
            (context_df["step"] <= maximum_step)
        ]

        # Same account relationship rule as Step 6
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
        ]

        accounts = set()

        accounts.update(
            related[
                "nameOrig"
            ]
            .dropna()
            .tolist()
        )

        accounts.update(
            related[
                "nameDest"
            ]
            .dropna()
            .tolist()
        )

        # Always include seed accounts
        accounts.add(
            seed_origin
        )

        accounts.add(
            seed_destination
        )

        seed_case_accounts[
            seed_index
        ] = accounts

        # Context transactions excluding the seed itself
        seed_transaction_id = (
            seed.transaction_id
        )

        context_count = (
            related[
                "transaction_id"
            ]
            .nunique()
        )

        if (
            seed_transaction_id
            in set(
                related[
                    "transaction_id"
                ]
            )
        ):
            context_count -= 1

        seed_context_counts[
            seed_index
        ] = max(
            0,
            context_count
        )


    # --------------------------------------------------------
    # Merge seeds sharing contextual accounts
    # --------------------------------------------------------

    account_to_seeds = {}

    for seed_index, accounts in (
        seed_case_accounts.items()
    ):

        for account in accounts:

            if account not in account_to_seeds:

                account_to_seeds[
                    account
                ] = []

            account_to_seeds[
                account
            ].append(
                seed_index
            )


    for account, indices in (
        account_to_seeds.items()
    ):

        if len(indices) <= 1:
            continue

        first = indices[0]

        for other in indices[1:]:

            union_sets(
                first,
                other
            )


    # --------------------------------------------------------
    # Build case groups
    # --------------------------------------------------------

    case_groups = {}

    for seed_index in range(
        len(seeds)
    ):

        root = find_root(
            seed_index
        )

        if root not in case_groups:

            case_groups[
                root
            ] = []

        case_groups[
            root
        ].append(
            seed_index
        )


    total_cases = len(
        case_groups
    )

    cases_with_context = 0

    cases_with_multiple_seeds = 0

    total_context_transactions = 0

    total_case_transactions = 0

    case_evidence_counts = []


    # --------------------------------------------------------
    # Case-level statistics
    # --------------------------------------------------------

    for root, indices in (
        case_groups.items()
    ):

        seed_count = len(
            indices
        )

        if seed_count > 1:

            cases_with_multiple_seeds += 1

        context_count = sum(
            seed_context_counts[
                index
            ]
            for index in indices
        )

        if context_count > 0:

            cases_with_context += 1

        total_context_transactions += (
            context_count
        )

        # Approximate unique case transactions:
        # seeds + contextual records
        case_transaction_count = (
            seed_count
            +
            context_count
        )

        total_case_transactions += (
            case_transaction_count
        )

        case_evidence_counts.append(
            case_transaction_count
        )


        # Save a compact detail record
        seed_ids = []

        for index in indices:

            seed_ids.append(
                seeds.iloc[
                    index
                ]["transaction_id"]
            )

        detail_rows.append(
            {
                "window_steps":
                    window,

                "case_group_root":
                    int(root),

                "seed_count":
                    seed_count,

                "context_transaction_count":
                    context_count,

                "case_transaction_count":
                    case_transaction_count,

                "seed_transaction_ids":
                    ";".join(
                        seed_ids
                    )
            }
        )


    # --------------------------------------------------------
    # Summary metrics
    # --------------------------------------------------------

    total_seeds = len(
        seeds
    )

    context_enrichment_rate = (
        cases_with_context
        /
        total_cases
        if total_cases > 0
        else 0
    )

    context_seed_rate = (
        sum(
            1
            for value in (
                seed_context_counts.values()
            )
            if value > 0
        )
        /
        total_seeds
        if total_seeds > 0
        else 0
    )

    average_evidence_per_case = (
        total_case_transactions
        /
        total_cases
        if total_cases > 0
        else 0
    )

    average_context_per_case = (
        total_context_transactions
        /
        total_cases
        if total_cases > 0
        else 0
    )

    multiple_seed_case_rate = (
        cases_with_multiple_seeds
        /
        total_cases
        if total_cases > 0
        else 0
    )

    print(
        f"Cases: "
        f"{total_cases:,}"
    )

    print(
        f"Cases with context: "
        f"{cases_with_context:,}"
    )

    print(
        f"Context enrichment: "
        f"{context_enrichment_rate * 100:.2f}%"
    )

    print(
        f"Seeds with context: "
        f"{context_seed_rate * 100:.2f}%"
    )

    print(
        f"Multi-seed cases: "
        f"{cases_with_multiple_seeds:,}"
    )

    print(
        f"Average evidence/case: "
        f"{average_evidence_per_case:.4f}"
    )


    results.append(
        {
            "window_steps":
                window,

            "window_hours_approx":
                window,

            "total_suspicious_seeds":
                total_seeds,

            "candidate_cases":
                total_cases,

            "cases_with_context":
                cases_with_context,

            "cases_without_context":
                (
                    total_cases
                    -
                    cases_with_context
                ),

            "context_enrichment_rate":
                context_enrichment_rate,

            "seeds_with_context":
                sum(
                    1
                    for value in (
                        seed_context_counts.values()
                    )
                    if value > 0
                ),

            "seed_context_rate":
                context_seed_rate,

            "multi_seed_cases":
                cases_with_multiple_seeds,

            "multi_seed_case_rate":
                multiple_seed_case_rate,

            "context_transactions":
                total_context_transactions,

            "total_case_transactions":
                total_case_transactions,

            "average_evidence_per_case":
                average_evidence_per_case,

            "average_context_per_case":
                average_context_per_case
        }
    )


# ============================================================
# 6. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    RESULT_FILE,
    index=False
)

details_df = pd.DataFrame(
    detail_rows
)

details_df.to_csv(
    DETAIL_FILE,
    index=False
)


# ============================================================
# 7. SAVE JSON REPORT
# ============================================================

report = {

    "experiment":
        "Step 8 - Reconstruction Window Sensitivity",

    "windows_tested":
        WINDOWS,

    "fixed_components": {

        "prediction_file":
            PREDICTION_FILE,

        "source_file":
            SOURCE_FILE,

        "suspicious_seed_count":
            int(len(seeds)),

        "relationship_rule":
            [
                "same origin account",
                "same destination account",
                "origin-destination cross relationship"
            ]
    },

    "variable":
        "Temporal context window in PaySim steps",

    "results":
        results,

    "interpretation_guidance":
        "Increasing the temporal window may increase "
        "contextual evidence and case connectivity, "
        "but may also introduce less directly relevant "
        "transactions. The experiment measures this "
        "trade-off descriptively.",

    "limitation":
        "PaySim provides transactional data only. "
        "The sensitivity experiment therefore evaluates "
        "transactional reconstruction, not completeness "
        "of real-world digital forensic evidence."
}

with open(
    REPORT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=4
    )


# ============================================================
# 8. FINAL DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("STEP 8 COMPLETE")
print("=" * 70)

print("\nWindow sensitivity summary:")

display_columns = [
    "window_steps",
    "candidate_cases",
    "cases_with_context",
    "context_enrichment_rate",
    "multi_seed_cases",
    "average_evidence_per_case"
]

display_df = results_df[
    display_columns
].copy()

display_df[
    "context_enrichment_rate"
] = (
    display_df[
        "context_enrichment_rate"
    ] * 100
).round(2)

print(
    display_df.to_string(
        index=False
    )
)

print("\nFiles created:")

print(
    RESULT_FILE
)

print(
    DETAIL_FILE
)

print(
    REPORT_JSON
)

print(
    f"\nExecution time: "
    f"{time.time() - start_time:.2f} seconds"
)

print("\nResearch safeguards:")
print(
    "✓ Same suspicious seeds for every window"
)
print(
    "✓ Same account relationship rule"
)
print(
    "✓ Only temporal window changes"
)
print(
    "✓ No model retraining"
)
print(
    "✓ No threshold changes"
)
print(
    "✓ Transaction identifiers preserved"
)
print(
    "✓ Results recorded for reproducibility"
)

print("\nNext step:")
print(
    "Interpret the sensitivity results and select "
    "the reconstruction configuration for the final "
    "forensic evaluation."
)