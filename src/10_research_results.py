import json
import os
import pandas as pd


# ============================================================
# STEP 10: RESEARCH RESULTS & FINDINGS
# ============================================================

MODEL_REPORT = (
    "outputs/reports/model_training_audited_report.json"
)

MODEL_COMPARISON = (
    "outputs/reports/model_comparison_audited.csv"
)

THRESHOLD_REPORT = (
    "outputs/reports/step5_threshold_report.json"
)

WINDOW_REPORT = (
    "outputs/reports/window_sensitivity_results.csv"
)

FINAL_EVALUATION = (
    "outputs/reports/final_forensic_evaluation_results.csv"
)

FINAL_EVALUATION_JSON = (
    "outputs/reports/step9_final_evaluation_report.json"
)

OUTPUT_DIR = "outputs/reports"

RESULTS_TABLE = (
    "outputs/reports/research_results_summary.csv"
)

FINDINGS_FILE = (
    "outputs/reports/research_findings.json"
)

RESEARCH_SUMMARY_FILE = (
    "outputs/reports/research_results_summary.txt"
)


# ============================================================
# HELPER
# ============================================================

def load_json(path, name):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def load_csv(path, name):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} not found: {path}"
        )

    return pd.read_csv(path)


def safe_divide(a, b):

    if b == 0:
        return 0.0

    return a / b


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 10: RESEARCH RESULTS & FINDINGS")
print("=" * 70)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 1. LOAD VERIFIED RESULTS
# ============================================================

print("\nLoading verified research outputs...")

model_report = load_json(
    MODEL_REPORT,
    "Audited model report"
)

model_comparison = load_csv(
    MODEL_COMPARISON,
    "Audited model comparison"
)

threshold_report = load_json(
    THRESHOLD_REPORT,
    "Threshold report"
)

window_results = load_csv(
    WINDOW_REPORT,
    "Window sensitivity results"
)

final_evaluation = load_csv(
    FINAL_EVALUATION,
    "Final forensic evaluation"
)

final_evaluation_json = load_json(
    FINAL_EVALUATION_JSON,
    "Final forensic evaluation JSON"
)


# ============================================================
# 2. DISPLAY AVAILABLE MODEL RESULTS
# ============================================================

print("\n" + "-" * 70)
print("AUDITED MODEL RESULTS")
print("-" * 70)

print(
    f"Models/configurations evaluated: "
    f"{len(model_comparison)}"
)

print(
    "\nAvailable columns:"
)

print(
    model_comparison.columns.tolist()
)


# ============================================================
# 3. IDENTIFY FINAL RANDOM FOREST RESULT
# ============================================================

# The audited Random Forest is the selected predictive model
# used for the forensic reconstruction stage.

rf_rows = model_comparison[
    model_comparison.apply(
        lambda row:
        "random forest"
        in " ".join(
            str(value).lower()
            for value in row.values
        ),
        axis=1
    )
].copy()


print(
    f"\nRandom Forest-related rows found: "
    f"{len(rf_rows)}"
)


if len(rf_rows) > 0:

    print(
        rf_rows.to_string(
            index=False
        )
    )


# ============================================================
# 4. FINAL FORENSIC RESULTS
# ============================================================

print("\n" + "-" * 70)
print("FINAL FORENSIC RESULTS")
print("-" * 70)

final_metrics = {}

for _, row in final_evaluation.iterrows():

    metric = str(
        row["metric"]
    )

    value = row["value"]

    final_metrics[metric] = value

    print(
        f"{metric}: {value}"
    )


# ============================================================
# 5. EXTRACT CORE FORENSIC METRICS
# ============================================================

suspicious_seeds = int(
    final_metrics[
        "Suspicious seeds"
    ]
)

candidate_cases = int(
    final_metrics[
        "Candidate forensic cases"
    ]
)

complete_cases = int(
    final_metrics[
        "Structurally complete cases"
    ]
)

case_completeness = float(
    final_metrics[
        "Case completeness rate"
    ]
)

context_cases = int(
    final_metrics[
        "Cases with contextual evidence"
    ]
)

context_enrichment = float(
    final_metrics[
        "Context enrichment rate"
    ]
)

total_evidence = int(
    final_metrics[
        "Total evidence records"
    ]
)

seed_evidence = int(
    final_metrics[
        "Suspicious-seed evidence records"
    ]
)

context_evidence = int(
    final_metrics[
        "Contextual evidence records"
    ]
)

context_ratio = float(
    final_metrics[
        "Context evidence ratio"
    ]
)

average_evidence = float(
    final_metrics[
        "Average evidence per case"
    ]
)

average_context = float(
    final_metrics[
        "Average contextual evidence per case"
    ]
)

seed_traceability = float(
    final_metrics[
        "Seed traceability"
    ]
)

evidence_traceability = float(
    final_metrics[
        "Evidence traceability"
    ]
)

finding_traceability = float(
    final_metrics[
        "Finding traceability"
    ]
)

multi_seed_cases = int(
    final_metrics[
        "Multi-seed cases"
    ]
)

multi_seed_rate = float(
    final_metrics[
        "Multi-seed case rate"
    ]
)

final_window = int(
    final_metrics[
        "Final reconstruction window"
    ]
)

final_threshold = float(
    final_metrics[
        "Final suspicion threshold"
    ]
)


# ============================================================
# 6. WINDOW SENSITIVITY FINDINGS
# ============================================================

print("\n" + "-" * 70)
print("WINDOW SENSITIVITY FINDINGS")
print("-" * 70)

print(
    window_results[
        [
            "window_steps",
            "candidate_cases",
            "cases_with_context",
            "context_enrichment_rate",
            "multi_seed_cases",
            "average_evidence_per_case"
        ]
    ].to_string(
        index=False
    )
)


# Convert rates to percentages for analysis.

window_analysis = window_results.copy()

window_analysis[
    "context_enrichment_percent"
] = (
    window_analysis[
        "context_enrichment_rate"
    ] * 100
)


# Find maximum contextual enrichment among
# tested configurations.

best_window_row = window_analysis.loc[
    window_analysis[
        "context_enrichment_percent"
    ].idxmax()
]

best_tested_window = int(
    best_window_row[
        "window_steps"
    ]
)

best_tested_enrichment = float(
    best_window_row[
        "context_enrichment_percent"
    ]
)


# ============================================================
# 7. THRESHOLD FINDING
# ============================================================

print("\n" + "-" * 70)
print("THRESHOLD FINDING")
print("-" * 70)

print(
    f"Selected threshold: "
    f"{final_threshold}"
)

print(
    "Threshold selection criterion: "
    "maximum validation F1"
)


# ============================================================
# 8. RESEARCH FINDINGS
# ============================================================

findings = []


# ------------------------------------------------------------
# Finding 1: Detection
# ------------------------------------------------------------

findings.append({

    "finding_id":
        "F1",

    "category":
        "Fraud Detection",

    "finding":
        "The audited Random Forest provided the predictive "
        "basis for identifying suspicious transactions for "
        "subsequent forensic case reconstruction.",

    "evidence":
        "Audited model comparison and final prediction output.",

    "interpretation":
        "The machine-learning stage was used as a seed-generation "
        "stage rather than as proof of criminal activity."
})


# ------------------------------------------------------------
# Finding 2: Suspicious seeds
# ------------------------------------------------------------

findings.append({

    "finding_id":
        "F2",

    "category":
        "Suspicious Transaction Identification",

    "finding":
        f"{suspicious_seeds:,} transactions were flagged as "
        "suspicious using the frozen threshold.",

    "evidence":
        f"Threshold = {final_threshold}; "
        f"{suspicious_seeds:,} suspicious seeds.",

    "interpretation":
        "These transactions represent analytical investigation "
        "seeds and require contextual interpretation."
})


# ------------------------------------------------------------
# Finding 3: Case construction
# ------------------------------------------------------------

findings.append({

    "finding_id":
        "F3",

    "category":
        "Forensic Case Reconstruction",

    "finding":
        f"{candidate_cases:,} candidate forensic cases were "
        "constructed from the suspicious transaction seeds.",

    "evidence":
        f"{candidate_cases:,} candidate cases generated.",

    "interpretation":
        "The reconstruction stage converted individual analytical "
        "alerts into structured case representations."
})


# ------------------------------------------------------------
# Finding 4: Structural completeness
# ------------------------------------------------------------

findings.append({

    "finding_id":
        "F4",

    "category":
        "Case Completeness",

    "finding":
        f"{complete_cases:,} of {candidate_cases:,} cases were "
        f"structurally complete, giving a completeness rate of "
        f"{case_completeness:.2f}% .",

    "evidence":
        "Each case contains case, seed, evidence and finding "
        "references required by the case representation.",

    "interpretation":
        "This demonstrates structural completeness of the "
        "generated case records, not completeness of all possible "
        "real-world forensic evidence."
})


# ------------------------------------------------------------
# Finding 5: Contextual evidence
# ------------------------------------------------------------

findings.append({

    "finding_id":
        "F5",

    "category":
        "Contextual Evidence",

    "finding":
        f"{context_cases} of {candidate_cases} cases "
        f"received additional contextual evidence, corresponding "
        f"to {context_enrichment:.2f}% contextual enrichment.",

    "evidence":
        f"±{final_window} PaySim-step reconstruction window.",

    "interpretation":
        "Most suspicious seeds remained isolated under the "
        "current account-relationship and temporal rules."
})


# ------------------------------------------------------------
# Finding 6: Evidence composition
# ------------------------------------------------------------

findings.append({

    "finding_id":
        "F6",

    "category":
        "Evidence Coverage",

    "finding":
        f"The final case collection contains {total_evidence:,} "
        f"evidence records: {seed_evidence:,} suspicious-seed "
        f"records and {context_evidence:,} contextual records.",

    "evidence":
        f"Contextual evidence ratio = {context_ratio:.2f}% .",

    "interpretation":
        "The evidence collection is dominated by the original "
        "analytical seed transactions rather than additional "
        "contextual transactions."
})


# ------------------------------------------------------------
# Finding 7: Traceability
# ------------------------------------------------------------

findings.append({

    "finding_id":
        "F7",

    "category":
        "Evidence Traceability",

    "finding":
        "Seed, evidence and finding traceability each reached "
        "100%.",

    "evidence":
        f"Seed traceability = {seed_traceability:.2f}%; "
        f"Evidence traceability = {evidence_traceability:.2f}%; "
        f"Finding traceability = {finding_traceability:.2f}%.",

    "interpretation":
        "Every generated seed, evidence transaction and "
        "investigative finding can be linked back to its recorded "
        "transactional source within the available dataset."
})


# ------------------------------------------------------------
# Finding 8: Multi-seed reconstruction
# ------------------------------------------------------------

findings.append({

    "finding_id":
        "F8",

    "category":
        "Relational Reconstruction",

    "finding":
        f"No multi-seed cases were formed; {multi_seed_cases} "
        f"multi-seed cases were identified.",

    "evidence":
        f"Multi-seed case rate = {multi_seed_rate:.2f}% .",

    "interpretation":
        "The tested account relationship and temporal rules did "
        "not connect multiple suspicious seeds into common cases."
})


# ------------------------------------------------------------
# Finding 9: Window sensitivity
# ------------------------------------------------------------

findings.append({

    "finding_id":
        "F9",

    "category":
        "Sensitivity Analysis",

    "finding":
        f"Among the tested temporal windows, ±{best_tested_window} "
        f"steps produced the highest contextual enrichment of "
        f"{best_tested_enrichment:.2f}% .",

    "evidence":
        "Window sensitivity experiment using identical suspicious "
        "seeds and account relationship rules.",

    "interpretation":
        "Increasing the tested temporal window increased the "
        "observed contextual coverage, but contextual enrichment "
        "remained limited."
})


# ============================================================
# 9. RESEARCH QUESTION ANSWER
# ============================================================

research_question = (
    "How can analytical outputs from financial fraud detection "
    "be systematically converted into evidence-linked "
    "investigative cases, and how can the completeness and "
    "traceability of those cases be evaluated?"
)


research_answer = (
    "The implemented workflow converts analytically suspicious "
    "transactions into structured forensic cases by using each "
    "suspicious transaction as an investigation seed, linking "
    "related transactional evidence through account relationships "
    "within a defined temporal window, and recording the resulting "
    "evidence and findings with source transaction references. "
    "On the PaySim dataset, the final configuration produced "
    f"{candidate_cases} candidate cases with "
    f"{case_completeness:.2f}% structural case completeness and "
    f"{seed_traceability:.2f}% seed traceability, "
    f"{evidence_traceability:.2f}% evidence traceability and "
    f"{finding_traceability:.2f}% finding traceability. "
    f"However, only {context_cases} cases ({context_enrichment:.2f}%) "
    "received additional contextual evidence, showing that the "
    "available transactional dataset and reconstruction rules "
    "provide strong source traceability but limited contextual "
    "enrichment."
)


# ============================================================
# 10. LIMITATIONS
# ============================================================

limitations = [

    {
        "limitation_id": "L1",
        "limitation":
            "PaySim is a synthetic financial transaction dataset "
            "and does not represent complete real-world banking "
            "forensic evidence."
    },

    {
        "limitation_id": "L2",
        "limitation":
            "The dataset does not contain external artifacts such "
            "as IP addresses, devices, authentication logs, emails "
            "or CCTV data."
    },

    {
        "limitation_id": "L3",
        "limitation":
            "PaySim step values represent simulated temporal steps "
            "rather than exact real-world timestamps."
    },

    {
        "limitation_id": "L4",
        "limitation":
            "The current reconstruction approach is based on "
            "transactional account relationships and a temporal "
            "window."
    },

    {
        "limitation_id": "L5",
        "limitation":
            "Structural case completeness should not be interpreted "
            "as complete real-world forensic evidence coverage."
    },

    {
        "limitation_id": "L6",
        "limitation":
            "A suspicious analytical seed does not establish that "
            "a person or account committed a crime."
    }
]


# ============================================================
# 11. FINAL RESEARCH SUMMARY
# ============================================================

summary_text = f"""
RESEARCH RESULTS SUMMARY
========================

Research Question
-----------------
{research_question}

Final Configuration
-------------------
Model: Audited Random Forest
Suspicion threshold: {final_threshold}
Reconstruction window: +/- {final_window} PaySim steps

Detection / Seed Generation
---------------------------
Suspicious transaction seeds: {suspicious_seeds}

Forensic Case Reconstruction
----------------------------
Candidate cases: {candidate_cases}
Structurally complete cases: {complete_cases}
Case completeness: {case_completeness:.2f}%

Contextual Evidence
-------------------
Cases with contextual evidence: {context_cases}
Context enrichment: {context_enrichment:.2f}%
Total evidence records: {total_evidence}
Suspicious-seed evidence: {seed_evidence}
Contextual evidence: {context_evidence}
Context evidence ratio: {context_ratio:.2f}%

Traceability
------------
Seed traceability: {seed_traceability:.2f}%
Evidence traceability: {evidence_traceability:.2f}%
Finding traceability: {finding_traceability:.2f}%

Relational Reconstruction
-------------------------
Multi-seed cases: {multi_seed_cases}
Multi-seed case rate: {multi_seed_rate:.2f}%

Research Question Answer
------------------------
{research_answer}

Important Interpretation
------------------------
The results demonstrate a reproducible method for converting
analytical fraud-detection outputs into structured,
evidence-linked transactional forensic cases.

The results do not demonstrate complete real-world forensic
investigation because PaySim does not contain external digital
artifacts. The low contextual enrichment rate is therefore
reported as an empirical limitation of the current dataset and
reconstruction configuration rather than being interpreted as
evidence that real-world investigations would have similarly
limited context.
"""


# ============================================================
# 12. SAVE FINDINGS JSON
# ============================================================

research_output = {

    "step": 10,

    "title":
        "Research Results and Findings",

    "research_question":
        research_question,

    "research_answer":
        research_answer,

    "final_configuration": {

        "model":
            "Audited Random Forest",

        "threshold":
            final_threshold,

        "temporal_window_steps":
            final_window
    },

    "core_results": {

        "suspicious_seeds":
            suspicious_seeds,

        "candidate_cases":
            candidate_cases,

        "complete_cases":
            complete_cases,

        "case_completeness_percent":
            case_completeness,

        "cases_with_context":
            context_cases,

        "context_enrichment_percent":
            context_enrichment,

        "total_evidence_records":
            total_evidence,

        "seed_evidence_records":
            seed_evidence,

        "contextual_evidence_records":
            context_evidence,

        "context_evidence_ratio_percent":
            context_ratio,

        "average_evidence_per_case":
            average_evidence,

        "average_contextual_evidence_per_case":
            average_context,

        "seed_traceability_percent":
            seed_traceability,

        "evidence_traceability_percent":
            evidence_traceability,

        "finding_traceability_percent":
            finding_traceability,

        "multi_seed_cases":
            multi_seed_cases
    },

    "findings":
        findings,

    "limitations":
        limitations
}


with open(
    FINDINGS_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        research_output,
        f,
        indent=4
    )


# ============================================================
# 13. SAVE RESULTS TABLE
# ============================================================

results_rows = [

    [
        "Suspicious seeds",
        suspicious_seeds
    ],

    [
        "Candidate forensic cases",
        candidate_cases
    ],

    [
        "Structurally complete cases",
        complete_cases
    ],

    [
        "Case completeness (%)",
        case_completeness
    ],

    [
        "Cases with contextual evidence",
        context_cases
    ],

    [
        "Context enrichment (%)",
        context_enrichment
    ],

    [
        "Total evidence records",
        total_evidence
    ],

    [
        "Suspicious-seed evidence",
        seed_evidence
    ],

    [
        "Contextual evidence",
        context_evidence
    ],

    [
        "Context evidence ratio (%)",
        context_ratio
    ],

    [
        "Average evidence/case",
        average_evidence
    ],

    [
        "Average contextual evidence/case",
        average_context
    ],

    [
        "Seed traceability (%)",
        seed_traceability
    ],

    [
        "Evidence traceability (%)",
        evidence_traceability
    ],

    [
        "Finding traceability (%)",
        finding_traceability
    ],

    [
        "Multi-seed cases",
        multi_seed_cases
    ],

    [
        "Multi-seed case rate (%)",
        multi_seed_rate
    ],

    [
        "Final threshold",
        final_threshold
    ],

    [
        "Final reconstruction window",
        final_window
    ]
]


results_summary_df = pd.DataFrame(
    results_rows,
    columns=[
        "metric",
        "value"
    ]
)


results_summary_df.to_csv(
    RESULTS_TABLE,
    index=False
)


# ============================================================
# 14. SAVE TEXT SUMMARY
# ============================================================

with open(
    RESEARCH_SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        summary_text.strip()
    )


# ============================================================
# 15. DISPLAY FINAL FINDINGS
# ============================================================

print("\n" + "=" * 70)
print("STEP 10 COMPLETE")
print("=" * 70)

print("\nFINAL RESEARCH RESULTS")

print(
    f"\nSuspicious seeds: "
    f"{suspicious_seeds:,}"
)

print(
    f"Candidate cases: "
    f"{candidate_cases:,}"
)

print(
    f"Case completeness: "
    f"{case_completeness:.2f}%"
)

print(
    f"Cases with contextual evidence: "
    f"{context_cases:,}"
)

print(
    f"Context enrichment: "
    f"{context_enrichment:.2f}%"
)

print(
    f"Total evidence records: "
    f"{total_evidence:,}"
)

print(
    f"Seed evidence: "
    f"{seed_evidence:,}"
)

print(
    f"Contextual evidence: "
    f"{context_evidence:,}"
)

print(
    f"Context evidence ratio: "
    f"{context_ratio:.2f}%"
)

print(
    f"Average evidence/case: "
    f"{average_evidence:.4f}"
)

print(
    f"Seed traceability: "
    f"{seed_traceability:.2f}%"
)

print(
    f"Evidence traceability: "
    f"{evidence_traceability:.2f}%"
)

print(
    f"Finding traceability: "
    f"{finding_traceability:.2f}%"
)

print(
    f"Multi-seed cases: "
    f"{multi_seed_cases}"
)

print(
    f"\nSelected reconstruction window: "
    f"+/- {final_window} steps"
)

print(
    f"Selected suspicion threshold: "
    f"{final_threshold}"
)


print("\nRESEARCH QUESTION ANSWER")
print("------------------------")
print(
    research_answer
)


print("\nIMPORTANT LIMITATIONS")
print("---------------------")

for limitation in limitations:

    print(
        f"- {limitation['limitation']}"
    )


print("\nFiles created:")

print(
    f"  {RESULTS_TABLE}"
)

print(
    f"  {FINDINGS_FILE}"
)

print(
    f"  {RESEARCH_SUMMARY_FILE}"
)


print("\n" + "=" * 70)