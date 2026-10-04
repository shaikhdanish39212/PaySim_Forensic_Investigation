from pathlib import Path
import json
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
CASES_DIR = OUTPUTS_DIR / "cases"


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="PaySim Digital Forensics API",
    description=(
        "Backend API for the Banking and Financial Cybercrime "
        "Investigation research demonstration."
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HELPERS
# ============================================================

def clean_records(df: pd.DataFrame):
    """
    Convert pandas values into JSON-safe Python values.
    """
    df = df.copy()

    for column in df.columns:
        if pd.api.types.is_bool_dtype(df[column]):
            df[column] = df[column].astype(bool)

    df = df.where(pd.notnull(df), None)

    return df.to_dict(orient="records")


def load_csv(path: Path):
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {path.name}",
        )

    try:
        return clean_records(pd.read_csv(path))
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to read {path.name}: {error}",
        )


def load_json(path: Path):
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {path.name}",
        )

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to read {path.name}: {error}",
        )


def get_case(case_id: str):
    path = CASES_DIR / "candidate_forensic_cases.csv"

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Candidate case file not found.",
        )

    df = pd.read_csv(path)

    result = df[df["case_id"].astype(str) == str(case_id)]

    if result.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Case {case_id} not found.",
        )

    return clean_records(result)[0]


# ============================================================
# ROOT / HEALTH
# ============================================================

@app.get("/")
def root():
    return {
        "message": "PaySim Digital Forensics API is running",
        "project": "Role of Digital Forensics in Banking and Financial Cybercrime Investigation",
        "status": "active",
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "outputs_exists": OUTPUTS_DIR.exists(),
        "reports_exists": REPORTS_DIR.exists(),
        "cases_exists": CASES_DIR.exists(),
    }


# ============================================================
# DASHBOARD
# ============================================================

@app.get("/api/dashboard")
def dashboard():
    return {
        "total_transactions": 6362620,
        "fraud_transactions": 8213,
        "suspicious_seeds": 480,
        "candidate_cases": 480,
        "evidence_records": 491,
        "contextual_cases": 11,
        "contextual_enrichment": 2.2917,
        "structural_completeness": 100.0,
        "seed_traceability": 100.0,
        "evidence_traceability": 100.0,
        "finding_traceability": 100.0,
        "forensic_threshold": 0.95,
        "selected_model": "Random Forest",
        "roc_auc": 0.9703,
        "pr_auc": 0.3153,
    }


# ============================================================
# RESEARCH / MODEL
# ============================================================

@app.get("/api/research-findings")
def research_findings():
    return load_json(REPORTS_DIR / "research_findings.json")


@app.get("/api/model-comparison")
def model_comparison():
    return load_csv(
        REPORTS_DIR / "model_comparison_audited.csv"
    )


@app.get("/api/feature-importance")
def feature_importance():
    return load_csv(
        REPORTS_DIR /
        "random_forest_audited_feature_importance.csv"
    )


# ============================================================
# CASES
# ============================================================

@app.get("/api/cases")
def cases():
    return load_csv(
        CASES_DIR / "candidate_forensic_cases.csv"
    )


@app.get("/api/cases/{case_id}")
def case_details(case_id: str):
    return get_case(case_id)


@app.get("/api/case-seeds")
def case_seeds():
    return load_csv(
        CASES_DIR / "case_seeds.csv"
    )


@app.get("/api/case-evidence")
def case_evidence():
    return load_csv(
        CASES_DIR / "case_evidence.csv"
    )


@app.get("/api/cases/{case_id}/evidence")
def case_evidence_details(case_id: str):

    path = CASES_DIR / "case_evidence.csv"

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Case evidence file not found.",
        )

    df = pd.read_csv(path)

    result = df[
        df["case_id"].astype(str) == str(case_id)
    ]

    if result.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No evidence found for {case_id}.",
        )

    return clean_records(result)


# ============================================================
# TRACEABILITY
# ============================================================

@app.get("/api/traceability")
def traceability():
    return load_csv(
        REPORTS_DIR / "case_traceability_results.csv"
    )


# ============================================================
# CASE COMPLETENESS
# ============================================================

@app.get("/api/case-completeness")
def case_completeness():
    return load_csv(
        REPORTS_DIR / "case_completeness_results.csv"
    )


# ============================================================
# WINDOW SENSITIVITY
# ============================================================

@app.get("/api/window-sensitivity")
def window_sensitivity():
    return load_csv(
        REPORTS_DIR / "window_sensitivity_results.csv"
    )


@app.get("/api/window-sensitivity-details")
def window_sensitivity_details():
    return load_csv(
        REPORTS_DIR /
        "window_sensitivity_case_details.csv"
    )


# ============================================================
# FINAL EVALUATION
# ============================================================

@app.get("/api/forensic-evaluation")
def forensic_evaluation():
    return load_csv(
        REPORTS_DIR /
        "final_forensic_evaluation_results.csv"
    )


@app.get("/api/research-summary")
def research_summary():
    return load_csv(
        REPORTS_DIR /
        "research_results_summary.csv"
    )


# ============================================================
# PROJECT INFO
# ============================================================

@app.get("/api/info")
def project_info():
    return {
        "project": "PaySim Digital Forensics Investigation",
        "research_title": (
            "Role of Digital Forensics in Banking and "
            "Financial Cybercrime Investigation"
        ),
        "dataset": "PaySim",
        "model": "Audited Random Forest",
        "model_comparison_threshold": 0.5,
        "forensic_seed_threshold": 0.95,
        "suspicious_seeds": 480,
        "candidate_cases": 480,
        "evidence_records": 491,
        "contextual_cases": 11,
        "selected_window": "±24 PaySim steps",
        "contextual_enrichment": 2.2917,
        "traceability": {
            "seed": 100.0,
            "evidence": 100.0,
            "finding": 100.0,
        },
        "important_note": (
            "Suspicious transactions are investigation seeds "
            "and do not constitute proof of criminal activity."
        ),
    }