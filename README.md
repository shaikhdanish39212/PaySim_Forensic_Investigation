# PaySim Forensic Investigation

## Role of Digital Forensics in Banking and Financial Cybercrime Investigation

A research implementation for detecting suspicious financial transactions and transforming machine-learning outputs into structured, evidence-linked forensic investigation cases.

The project combines **machine learning-based fraud detection** with a **digital forensic reconstruction workflow** using the PaySim financial transaction dataset.

---

## 📌 Project Overview

Financial fraud detection systems generally focus on identifying suspicious transactions. However, identifying a suspicious transaction is only the beginning of a forensic investigation.

This project investigates a workflow that converts analytical fraud-detection outputs into structured investigative cases through:

```text
Transaction Data
      ↓
Data Preprocessing
      ↓
Feature Engineering
      ↓
Fraud Detection
      ↓
Suspicious Transaction Seeds
      ↓
Temporal & Account-Based Reconstruction
      ↓
Forensic Cases
      ↓
Evidence Mapping
      ↓
Traceable Findings
      ↓
Forensic Evaluation
```

The system evaluates not only fraud-detection performance but also the **completeness, contextual enrichment, and traceability of reconstructed forensic cases**.

---

## 🎯 Research Objectives

The project aims to:

1. Identify suspicious financial transactions using machine learning.
2. Construct structured forensic investigation cases from suspicious transaction seeds.
3. Use temporal and account relationships to reconstruct transaction context.
4. Map investigative findings to supporting evidence records.
5. Evaluate evidence and finding traceability.
6. Measure the sensitivity of forensic reconstruction to different temporal investigation windows.

---

## 🔬 Research Question

> **How can analytical outputs from financial fraud detection be systematically converted into evidence-linked investigative cases, and how can the completeness and traceability of those cases be evaluated?**

---

## 🗃️ Dataset

### PaySim

The project uses the **PaySim synthetic financial transaction dataset**.

Dataset characteristics used in the implementation:

| Property | Value |
|---|---:|
| Total transactions | 6,362,620 |
| Fraudulent transactions | 8,213 |
| Genuine transactions | 6,354,407 |
| Fraud rate | 0.1291% |
| Original attributes | 11 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Transaction steps | 1–743 |

PaySim represents simulated mobile-money transactions and contains transaction-level financial and account information.

### Dataset availability

The raw PaySim CSV is **not included in this repository** because of its large size.

Place the dataset inside:

```text
data/raw/
```

The implementation scripts expect the dataset to be available locally before running the complete pipeline.

---

# 🧠 Methodology

The implementation follows a leakage-audited machine-learning and forensic reconstruction workflow.

## 1. Dataset Inspection

The raw dataset is inspected for:

- Dataset dimensions
- Missing values
- Duplicate records
- Transaction types
- Fraud distribution
- Account relationships
- Temporal characteristics

---

## 2. Preprocessing

The preprocessing stage:

- Preserves the original transaction information.
- Creates deterministic transaction identifiers.
- Maintains original row references.
- Verifies temporal ordering.
- Prepares the dataset for reproducible analysis.

---

## 3. Feature Engineering

Historical and transaction-level features are generated while avoiding future information leakage.

The final audited feature set excludes inappropriate variables such as:

- Target variable
- Raw account identifiers
- Balance-derived leakage variables
- Other variables that could expose future transaction information

---

## 4. Fraud Detection

Two machine-learning models were evaluated:

- Logistic Regression
- Random Forest

The dataset was divided using a **temporal train-validation-test split**.

### Temporal split

| Dataset | PaySim Steps |
|---|---|
| Training | 1–323 |
| Validation | 324–399 |
| Testing | 400–743 |

The final forensic seed-generation process uses a Random Forest model.

---

## 5. Suspicious Transaction Generation

The Random Forest model produces fraud probabilities.

A suspicion threshold of:

```text
0.95
```

was selected using the validation set based on maximum F1-score and then frozen before evaluating the test data.

Transactions exceeding this threshold are treated as **suspicious investigation seeds**, not as confirmed criminal transactions.

---

## 6. Forensic Case Reconstruction

Each suspicious transaction is converted into a structured investigation case.

Contextual transactions are searched using:

- Temporal proximity
- Same source account
- Same destination account
- Relevant source/destination relationships

The primary reconstruction window used in the final evaluation is:

```text
±24 PaySim steps
```

---

## 7. Evidence Traceability

The investigation maintains explicit relationships between:

```text
Finding
   ↓
Evidence Record
   ↓
Transaction ID
   ↓
Original PaySim Record
```

This allows every reconstructed finding to be traced back to its supporting transaction evidence.

---

# 📊 Model Results

The audited models were evaluated using accuracy, precision, recall, F1-score, ROC-AUC and PR-AUC.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 82.33% | 3.12% | 91.58% | 6.03% | 0.9426 | 0.2125 |
| Random Forest | 98.30% | 23.87% | 79.82% | 36.75% | 0.9703 | 0.3153 |

The Random Forest model achieved the strongest overall performance and was selected for the forensic reconstruction workflow.

---

# 🔎 Final Forensic Results

Using the frozen suspicion threshold of **0.95**:

| Metric | Result |
|---|---:|
| Suspicious seeds | 480 |
| Candidate forensic cases | 480 |
| Contextual evidence cases | 11 |
| Evidence records | 491 |
| Structural completeness | 100% |
| Seed traceability | 100% |
| Evidence traceability | 100% |
| Finding traceability | 100% |
| Contextual enrichment | 2.29% |
| Multi-seed cases | 0 |

The results show that all suspicious seeds were successfully converted into structurally complete cases and maintained complete source-to-finding traceability.

However, only **11 of 480 cases (2.29%)** received additional contextual transaction evidence within the tested reconstruction window.

---

# ⏱️ Temporal Window Sensitivity

The effect of different investigation windows was evaluated without retraining the model or changing the suspicion threshold.

| Window | Contextual Cases | Context Enrichment |
|---|---:|---:|
| ±1 step | 2 | 0.42% |
| ±6 steps | 3 | 0.62% |
| ±12 steps | 5 | 1.04% |
| ±24 steps | 11 | 2.29% |

The **±24-step window** produced the highest contextual enrichment among the tested windows.

This does not imply that ±24 steps is universally optimal for real-world investigations.

---

# 🖥️ Research Demonstration Website

The project also includes a web-based demonstration interface for presenting the research implementation and results.

### Technology Stack

**Frontend**
- React.js
- JavaScript (ES6)
- HTML5
- CSS3
- Tailwind CSS

**Backend**
- Python
- FastAPI

**Machine Learning / Research**
- Python
- Scikit-learn
- Pandas
- NumPy

### Main Website Sections

```text
Dashboard
    ↓
Methodology
    ↓
Fraud Detection
    ↓
Suspicious Transactions
    ↓
Forensic Cases
    ↓
Evidence Traceability
    ↓
Window Sensitivity
    ↓
Research Results
```

The website is a **research demonstration layer**. It presents the outputs of the research implementation and does not replace the underlying Python research pipeline.

---

# 📁 Project Structure

```text
PaySim_Forensic_Investigation/
│
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── ...
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   ├── package-lock.json
│   └── ...
│
├── data/
│   ├── raw/
│   └── processed/
│
├── outputs/
│   ├── models/
│   ├── predictions/
│   ├── cases/
│   ├── figures/
│   └── reports/
│
├── src/
│   ├── 01_dataset_inspection.py
│   ├── 02_preprocess.py
│   ├── 03_feature_engineering.py
│   ├── 04_train_models.py
│   ├── 05_generate_predictions.py
│   ├── 06_reconstruct_cases.py
│   ├── 07_evaluate_cases.py
│   ├── 08_window_sensitivity.py
│   ├── 09_evaluation.py
│   └── 10_research_results.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# ⚙️ Installation

## Prerequisites

Make sure the following are installed:

- Python 3.11+
- Node.js
- npm
- Git

---

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/PaySim-Forensic-Investigation.git
cd PaySim-Forensic-Investigation
```

---

# 🐍 Backend Setup

Navigate to the backend directory:

```bash
cd backend
```

Create a Python virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

The backend will normally be available at:

```text
http://127.0.0.1:8000
```

---

# ⚛️ Frontend Setup

Open another terminal.

Navigate to the frontend:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

The frontend will normally be available at:

```text
http://localhost:5173
```

---

# 🧪 Running the Research Pipeline

The research scripts are organized sequentially:

```text
01_dataset_inspection.py
        ↓
02_preprocess.py
        ↓
03_feature_engineering.py
        ↓
04_train_models.py
        ↓
05_generate_predictions.py
        ↓
06_reconstruct_cases.py
        ↓
07_evaluate_cases.py
        ↓
08_window_sensitivity.py
        ↓
09_evaluation.py
        ↓
10_research_results.py
```

Run each stage according to the dataset and output requirements described in the source code.

---

# 📌 Important Research Notes

### Suspicious ≠ Confirmed Fraud

A transaction identified by the machine-learning model is treated as a **suspicious investigation seed**.

The model does not establish criminal liability or prove that a person committed fraud.

### Structural Completeness ≠ Real-World Forensic Completeness

The reported 100% structural completeness means that the generated cases contain the required case structure and seed information.

It does not mean that a real-world investigation would have complete evidence.

### Dataset Limitation

PaySim is a synthetic dataset. It does not provide external forensic artifacts such as:

- IP addresses
- Device information
- Email records
- CCTV footage
- Authentication logs
- Bank internal investigation records

Therefore, the forensic reconstruction is limited to the transactional evidence available in PaySim.

---

# ⚠️ Limitations

1. PaySim is a simulated financial transaction dataset.
2. External forensic evidence is unavailable.
3. Contextual enrichment was relatively low at 2.29%.
4. No multi-seed investigation cases were observed.
5. The ±24-step window was selected based on the tested windows and should not be interpreted as universally optimal.
6. Machine-learning predictions are investigation seeds rather than proof of criminal activity.

---

# 🔐 Data and Security

The following should **not** be committed to the repository:

```text
.env
API keys
Passwords
Python virtual environments
node_modules
Large raw datasets
Private credentials
Large model binaries
```

The PaySim dataset should be obtained separately and placed in:

```text
data/raw/
```

---

# 📚 Research Context

This implementation supports the research study:

> **Role of Digital Forensics in Banking and Financial Cybercrime Investigation**

The research focuses on bridging the gap between:

```text
Fraud Detection
       ↓
Investigation Seeds
       ↓
Context Reconstruction
       ↓
Evidence Mapping
       ↓
Traceable Findings
```

The primary research contribution is the evaluation of this **detection-to-forensic-case workflow**, rather than simply developing another fraud-classification model.

---

# 👨‍💻 Author

**Danish Shaikh**

M.Sc. Computer Science, Semester I  
Thakur College of Science & Commerce  
Mumbai, Maharashtra, India

---

# 📄 Research Paper

The implementation supports the research paper:

**“Role of Digital Forensics in Banking and Financial Cybercrime Investigation”**

The research paper follows an IEEE-style structure and reports the methodology, experimental results, forensic reconstruction analysis, limitations, and conclusions derived from this implementation.

---

## ⭐ Acknowledgement

This project was developed as part of postgraduate academic research in Computer Science, focusing on the intersection of **machine learning, financial fraud detection, and digital forensic investigation**.