# PaySim Forensic Investigation & Cybercrime Analysis

A comprehensive machine learning, transaction reconstruction, and digital forensic investigation platform for financial cybercrime analysis on mobile money networks using the PaySim synthetic dataset.

---

## 📌 Project Overview

This repository contains an end-to-end research framework and web application designed to detect, reconstruct, and evaluate fraudulent financial transaction networks. The project bridges traditional machine learning fraud detection with forensic case reconstruction to assist cybercrime investigators.

### Key Features
* **10-Step Automated Forensic Pipeline**: From raw dataset ingestion and feature engineering to model training, threshold search, case reconstruction, evaluation, and window sensitivity analysis.
* **Audited ML Models**: Logistic Regression and Random Forest models tuned with decision threshold optimization for high-precision fraud identification.
* **Forensic Case Reconstruction**: Algorithmic grouping of fraud seeds, transaction chains, and supporting evidence.
* **FastAPI Backend**: REST API delivering case data, model performance metrics, evidence chains, and research summaries.
* **React + Vite Forensic Dashboard**: Modern interactive UI with dark mode, real-time filtering, risk analysis, and visual case investigation tools.

---

## 📁 Repository Structure

```
PaySim_Forensic_Investigation/
├── src/                          # Forensic Analysis & ML Pipeline Scripts
│   ├── 01_dataset_inspection.py  # Data validation & summary stats
│   ├── 02_preprocess.py          # Data cleaning & encoding
│   ├── 03_feature_engineering.py # Financial network & velocity features
│   ├── 04_train_models.py        # Model training & auditing
│   ├── 05_generate_predictions.py# Prediction generation
│   ├── 06_reconstruct_cases.py   # Transaction network case reconstruction
│   ├── 07_evaluate_cases.py      # Case-level precision & recall evaluation
│   ├── 08_window_sensitivity.py # Sensitivity analysis on time windows
│   ├── 09_evaluation.py          # Final forensic metric evaluation
│   └── 10_research_results.py    # Summary report generator
├── backend/                      # FastAPI Backend Service
│   ├── main.py                   # REST API routes and data loaders
│   └── requirements.txt          # Python dependencies
├── frontend/                     # React + Vite Frontend Application
│   ├── src/                      # Components, pages, and dynamic styling
│   ├── package.json              # Node dependencies
│   └── vite.config.js            # Vite build configuration
├── outputs/                      # Generated Reports & Audited Models
│   ├── cases/                    # Reconstructed case summaries and evidence
│   ├── models/                   # Serialized ML models (.joblib)
│   └── reports/                  # JSON & CSV performance reports
└── README.md                     # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
* **Python 3.10+**
* **Node.js 18+** & **npm**

---

### 1. Data Setup

Download the PaySim synthetic financial dataset from Kaggle (`PS_20174392719_1491204439457_log.csv`) and place it in the `data/raw/` directory:
```
data/raw/PS_20174392719_1491204439457_log.csv
```

### 2. Backend Setup & Pipeline Execution

```bash
# Navigate to backend directory
cd backend

# Create & activate a virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server
uvicorn main:app --reload --port 8000
```
The API documentation will be available at [http://localhost:8000/docs](http://localhost:8000/docs).

### 3. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```
The investigation dashboard will open at [http://localhost:5173](http://localhost:5173).

---

## 🧪 Forensic Pipeline Steps

To re-run the complete pipeline from scratch:
```bash
python src/01_dataset_inspection.py
python src/02_preprocess.py
python src/03_feature_engineering.py
python src/04_train_models.py
python src/05_generate_predictions.py
python src/06_reconstruct_cases.py
python src/07_evaluate_cases.py
python src/08_window_sensitivity.py
python src/09_evaluation.py
python src/10_research_results.py
```

---

## 📜 License & Citation

Developed for M.Sc. Computer Science / Forensic Research on Financial Cybercrime Investigation.
Dataset Source: PaySim Synthetic Financial Datasets (Kaggle).
