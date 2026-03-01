# 🏥 VitalWatch — Early Sepsis Detection & Clinical Decision Support

> A multi-model deep learning system for early sepsis prediction in ICU patients, built on MIMIC-IV clinical data with a real-time clinical decision support interface.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

**VitalWatch** is an end-to-end clinical AI pipeline that predicts sepsis onset in ICU patients using temporal vital sign data. The system compares three progressively complex architectures — XGBoost, LSTM, and a custom Temporal Convolutional Network (TECO) — and wraps the best-performing model in a production-ready clinical interface with real-time SOFA scoring, SHAP-based explainability, and AI-generated clinical narratives.

### Key Results

| Model | AUROC | AUPRC | Architecture |
|-------|-------|-------|-------------|
| XGBoost | Baseline | Baseline | Gradient-boosted trees with engineered features |
| LSTM | Improved | Improved | Bidirectional LSTM with attention |
| **TECO** | **0.9206** | **Best** | **Temporal CNN + Multi-Head Attention** |

> TECO outperforms Epic's published sepsis model (AUROC 0.76) on in-distribution evaluation. Fair comparison requires prospective external validation.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    VitalWatch Pipeline                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  📂 Data Ingestion (CSV · JSON · HL7 v2 · FHIR R4)      │
│       ↓                                                  │
│  🔧 Universal Parser → Column Normalisation              │
│       ↓                                                  │
│  ⚕️  SOFA Score Computation (6 organ systems)             │
│       ↓                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐       │
│  │ XGBoost  │  │   LSTM   │  │ TECO (Primary)   │       │
│  │ (confirm)│  │ (compare)│  │ TCN + Attention   │       │
│  └────┬─────┘  └──────────┘  └────────┬─────────┘       │
│       ↓                               ↓                  │
│  SHAP Explainability          Risk Prediction             │
│       ↓                               ↓                  │
│  ┌─────────────────────────────────────────────┐         │
│  │         AI Clinical Narrative Agent          │         │
│  │         (Claude API Integration)             │         │
│  └─────────────────────────────────────────────┘         │
│       ↓                                                  │
│  🖥️  Streamlit Clinical Dashboard                        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
vitalwatch/
│
├── notebooks/                    # Research & development pipeline
│   ├── 01_eda.ipynb             # Exploratory data analysis (MIMIC-IV)
│   ├── 02_preprocessing.ipynb   # Data cleaning, cohort selection, labelling
│   ├── 03_xgboost.ipynb         # XGBoost baseline with feature engineering
│   ├── 04_lstm.ipynb            # LSTM sequence model
│   └── 05_teco_runpod.ipynb     # TECO training (RunPod GPU)
│
├── app/                          # Production application
│   ├── main.py                  # FastAPI backend — model inference & AI agent
│   ├── parser.py                # Universal clinical file parser
│   ├── streamlit_app.py         # Streamlit clinical dashboard
│   └── models/                  # Model weights directory (git-ignored)
│       └── README.md            # Instructions to obtain model files
│
├── results/                      # Model performance & evaluation
│   ├── teco_results.json        # TECO metrics (AUROC, AUPRC, etc.)
│   ├── teco_results.png         # Training curves & ROC/PR plots
│   ├── xgboost_results.json     # XGBoost metrics
│   └── lstm_results.json        # LSTM metrics
│
├── samples/                      # Sample patient files for demo
│   ├── sepsis_sample_P001_severe.csv
│   ├── sepsis_sample_P002_non_severe.csv
│   └── README.md                # Sample data documentation
│
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Features

### Multi-Format Clinical Data Ingestion
The parser accepts **CSV**, **JSON**, **HL7 v2**, and **FHIR R4** formats with automatic column normalisation via a 100+ alias map. Clinical data from any EHR system can be ingested without manual preprocessing.

### Real-Time SOFA Score Computation
When pre-computed SOFA scores are not present in the input data, VitalWatch derives all 6 organ system scores from raw clinical values using Sepsis-3 criteria:

| Component | Derived From | Fallback Chain |
|-----------|-------------|----------------|
| Respiratory | PaO2/FiO2 | → SpO2/FiO2 → SpO2 alone → Resp rate |
| Coagulation | Platelet count | Standard thresholds |
| Liver | Bilirubin | Standard thresholds |
| Cardiovascular | Vasopressor data | → MAP + Lactate heuristic |
| Neurological | GCS | Defaults to 0 if unavailable |
| Renal | Creatinine + Urine output | Takes the worse score |

### Dual-Model Inference with Confirmation
TECO serves as the primary predictor, while XGBoost independently confirms the risk assessment. Agreement between models increases clinical confidence.

### SHAP-Based Explainability
Every prediction is accompanied by the top 8 clinical drivers ranked by SHAP importance, showing which vitals or lab values are pushing the risk score up or down.

### AI Clinical Narrative
An integrated AI agent (Claude API) generates physician-style clinical summaries including urgency assessment, next steps, and monitoring recommendations — no AI jargon, written as a senior ICU physician would communicate.

### Clinical Dashboard
A Streamlit-based interface featuring:
- Animated risk gauge with colour-coded severity
- Haemodynamic, oxygenation, and SOFA component timelines
- SHAP feature importance visualisation
- Typewriter-effect clinical narrative
- Actionable next steps with urgency classification

---

## Getting Started

### Prerequisites
- Python 3.10+
- MIMIC-IV access (for training — [PhysioNet credentialed access](https://physionet.org/content/mimiciv/))
- Anthropic API key (for AI narrative feature)

### Installation

```bash
git clone https://github.com/vineek28/vitalwatch.git
cd vitalwatch
pip install -r requirements.txt
```

### Running the App

1. **Place model files** in `app/models/` (see `app/models/README.md` for details)

2. **Set environment variables:**
```bash
echo "ANTHROPIC_API_KEY=your_key_here" > app/.env
```

3. **Start the FastAPI backend:**
```bash
cd app
uvicorn main:app --reload --port 8000
```

4. **Start the Streamlit dashboard** (in a new terminal):
```bash
cd app
streamlit run streamlit_app.py
```

5. **Upload a sample patient file** from the `samples/` directory

---

## Data & Ethics

- **Data Source:** MIMIC-IV v2.2 (PhysioNet — credentialed access required)
- **Patient Privacy:** All data is de-identified per HIPAA Safe Harbor. No patient data is stored in this repository.
- **Model Limitations:** VitalWatch is trained and evaluated on MIMIC-IV (single-centre, retrospective). Prospective external validation is required before clinical deployment.
- **Not FDA-Cleared:** This is a research tool and clinical decision support prototype. It does not replace clinical judgment and is not approved for autonomous clinical decision-making.

---

## Sepsis-3 Label Definition

Patients were labelled using the internationally accepted **Sepsis-3** clinical criteria:
1. **Suspected infection** — blood culture ordered + antibiotics administered within 72 hours
2. **Organ dysfunction** — SOFA score increase ≥ 2 points from baseline

This dual-criteria approach ensures clinically meaningful labels aligned with current ICU practice guidelines.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Processing | PySpark, Pandas, NumPy |
| ML / DL | XGBoost, PyTorch (LSTM, TECO) |
| Explainability | SHAP |
| Backend | FastAPI |
| Frontend | Streamlit, Plotly |
| AI Agent | Anthropic Claude API |
| Training Infra | RunPod (GPU) |
| Data Source | MIMIC-IV (PhysioNet) |

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Vineeth Krishna S K**
- GitHub: [@vineek28](https://github.com/vineek28)
- Institution: VIT University

---

## Acknowledgements

- [MIMIC-IV](https://physionet.org/content/mimiciv/) — Johnson, A. et al. (PhysioNet)
- [Sepsis-3 Definition](https://jamanetwork.com/journals/jama/fullarticle/2492881) — Singer, M. et al. (JAMA 2016)
- [SHAP](https://github.com/shap/shap) — Lundberg, S.M. & Lee, S.I. (NeurIPS 2017)
