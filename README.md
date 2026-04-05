# Clinical NLP Pipeline

An end-to-end data pipeline that ingests raw clinical transcription notes,
processes them using NLP, classifies them by medical specialty using machine
learning, and surfaces insights through a REST API and Power BI dashboard.

---

## The Problem

Medical transcription notes are unstructured, messy, and scattered. Hospitals
generate thousands of clinical notes daily with no automated way to route them
to the correct department or extract structured insights from them.

This project builds a pipeline that solves that — taking raw doctor's notes
and automatically predicting which medical specialty they belong to.

---

## Pipeline Architecture

Raw CSV (MTSamples) → Python ETL → MySQL (raw_notes)
↓
NLP Transform Layer
- Text cleaning
- spaCy entity extraction
- TF-IDF vectorization
- Logistic Regression classifier
↓
MySQL (processed_notes)
↙                    ↘
FastAPI Endpoint          Power BI Dashboard
(live predictions)        (specialty insights)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data ingestion | Python, Pandas |
| Database | MySQL, SQLAlchemy |
| NLP | spaCy (en_core_web_sm) |
| Machine Learning | scikit-learn (TF-IDF + Logistic Regression) |
| API | FastAPI, Uvicorn |
| Dashboard | Power BI |
| Version Control | Git, GitHub |

---

## Dataset

**MTSamples** — 4,999 real medical transcription notes across 40 specialties.
Source: Kaggle (https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)

After cleaning: 4,966 usable records (33 dropped due to missing values).

---

## Project Structure

├── data/                    # Raw dataset (not tracked in Git)
├── models/                  # Saved model files (not tracked in Git)
├── src/
│   ├── db.py                # MySQL connection layer
│   ├── ingest.py            # Extract and load raw notes into MySQL
│   └── transform.py         # NLP pipeline, model training, processed notes
├── api.py                   # FastAPI prediction endpoint
├── .env                     # Database credentials (not tracked in Git)
├── .gitignore
├── requirements.txt



---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/bishowAd/clinical-nlp-pipeline.git
cd clinical-nlp-pipeline
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**4. Configure environment**
Create a `.env` file in the root directory:


DB_USER=root
DB_PASSWORD=your_password
DB_HOST=localhost
DB_NAME=clinical_nlp

**5. Run the pipeline**
```bash
# Step 1: Ingest raw data into MySQL
python src/ingest.py

# Step 2: Run NLP pipeline and train classifier
python src/transform.py

# Step 3: Start the API
python api.py
```

**6. Test the API**

Navigate to `http://localhost:8000/docs` and test the `/predict` endpoint:
```json
{
  "text": "Patient presents with chest pain radiating to the left arm,
  shortness of breath, elevated troponin and ST elevation on ECG
  suggesting acute myocardial infarction"
}
```

Expected response:
```json
{
  "predicted_specialty": "Cardiovascular / Pulmonary",
  "confidence": 0.1974
}
```

---

## Model Performance

- Dataset: 4,966 clinical notes across 40 specialties
- Train/test split: 80/20
- Overall accuracy: 26%
- Best performing specialties: Surgery (F1: 0.43), Cardiovascular (F1: 0.28)

**Why 26%?** The dataset is heavily imbalanced — Surgery has 1,000+ notes
while some specialties have fewer than 5. The model performs well on
high-volume specialties but struggles with rare ones. Future improvements
include SMOTE oversampling, merging rare classes, and experimenting with
BioBERT for domain-specific embeddings.

---

## What I Learned

- Building a production-style ETL pipeline from scratch
- Working with unstructured medical text using NLP
- Training and evaluating a multi-class text classifier
- Designing and deploying a REST API with FastAPI
- Connecting a live database to a Power BI dashboard
- The real challenge of class imbalance in medical datasets

---

## Future Improvements

- Add FDA Drug API as a second data source for multi-source ETL
- Apply SMOTE to handle class imbalance
- Replace Logistic Regression with BioBERT for medical NLP
- Apply for MIMIC-IV access for 10x more training data
- Deploy API to cloud (Azure or AWS)
- Add authentication to the FastAPI endpoint

---
