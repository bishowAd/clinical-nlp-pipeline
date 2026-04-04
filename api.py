from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import uvicorn

app = FastAPI(
    title="Clinical NLP API",
    description="Predicts medical specialty from clinical notes",
    version="1.0.0"
)

model = joblib.load("models/classifier.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

class NoteRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    predicted_specialty: str
    confidence: float

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = ' '.join(text.split())
    return text

@app.get("/")
def root():
    return {"message": "Clinical NLP API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: NoteRequest):
    cleaned = clean_text(request.text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    confidence = model.predict_proba(vectorized).max()
    return PredictionResponse(
        predicted_specialty=prediction,
        confidence=round(float(confidence), 4)
    )

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)