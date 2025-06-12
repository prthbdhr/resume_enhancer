# main.py
from pydantic import BaseModel

from utils.keyword_extractor import extract_keywords_tfidf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import spacy
from nltk.corpus import stopwords
import nltk
from typing import List
from fastapi import UploadFile
from fastapi import FastAPI, UploadFile, File, Form
import pdfplumber
from fastapi.responses import HTMLResponse  # Add this import
from utils.file_parser import extract_text

app = FastAPI()

# Allow CORS (for frontend connection)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (adjust for production)
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load NLP model and stopwords
nlp = spacy.load("en_core_web_sm")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Define a request model
class AnalysisRequest(BaseModel):
    job_description: str
    resume_text: str


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <h1>AI Resume Enhancer API</h1>
    <p>Endpoints:</p>
    <ul>
        <li><b>/analyze</b> (POST): Analyze text resume</li>
        <li><b>/analyze-pdf</b> (POST): Analyze PDF resume</li>
    </ul>
    <p>Go to <a href="/docs">/docs</a> for Swagger UI.</p>
    """


@app.post("/analyze")
async def analyze_resume(job_description: str, resume_text: str):  # Changed to use Pydantic model
    try:
        job_keywords = extract_keywords_tfidf(job_description)  # Note: now using request.job_description
        resume_keywords = extract_keywords_tfidf(resume_text)

        match_score = calculate_match(job_keywords, resume_keywords)

        return {
            "score": match_score,
            "missing_keywords": list(set(job_keywords) - set(resume_keywords)),
            "job_keywords": job_keywords,
            "resume_keywords": resume_keywords,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def extract_keywords(text: str) -> List[str]:
    """Extract keywords using spaCy and NLTK."""
    doc = nlp(text.lower())
    keywords = [
        token.text for token in doc
        if not token.is_stop
        and not token.is_punct
        and token.is_alpha
    ]
    return list(set(keywords))  # Remove duplicates

def calculate_match(job_keywords: List[str], resume_keywords: List[str]) -> float:
    """Calculate match percentage between job and resume keywords."""
    if not job_keywords:
        return 0.0
    matched = sum(1 for word in job_keywords if word in resume_keywords)
    return round((matched / len(job_keywords)) * 100, 2)


@app.post("/analyze-pdf")
async def analyze_pdf(
        job_description: str = Form(...),  # Get from form data
        resume_pdf: UploadFile = File(...)  # Get uploaded file
):
    try:
        # Extract text from PDF
        resume_text = extract_text(resume_pdf)

        # Call your existing analyze_resume function
        return await analyze_resume(job_description, resume_text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Run locally: uvicorn main:app --reload

