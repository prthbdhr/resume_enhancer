import pdfplumber
from docx import Document
import textract
from fastapi import UploadFile


def extract_text(file: UploadFile) -> str:
    if file.filename.endswith('.pdf'):
        with pdfplumber.open(file.file) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages)
    elif file.filename.endswith('.docx'):
        doc = Document(file.file)
        return "\n".join(para.text for para in doc.paragraphs)
    else:  # .txt
        return file.file.read().decode()