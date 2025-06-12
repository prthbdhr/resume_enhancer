import pdfplumber

try:
    from docx import Document
except ImportError:
    Document = None  # Fallback if docx not available

def extract_text(file):
    if file.filename.lower().endswith('.pdf'):
        with pdfplumber.open(file.file) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages)
    elif Document and file.filename.lower().endswith('.docx'):
        doc = Document(file.file)
        return "\n".join(para.text for para in doc.paragraphs)
    else:
        file.file.seek(0)
        return file.file.read().decode('utf-8', errors='ignore')