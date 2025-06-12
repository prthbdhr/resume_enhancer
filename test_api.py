import requests
import os


def test_pdf():
    if not os.path.exists('test.pdf'):
        print("Error: test.pdf not found")
        return

    with open('test.pdf', 'rb') as f:
        response = requests.post(
            "http://localhost:8000/analyze-pdf",
            files={"resume_pdf": ("test.pdf", f, "application/pdf")},
            data={"job_description": "Python developer"}
        )
        print("PDF Test Results:", response.json())


def test_docx():
    try:
        with open('test.docx', 'rb') as f:
            response = requests.post(
                "http://localhost:8000/analyze-pdf",
                files={"resume_pdf": ("test.docx", f,
                                      "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
                data={"job_description": "Python developer"}
            )
            print("DOCX Test Results:", response.json())
    except FileNotFoundError:
        print("Note: test.docx not found - skipping DOCX test")


# Run tests
test_pdf()
test_docx()