import streamlit as st
import requests

st.title("AI Resume Enhancer")
job_desc = st.text_area("Paste Job Description")
resume_text = st.text_area("Paste Resume Text")

if st.button("Optimize"):
    response = requests.post(
        "http://localhost:8000/analyze",
        json={"job_description": job_desc, "resume_text": resume_text},
    ).json()
    st.write("Match Score:", response["score"], "%")
    st.write("Missing Keywords:", ", ".join(response["missing_keywords"]))