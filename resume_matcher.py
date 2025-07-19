import json
import re
from fastapi import HTTPException
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import PyPDF2
import docx
import io
from pathlib import Path


class ResumeKeywordMatcher:
    def __init__(self, gemini_api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=gemini_api_key,
            temperature=0.3
        )

    def extract_text_from_file(self, file_content: bytes, filename: str) -> str:
        """Extract text from supported file types with error handling"""
        file_extension = Path(filename).suffix.lower()

        try:
            if file_extension == '.pdf':
                return self._extract_from_pdf(file_content)
            elif file_extension in ['.docx', '.doc']:
                return self._extract_from_docx(file_content)
            elif file_extension == '.txt':
                return file_content.decode('utf-8')
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Text extraction failed: {str(e)}"
            )

    def _extract_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF with section detection"""
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""

        for page in pdf_reader.pages:
            page_text = page.extract_text()
            text += page_text + "\n\n"

        # Normalize section headers
        sections = ["experience", "education", "skills", "projects", "summary"]
        for section in sections:
            text = re.sub(
                rf'\b{section}\b:?',
                f"\n\nSECTION:{section.upper()}\n",
                text,
                flags=re.IGNORECASE
            )
        return text.strip()

    def _extract_from_docx(self, file_content: bytes) -> str:
        """Extract text from DOCX with formatting awareness"""
        doc = docx.Document(io.BytesIO(file_content))
        structured_text = []

        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue

            if any(run.bold for run in paragraph.runs) or paragraph.style.name.startswith('Heading'):
                structured_text.append(f"\n\nSECTION:{text.upper()}\n")
            else:
                structured_text.append(text + " ")

        return "".join(structured_text).strip()

    def analyze_resume_vs_jd(
            self,
            resume_text: str,
            jd_text: str
    ) -> Dict[str, Any]:
        """
        Analyze resume against job description in a single LLM call

        Returns:
            {
                "overall_match_score": float (0-10),
                "section_analysis": {
                    "section_name": {
                        "match_score": float (0-10),
                        "matching_keywords": [],
                        "missing_keywords": []
                    }
                },
                "missing_keywords": [],
                "strengths": [],
                "weaknesses": []
            }
        """
        prompt = f"""
        Analyze the following resume against the job description. Provide:
        1. Overall matching score (0-10)
        2. Section-by-section analysis
        3. Missing keywords from JD
        4. Strengths and weaknesses

        Instructions:
        - Compare content, not exact wording (consider synonyms)
        - Score based on relevance to JD requirements
        - For each resume section:
            * Calculate match score (0-10)
            * List matching keywords
            * List missing JD keywords relevant to section
        - Identify overall missing keywords from JD
        - Highlight 3 key strengths and 3 weaknesses

        Output JSON format:
        {{
            "overall_match_score": 7.5,
            "section_analysis": {{
                "WORK EXPERIENCE": {{
                    "match_score": 8.0,
                    "matching_keywords": ["Python", "Project Management"],
                    "missing_keywords": ["Agile Methodology"]
                }},
                "EDUCATION": {{...}},
                ...
            }},
            "missing_keywords": ["AWS Certification", "Kubernetes"],
            "strengths": ["Strong Python experience", ...],
            "weaknesses": ["Lacking cloud experience", ...]
        }}

        --- JOB DESCRIPTION ---
        {jd_text[:10000]}

        --- RESUME ---
        {resume_text[:10000]}
        """

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()

            # Extract JSON from response
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]

            return json.loads(content)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Analysis failed: {str(e)}"
            )

    def generate_recommendations(
            self,
            analysis: Dict[str, Any],
            resume_text: str
    ) -> Dict[str, Any]:
        """
        Generate actionable recommendations based on analysis

        Returns:
            {
                "priority_actions": [],
                "section_improvements": {
                    "section_name": []
                },
                "ats_optimization": [],
                "strengths_to_highlight": [],
                "weaknesses_to_address": []
            }
        """
        recommendations = {
            "priority_actions": [],
            "section_improvements": {},
            "ats_optimization": self._get_ats_tips(resume_text),
            "strengths_to_highlight": [],
            "weaknesses_to_address": []
        }

        # Add strengths and weaknesses if available
        if analysis.get("strengths"):
            recommendations["strengths_to_highlight"] = analysis["strengths"][:3]
        if analysis.get("weaknesses"):
            recommendations["weaknesses_to_address"] = analysis["weaknesses"][:3]

        # Priority actions based on missing keywords
        if analysis.get("missing_keywords"):
            top_missing = analysis["missing_keywords"][:3]
            recommendations["priority_actions"].append(
                f"Add these missing keywords: {', '.join(top_missing)}"
            )

        # Section-specific improvements
        for section, data in analysis.get("section_analysis", {}).items():
            section_recs = []

            # Add missing keywords for the section
            if data.get("missing_keywords"):
                section_recs.extend(
                    f"Add keyword: {kw}" for kw in data["missing_keywords"][:2]
                )

            # Add score-based recommendations
            if data.get("match_score", 0) < 7:
                if "experience" in section.lower():
                    section_recs.append("Quantify achievements with metrics")
                elif "skill" in section.lower():
                    section_recs.append("List technical skills more prominently")
                else:
                    section_recs.append("Expand this section with more details")

            if section_recs:
                recommendations["section_improvements"][section] = section_recs

        # Additional general recommendations based on overall score
        overall_score = analysis.get("overall_match_score", 0)
        if overall_score < 6:
            recommendations["priority_actions"].append(
                "Significantly tailor resume to better match job requirements"
            )
        elif overall_score < 8:
            recommendations["priority_actions"].append(
                "Moderately improve alignment with job requirements"
            )

        return recommendations
    def _get_ats_tips(self, resume_text: str) -> list[str]:
        """Generate ATS optimization tips using LLM"""
        prompt = f"""
        Provide 3 concise ATS optimization tips for this resume:
        - Focus on keyword placement
        - Formatting improvements
        - Content structure

        Return as JSON array: ["tip1", "tip2", "tip3"]

        Resume:
        {resume_text[:5000]}
        """

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return json.loads(response.content.strip())
        except:
            return [
                "Use standard section headings",
                "Include relevant keywords from job description",
                "Avoid graphics and complex formatting"
            ]


# Usage Example:
if __name__ == "__main__":
    analyzer = ResumeKeywordMatcher("your-api-key")

    # Load files
    with open("resume.pdf", "rb") as f:
        resume = analyzer.extract_text_from_file(f.read(), "resume.pdf")

    with open("jd.txt", "rb") as f:
        jd = analyzer.extract_text_from_file(f.read(), "jd.txt")

    # Analyze
    analysis = analyzer.analyze_resume_vs_jd(resume, jd)
    recommendations = analyzer.generate_recommendations(analysis, resume)

    print("Match Score:", analysis["overall_match_score"])
    print("Recommendations:", recommendations)