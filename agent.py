import json
from typing import Dict, Any, Optional
from langchain.tools import Tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.runnables import RunnableConfig
import logging
from datetime import datetime
from fastapi import HTTPException
from pydantic import BaseModel
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------
# API Response Models
# -----------------------------
class ChatResponse(BaseModel):
    response: str
    session_id: str
    processed_at: str


class AnalysisResponse(BaseModel):
    overall_score: float
    match_percentage: float
    matching_keywords: list[str]
    missing_keywords: list[str]
    strengths: list[str]
    weaknesses: list[str]
    section_analysis: dict
    recommendations: dict
    processed_at: str
    processing_time_ms: float
    ats_scan: Optional[dict] = None


# -----------------------------
# Comprehensive Resume Chatbot
# -----------------------------
class ResumeChatbot:
    def __init__(self, google_api_key: str, resume_matcher):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=0.2,
            convert_system_message_to_human=True
        )
        self.resume_matcher = resume_matcher

        # Create resume analysis tool
        self.resume_tool = Tool(
            name="resume_analyzer",
            description="""Analyze and score a resume against a job description. Also provides general resume improvement suggestions when no job description is provided. Input should be a JSON string with 'resume' and optional 'job_description' keys.""",
            func=self._analyze_resume_wrapper
        )

        self.system_prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are a comprehensive career assistant chatbot. You can:
1. Analyze resumes against job descriptions with detailed scoring and recommendations
2. Provide general resume improvement advice when no job description is provided
3. Have casual conversations about careers, job searching, and professional development

IMPORTANT INSTRUCTIONS:
- When a user provides both a resume AND a job description, automatically use the resume_analyzer tool
- When a user provides only a resume, use the resume_analyzer tool for general improvement suggestions
- Look for keywords like 'evaluate', 'score', 'analyze', 'compare', 'improve', 'review' along with resume content
- If you detect resume content (experience, skills, education), offer to analyze or improve it
- After using the tool, provide a friendly interpretation of the results
- For casual chat without resume content, respond naturally without using tools
- Be proactive in offering resume analysis when appropriate

The resume_analyzer tool expects JSON input with 'resume' and optional 'job_description'."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        self.agent = create_tool_calling_agent(
            llm=self.llm,
            tools=[self.resume_tool],
            prompt=self.system_prompt
        )

        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=[self.resume_tool],
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )

        self.memory = {}
        self.chat_with_history = RunnableWithMessageHistory(
            self.agent_executor,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )

    def _get_session_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self.memory:
            self.memory[session_id] = ChatMessageHistory()
        return self.memory[session_id]

    def _analyze_resume_wrapper(self, input_str: str) -> str:
        try:
            input_data = json.loads(input_str)
            resume = input_data.get("resume", "")
            job_description = input_data.get("job_description", "")

            if not resume:
                return "Error: Resume content is required."

            # Analyze with job description if provided
            if job_description:
                analysis = self.resume_matcher.analyze_resume_vs_jd(resume, job_description)
                recommendations = self.resume_matcher.generate_recommendations(analysis, resume)

                result = {
                    "analysis_type": "jd_comparison",
                    "overall_score": analysis.get("overall_match_score", 5.0),
                    "match_percentage": analysis.get("overall_match_score", 5.0) * 10,
                    "matching_keywords": analysis.get("matching_keywords", []),
                    "missing_keywords": analysis.get("missing_keywords", []),
                    "strengths": analysis.get("strengths", []),
                    "weaknesses": analysis.get("weaknesses", []),
                    "section_analysis": analysis.get("section_analysis", {}),
                    "recommendations": recommendations,
                    "processed_at": datetime.isoformat(datetime.now())
                }
            else:
                # General resume analysis without JD
                result = self._general_resume_analysis(resume)

            return json.dumps(result, indent=2)

        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON input. {str(e)}"
        except Exception as e:
            return f"Error analyzing resume: {str(e)}"

    def _general_resume_analysis(self, resume_text: str) -> Dict[str, Any]:
        """Provide general resume improvement suggestions without JD"""
        prompt = f"""
        Analyze this resume and provide general improvement suggestions. Focus on:
        1. Overall structure and formatting
        2. Key strengths and areas for improvement
        3. Actionable suggestions for each section
        4. ATS optimization tips

        Provide output in JSON format with:
        - overall_assessment: brief summary
        - strengths: list of strong points
        - areas_for_improvement: list of suggestions
        - section_suggestions: dict with advice for each section
        - ats_tips: list of ATS optimization tips

        Resume:
        {resume_text[:8000]}
        """

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            return robust_json_parse(content)
        except Exception as e:
            # Fallback to basic analysis
            return {
                "analysis_type": "general_improvement",
                "overall_assessment": "Resume analysis completed",
                "strengths": ["Good content structure", "Relevant experience listed"],
                "areas_for_improvement": ["Add more quantifiable achievements", "Improve keyword optimization"],
                "section_suggestions": {
                    "experience": "Use more action verbs and metrics",
                    "skills": "Categorize technical and soft skills",
                    "education": "Include relevant coursework if recent graduate"
                },
                "ats_tips": [
                    "Use standard section headings (Experience, Education, Skills)",
                    "Include relevant keywords from target job descriptions",
                    "Avoid graphics and complex formatting"
                ]
            }

    def chat(self, message: str, session_id: str = "default") -> str:
        """Main chat interface"""
        try:
            response = self.chat_with_history.invoke(
                {"input": message},
                config=RunnableConfig(configurable={"session_id": session_id})
            )
            return response["output"]
        except Exception as e:
            logger.error(f"Chat error: {str(e)}", exc_info=True)
            return f"Sorry, I encountered an error: {str(e)}"

    def analyze_resume_directly(self, resume_markdown: str, job_description: str, session_id: str = "default") -> str:
        """Direct analysis without conversation"""
        tool_input = json.dumps({
            "resume": resume_markdown,
            "job_description": job_description
        })
        message = f"Please analyze this resume for the given job:\n\nRESUME:\n{resume_markdown}\n\nJOB DESCRIPTION:\n{job_description}"
        return self.chat(message, session_id)


# -----------------------------
# Utility Functions
# -----------------------------
def robust_json_parse(content: str):
    """
    Robustly parse JSON from LLM output, handling common formatting issues.
    """
    # Try to extract JSON block
    if '```json' in content:
        content = content.split('```json')[1].split('```')[0]
    elif '```' in content:
        content = content.split('```')[1].split('```')[0]

    # Fallback: extract first {...} block
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        content = match.group(0)

    # Try normal parsing
    try:
        return json.loads(content)
    except Exception:
        # Attempt to fix common issues (e.g., trailing commas)
        fixed = re.sub(r',\s*([}\]])', r'\1', content)
        try:
            return json.loads(fixed)
        except Exception as e:
            return {"error": f"Failed to parse JSON: {str(e)}", "raw": content}