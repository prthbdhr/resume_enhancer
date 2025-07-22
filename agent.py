import json
from typing import Dict, Any, Optional
from langchain.tools import Tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.runnables import RunnableConfig
import logging
from datetime import datetime
from fastapi import HTTPException
from pydantic import BaseModel
from resume_matcher import ResumeKeywordMatcher
from langchain_openai import ChatOpenAI
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
# Resume Scorer
# -----------------------------
class ResumeScorer:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key cannot be empty")
        self.matcher = ResumeKeywordMatcher(api_key)

    def analyze_resume(self, resume_text: str, job_description: str, file_meta: Optional[Dict[str, Any]] = None, detailed_analysis: Optional[Dict[str, Any]] = None) -> AnalysisResponse:
        start_time = datetime.now()
        try:
            analysis = self.matcher.analyze_resume_vs_jd(resume_text, job_description)
            recommendations = self.matcher.generate_recommendations(analysis, resume_text)
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            return AnalysisResponse(
                overall_score=analysis["overall_match_score"],
                match_percentage=analysis["overall_match_score"] * 10,
                matching_keywords=analysis.get("matching_keywords", []),
                missing_keywords=analysis.get("missing_keywords", []),
                strengths=analysis.get("strengths", []),
                weaknesses=analysis.get("weaknesses", []),
                section_analysis=analysis.get("section_analysis", {}),
                recommendations=recommendations,
                processed_at=datetime.isoformat(datetime.now()),
                processing_time_ms=round(processing_time, 2),
            )
        except Exception as e:
            logger.error(f"Error analyzing resume: {str(e)}", exc_info=True)
            raise RuntimeError(f"Resume analysis failed: {str(e)}") from e

# -----------------------------
# Resume Analyzer Chatbot
# -----------------------------
class ResumeAnalyzerChatbot:
    def __init__(self, google_api_key: str, existing_resume_tool_function):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=0.1,
            convert_system_message_to_human=True
        )
        self.free_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
        self.current_model = "gemini"
        self.call_count = 0
        self.resume_tool_function = existing_resume_tool_function
        self.resume_tool = Tool(
            name="resume_scorer",
            description="""Analyze and score a resume against a job description. Input should be a JSON string with 'resume' and 'job_description' keys.""",
            func=self._score_resume_wrapper
        )
        self.system_prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are a helpful career assistant chatbot. You can:
1. Have casual conversations about careers, job searching, and professional development
2. Automatically analyze resumes when provided with both a resume and job description
IMPORTANT INSTRUCTIONS:
- When a user provides both a resume (in any format) AND a job description, automatically use the resume_scorer tool
- Look for keywords like 'evaluate', 'score', 'analyze', 'compare' along with resume/job content
- If you detect resume content (experience, skills, education) AND job requirements, call the tool
- After using the tool, provide a friendly interpretation of the results
- For casual chat, respond naturally without using tools
- Be proactive in offering resume analysis when appropriate
Remember: The resume_scorer tool expects JSON input with 'resume' and 'job_description' keys."""),
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
    def _score_resume_wrapper(self, input_str: str) -> str:
        try:
            input_data = json.loads(input_str)
            resume = input_data.get("resume", "")
            job_description = input_data.get("job_description", "")
            if not resume or not job_description:
                return "Error: Both resume and job_description are required."
            result = self.resume_tool_function(resume, job_description)
            return json.dumps(result, indent=2)
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON input. {str(e)}"
        except Exception as e:
            return f"Error analyzing resume: {str(e)}"
    def chat(self, message: str, session_id: str = "default") -> str:
        alert = ""
        self.call_count += 1
        if self.call_count == 20 or (self.call_count > 20 and self.call_count % 10 == 0 and self.call_count <= 50):
            alert = f"\n[ALERT] You have made {self.call_count} LLM calls. Please be aware of usage limits."
        if self.call_count == 51:
            self._switch_to_free_model()
            alert += "\n[NOTICE] You have reached 50 calls. Switching to the free model (OpenAI GPT-3.5 Turbo)."
        try:
            response = self.chat_with_history.invoke(
                {"input": message},
                config=RunnableConfig(configurable={"session_id": session_id})
            )
            return response["output"] + alert
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}" + alert
    def _switch_to_free_model(self):
        self.llm = self.free_llm
        self.current_model = "free"
    def analyze_resume_directly(self, resume_markdown: str, job_description: str, session_id: str = "default") -> str:
        tool_input = json.dumps({
            "resume": resume_markdown,
            "job_description": job_description
        })
        message = f"Please analyze this resume for the given job:\n\nRESUME:\n{resume_markdown}\n\nJOB DESCRIPTION:\n{job_description}"
        return self.chat(message, session_id)

# -----------------------------
# Resume Enhancer Chatbot
# -----------------------------
class ResumeEnhancerChatbot:
    def __init__(self, google_api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=0.2,
            convert_system_message_to_human=True
        )
        self.free_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
        self.current_model = "gemini"
        self.call_count = 0
        self.memory = {}
        self.system_prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are a helpful AI resume coach. Your job is to help users iteratively improve their resumes. 
- Give actionable, section-by-section feedback.
- Suggest improvements, best practices, and highlight weaknesses.
- Ask clarifying questions if needed.
- Continue the conversation until the user is satisfied.
- Do NOT require a job description.
- If the user pastes their resume, analyze and suggest improvements.
- If the user asks for help with a section, focus on that section.
- Be friendly, supportive, and specific.
"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        self.agent = create_tool_calling_agent(
            llm=self.llm,
            tools=[],
            prompt=self.system_prompt
        )
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=[],
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )
        self.chat_with_history = RunnableWithMessageHistory(
            self.agent_executor,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )
    def _switch_to_free_model(self):
        self.llm = self.free_llm
        self.current_model = "free"
    def _get_session_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self.memory:
            self.memory[session_id] = ChatMessageHistory()
        return self.memory[session_id]
    def chat(self, message: str, session_id: str = "default") -> str:
        alert = ""
        self.call_count += 1
        if self.call_count == 20 or (self.call_count > 20 and self.call_count % 10 == 0 and self.call_count <= 50):
            alert = f"\n[ALERT] You have made {self.call_count} LLM calls. Please be aware of usage limits."
        if self.call_count == 51:
            self._switch_to_free_model()
            alert += "\n[NOTICE] You have reached 50 calls. Switching to the free model (OpenAI GPT-3.5 Turbo)."
        try:
            response = self.chat_with_history.invoke(
                {"input": message},
                config=RunnableConfig(configurable={"session_id": session_id})
            )
            return response["output"] + alert
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}" + alert

# -----------------------------
# Resume Extraction Agent
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
    import json
    try:
        return json.loads(content)
    except Exception:
        # Attempt to fix common issues (e.g., trailing commas)
        fixed = re.sub(r',\s*([}\]])', r'\1', content)
        try:
            return json.loads(fixed)
        except Exception as e:
            return {"error": f"Failed to parse JSON: {str(e)}", "raw": content}

class ResumeExtractionAgent:
    """
    Agent that sends raw resume file and job description to the LLM for extraction and (optional) analysis.
    """
    def __init__(self, google_api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=0.2,
            convert_system_message_to_human=True
        )
    def extract_and_analyze(self, file_content: bytes, filename: str, job_description: str) -> dict:
        prompt = f"""
You are an expert at reading resumes in any file format. Extract all readable text from the following file. If the file is a resume, return the text in markdown format under the key 'extracted_text'.

Then, analyze the resume for the following job description and return a JSON object with:
- extracted_text: the resume text in markdown
- analysis: (optional) a brief summary of how well the resume matches the job description

--- JOB DESCRIPTION ---
{job_description[:10000]}

--- FILE (base64, {filename}) ---
{file_content[:10000]}
"""
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            return robust_json_parse(content)
        except Exception as e:
            return {"error": f"Extraction/analysis failed: {str(e)}"}