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
# agent.py modifications (add these at the top)
from fastapi import HTTPException
from pydantic import BaseModel
from resume_matcher import ResumeKeywordMatcher

# Initialize matcher as None at module level
matcher: Optional[ResumeKeywordMatcher] = None
# Add this model class for API responses
class ChatResponse(BaseModel):
    response: str
    session_id: str
    processed_at: str


from datetime import datetime
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, validator
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalysisResponse(BaseModel):
    overall_score: float = Field(..., ge=0, le=10, description="Overall match score (0-10)")
    match_percentage: float = Field(..., ge=0, le=100, description="Match percentage")
    matching_keywords: List[str] = Field(..., description="Keywords matching between resume and JD")
    missing_keywords: List[str] = Field(..., description="Keywords in JD missing from resume")
    strengths: List[str] = Field(..., description="Key strengths identified")
    weaknesses: List[str] = Field(..., description="Key weaknesses identified")
    section_analysis: Dict[str, Any] = Field(..., description="Detailed section analysis")
    recommendations: Dict[str, Any] = Field(..., description="Actionable improvement suggestions")
    processed_at: str = Field(..., description="ISO timestamp of analysis completion")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    ats_scan: Optional[Dict[str, Any]] = Field(
        None,
        description="ATS optimization analysis if requested"
    )


class ResumeScorer:
    def __init__(self, api_key: str):
        """Initialize the scorer with required dependencies"""
        if not api_key:
            raise ValueError("API key cannot be empty")
        self.matcher = ResumeKeywordMatcher(api_key)

    def analyze_resume(
            self,
            resume_text: str,
            job_description: str,
            file_meta: Optional[Dict[str, Any]] = None,
            detailed_analysis: Optional[Dict[str, Any]] = None
    ) -> AnalysisResponse:
        """
        Analyze a resume against a job description

        Args:
            resume_text: Markdown or text content of the resume
            job_description: Text of the job description
            file_meta: Optional metadata about the resume file
            detailed_analysis: Optional additional analysis data

        Returns:
            AnalysisResponse containing all scoring results
        """
        start_time = datetime.now()

        try:
            # Extract keywords from both documents

            # Calculate matching scores
            analysis = self.matcher.analyze_resume_vs_jd(resume_text, job_description)
            recommendations = self.matcher.generate_recommendations(analysis, resume_text)

            # Optional ATS scan

            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            # Prepare response
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




class ResumeAnalyzerChatbot:
    def __init__(self, google_api_key: str, existing_resume_tool_function):
        """
        Initialize the chatbot with Gemini Flash and the resume scoring tool.

        Args:
            google_api_key: Your Google API key for Gemini
            existing_resume_tool_function: Your already implemented function that takes
                                         (resume_markdown: str, job_description: str) -> dict
        """
        # Initialize Gemini Flash model
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=0.1,
            convert_system_message_to_human=True  # Gemini requirement
        )

        # Store the existing resume tool function
        self.resume_tool_function = existing_resume_tool_function

        # Create the resume scoring tool
        self.resume_tool = Tool(
            name="resume_scorer",
            description="""Analyze and score a resume against a job description. 
            Use this tool when the user provides both a resume and job description, 
            or asks to evaluate/score/analyze a resume for a specific job.
            Input should be a JSON string with 'resume' and 'job_description' keys.""",
            func=self._score_resume_wrapper
        )

        # Create system prompt
        self.system_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful career assistant chatbot. You can:

1. Have casual conversations about careers, job searching, and professional development
2. Automatically analyze resumes when provided with both a resume and job description

IMPORTANT INSTRUCTIONS:
- When a user provides both a resume (in any format) AND a job description, automatically use the resume_scorer tool
- Look for keywords like "evaluate", "score", "analyze", "compare" along with resume/job content
- If you detect resume content (experience, skills, education) AND job requirements, call the tool
- After using the tool, provide a friendly interpretation of the results
- For casual chat, respond naturally without using tools
- Be proactive in offering resume analysis when appropriate

Remember: The resume_scorer tool expects JSON input with 'resume' and 'job_description' keys."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        # Create agent with tools
        self.agent = create_tool_calling_agent(
            llm=self.llm,
            tools=[self.resume_tool],
            prompt=self.system_prompt
        )

        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=[self.resume_tool],
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )

        # Set up conversation memory
        self.memory = {}

        # Create the runnable with message history
        self.chat_with_history = RunnableWithMessageHistory(
            self.agent_executor,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )

    def _get_session_history(self, session_id: str) -> ChatMessageHistory:
        """Get or create chat history for a session."""
        if session_id not in self.memory:
            self.memory[session_id] = ChatMessageHistory()
        return self.memory[session_id]

    def _score_resume_wrapper(self, input_str: str) -> str:
        """Wrapper for the resume scoring tool to handle JSON input/output."""
        try:
            # Parse JSON input
            input_data = json.loads(input_str)
            resume = input_data.get("resume", "")
            job_description = input_data.get("job_description", "")

            if not resume or not job_description:
                return "Error: Both resume and job_description are required."

            # Call your existing function
            result = self.resume_tool_function(resume, job_description)

            # Return JSON string
            return json.dumps(result, indent=2)

        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON input. {str(e)}"
        except Exception as e:
            return f"Error analyzing resume: {str(e)}"

    def chat(self, message: str, session_id: str = "default") -> str:
        """
        Main chat interface.

        Args:
            message: User's message
            session_id: Session identifier for conversation history

        Returns:
            Bot's response
        """
        try:
            response = self.chat_with_history.invoke(
                {"input": message},
                config=RunnableConfig(configurable={"session_id": session_id})
            )
            print(response)
            return response["output"]
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"

    def analyze_resume_directly(self, resume_markdown: str, job_description: str,
                                session_id: str = "default") -> str:
        """
        Direct method to analyze resume (bypasses automatic detection).

        Args:
            resume_markdown: Resume in markdown format
            job_description: Job description text
            session_id: Session identifier

        Returns:
            Analysis result and interpretation
        """
        tool_input = json.dumps({
            "resume": resume_markdown,
            "job_description": job_description
        })

        message = f"Please analyze this resume for the given job:\n\nRESUME:\n{resume_markdown}\n\nJOB DESCRIPTION:\n{job_description}"
        return self.chat(message, session_id)


# Example usage and testing
def example_resume_scorer(resume_markdown: str, job_description: str) -> Dict[str, Any]:
    """
    Example implementation of your resume scoring function.

    Args:
        resume_markdown: Resume content in markdown format
        job_description: Job description text

    Returns:
        Dictionary containing analysis results

    Raises:
        RuntimeError: If analysis fails
    """
    try:
        # In production, get API key from environment variables
        api_key = "AIzaSyD7FPBzvTL7wokvxpVQM4lwiQ9slUmAY_M"  # Remove hardcoded key in production

        # Initialize scorer
        scorer = ResumeScorer(api_key)

        # Perform analysis (without file metadata or detailed analysis)
        result = scorer.analyze_resume(resume_markdown, job_description)

        # Return as dictionary (could also return the AnalysisResponse directly)
        return result.dict()

    except Exception as e:
        logger.error(f"Error in resume scoring: {str(e)}")
        raise RuntimeError(f"Failed to score resume: {str(e)}") from e



# Initialize and run the chatbot
def main():
    # Replace with your actual Google API key
    GOOGLE_API_KEY = "AIzaSyASZydNF55Bct2-3PQknOBjNuXpWqNSiic"

    # Initialize chatbot with your existing resume tool function
    chatbot = ResumeAnalyzerChatbot(
        google_api_key=GOOGLE_API_KEY,
        existing_resume_tool_function=example_resume_scorer  # Replace with your function
    )

    print("Resume Analysis Chatbot Started!")
    print("You can chat normally or provide a resume and job description for analysis.")
    print("Type 'quit' to exit.\n")

    session_id = "user_session"

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye!")
            break

        if not user_input:
            continue

        # Get bot response
        response = chatbot.chat(user_input, session_id)
        print(f"\nBot: {response}\n")


# Example of how to use the direct analysis method
def example_direct_analysis():
    """Example showing direct resume analysis"""
    GOOGLE_API_KEY = "AIzaSyASZydNF55Bct2-3PQknOBjNuXpWqNSiic"

    chatbot = ResumeAnalyzerChatbot(
        google_api_key=GOOGLE_API_KEY,
        existing_resume_tool_function=example_resume_scorer
    )

    # Example resume (markdown)
    resume = """
# John Doe
## Software Engineer

### Experience
- **Senior Python Developer** at Tech Corp (2020-2024)
  - Developed ML models using scikit-learn and TensorFlow
  - Led team of 5 developers
  - Built RESTful APIs with Flask and FastAPI

### Skills
- Programming: Python, JavaScript, SQL
- ML/AI: scikit-learn, TensorFlow, pandas
- Tools: Git, Docker, Jenkinsq
    """

    job_description = """
We are looking for a Senior ML Engineer with:
- 3+ years Python development experience
- Experience with TensorFlow or PyTorch
- Container orchestration with Docker and Kubernetes
- AWS cloud platform experience
- Team leadership experience
    """

    result = chatbot.analyze_resume_directly(resume, job_description)
    print("-------------------result-----------------")
    print(result)


if __name__ == "__main__":
    main()