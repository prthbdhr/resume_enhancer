from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException

from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import os
from pathlib import Path
import logging
from datetime import datetime
# from markdown import markdown
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# agent.py modifications (add these at the top)

# Import the ResumeKeywordMatcher class
from resume_matcher import ResumeKeywordMatcher

# Initialize matcher as None at module level
matcher: Optional[ResumeKeywordMatcher] = None


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


class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None


# run.py modifications
from agent import ResumeAnalyzerChatbot, ChatResponse, example_resume_scorer
from typing import Union

# Add this to the lifespan manager (after matcher initialization)
chatbot: Optional[ResumeAnalyzerChatbot] = None


@asynccontextmanager
async def lifespan_manager(app: FastAPI):
    """Initialize and clean up resources"""
    # Startup
    global matcher, chatbot
    try:
        api_key = "AIzaSyD7FPBzvTL7wokvxpVQM4lwiQ9slUmAY_M"  # Remove default in production
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set")

        logger.info("Initializing ResumeKeywordMatcher...")
        matcher = ResumeKeywordMatcher(api_key)

        logger.info("Initializing Resume Analyzer Chatbot...")
        chatbot = ResumeAnalyzerChatbot(
            google_api_key=api_key,
            existing_resume_tool_function=example_resume_scorer
        )

        logger.info("Service startup completed")
        yield

    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise
    finally:
        # Shutdown logic would go here
        logger.info("Shutting down service")


# Initialize FastAPI app with lifespan
fastapi_app = FastAPI(
    title="Resume Matcher API",
    description="""Advanced resume analysis with:
        - Keyword matching
        - ATS compatibility scanning
        - HR recommendations""",
    version="2.0.0",
    lifespan=lifespan_manager,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
ALLOWED_FILE_TYPES = {'.pdf', '.docx', '.doc', '.txt'}
MAX_FILE_SIZE_MB = 5


@fastapi_app.post("/api/analyze",
                  response_model=AnalysisResponse,
                  responses={
                      400: {"model": ErrorResponse},
                      500: {"model": ErrorResponse}
                  })
async def analyze_resume(
        resume_file: UploadFile = File(..., description="Resume file (PDF, DOCX, or TXT)"),
        job_description: str = Form(..., description="Job description text", min_length=50),
        include_ats_scan: bool = Form(False, description="Include ATS optimization analysis")
):
    """
    Analyze resume against job description with comprehensive matching report

    Returns:
    - Overall matching score (0-10)
    - Matching and missing keywords
    - Strengths and weaknesses
    - Section-wise analysis
    - Actionable recommendations
    - Optional ATS optimization scan
    """
    start_time = datetime.now()

    try:
        # Validate input file
        file_extension = Path(resume_file.filename).suffix.lower()
        if file_extension not in ALLOWED_FILE_TYPES:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_file_type",
                    "message": f"Unsupported file type. Allowed: {', '.join(ALLOWED_FILE_TYPES)}"
                }
            )

        # Check file size
        file_content = await resume_file.read()
        if len(file_content) > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "file_too_large",
                    "message": f"File exceeds maximum size of {MAX_FILE_SIZE_MB}MB"
                }
            )

        # Process files
        resume_text = matcher.extract_text_from_file(file_content, resume_file.filename)
        if not resume_text.strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "empty_resume",
                    "message": "No readable text found in resume"
                }
            )

        # Perform analysis
        analysis = matcher.analyze_resume_vs_jd(resume_text, job_description)
        recommendations = matcher.generate_recommendations(analysis, resume_text)

        # Optional ATS scan
        ats_scan = None
        if include_ats_scan:
            ats_scan = matcher.ats_scan_refinement(resume_text)

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
            processed_at=datetime.utcnow().isoformat(),
            processing_time_ms=round(processing_time, 2),
            ats_scan=ats_scan
        )

    except HTTPException as he:
        logger.error(f"Client error: {he.detail}")
        raise
    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": "Failed to analyze documents",
                "details": str(e)
            }
        )


@fastapi_app.get("/api/health")
async def health_check():
    """Comprehensive health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.isoformat(datetime.now()),
        "services": {
            "gemini_api": matcher is not None,
            "chatbot": chatbot is not None
        }
    }




# Add these new models for chat API
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ResumeAnalysisRequest(BaseModel):
    resume_markdown: str
    job_description: str
    session_id: str = "default"


# Add these new endpoints
@fastapi_app.post("/api/chat", response_model=ChatResponse,
                  responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def chat_with_bot(request: ChatRequest):
    """
    Chat with the resume analysis bot

    Args:
        message: User's message
        session_id: Conversation session ID (defaults to "default")

    Returns:
        Bot's response with session information
    """
    try:
        if not chatbot:
            raise HTTPException(status_code=503, detail="Chatbot service not available")

        response = chatbot.chat(request.message, request.session_id)
        return ChatResponse(
            response=response,
            session_id=request.session_id,
            processed_at=datetime.isoformat(datetime.now())
        )

    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "chat_error",
                "message": "Failed to process chat message",
                "details": str(e)
            }
        )


@fastapi_app.post("/api/analyze-with-chat", response_model=ChatResponse,
                  responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def analyze_resume_with_chat(request: ResumeAnalysisRequest):
    """
    Directly analyze a resume with job description through the chatbot

    Args:
        resume_markdown: Resume in markdown format
        job_description: Job description text
        session_id: Conversation session ID (defaults to "default")

    Returns:
        Analysis results with interpretation from the chatbot
    """
    try:
        if not chatbot:
            raise HTTPException(status_code=503, detail="Chatbot service not available")

        response = chatbot.analyze_resume_directly(
            request.resume_markdown,
            request.job_description,
            request.session_id
        )
        return ChatResponse(
            response=response,
            session_id=request.session_id,
            processed_at=datetime.isoformat(datetime.now())
        )

    except Exception as e:
        logger.error(f"Resume analysis error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "analysis_error",
                "message": "Failed to analyze resume",
                "details": str(e)
            }
        )

@fastapi_app.get("/")
async def root():
    """API information endpoint"""
    return {
        "service": "Resume Matcher API",
        "version": "2.0.0",
        "documentation": {
            "swagger": "/api/docs",
            "redoc": "/api/redoc"
        },
        "endpoints": {
            "analyze": {
                "path": "/api/analyze",
                "methods": ["POST"],
                "description": "Analyze resume against job description"
            },
            "chat": {
                "path": "/api/chat",
                "methods": ["POST"],
                "description": "Chat with resume analysis bot"
            },
            "analyze-with-chat": {
                "path": "/api/analyze-with-chat",
                "methods": ["POST"],
                "description": "Direct resume analysis through chatbot"
            },
            "health": {
                "path": "/api/health",
                "methods": ["GET"],
                "description": "Service health check"
            }
        }
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "run:fastapi_app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("DEBUG", "false").lower() == "true",
        workers=int(os.getenv("WORKERS", 1)),
        log_level="info"
    )