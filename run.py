# Import standard and third-party libraries
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import os
from pathlib import Path
import logging
from datetime import datetime

# Configure logging for the application
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import core resume matcher logic
from resume_matcher import ResumeKeywordMatcher
# Import chatbot agents and response models
from agent import ResumeAnalyzerChatbot, ChatResponse, ResumeEnhancerChatbot, ResumeExtractionAgent

# Global variables for matcher and chatbots
matcher: Optional[ResumeKeywordMatcher] = None
chatbot: Optional[ResumeAnalyzerChatbot] = None
enhancer_chatbot: Optional[ResumeEnhancerChatbot] = None
extraction_agent: Optional[ResumeExtractionAgent] = None

# -----------------------------
# Pydantic Response Models
# -----------------------------
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

# -----------------------------
# FastAPI App Initialization
# -----------------------------
@asynccontextmanager
async def lifespan_manager(app: FastAPI):
    """Initialize and clean up resources for the app lifecycle."""
    global matcher, chatbot, enhancer_chatbot, extraction_agent
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set")
        logger.info("Initializing ResumeKeywordMatcher...")
        matcher = ResumeKeywordMatcher(api_key)
        logger.info("Initializing Resume Analyzer Chatbot...")
        from agent import ResumeScorer
        def scorer_func(resume_markdown, job_description):
            scorer = ResumeScorer(api_key)
            return scorer.analyze_resume(resume_markdown, job_description).dict()
        chatbot = ResumeAnalyzerChatbot(
            google_api_key=api_key,
            existing_resume_tool_function=scorer_func
        )
        logger.info("Initializing Resume Enhancer Chatbot...")
        enhancer_chatbot = ResumeEnhancerChatbot(google_api_key=api_key)
        logger.info("Initializing Resume Extraction Agent...")
        extraction_agent = ResumeExtractionAgent(google_api_key=api_key)
        logger.info("Service startup completed")
        yield
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise
    finally:
        logger.info("Shutting down service")

# Create FastAPI app with custom lifespan
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

# Enable CORS for all origins (customize in production)
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Constants for file validation
# -----------------------------
ALLOWED_FILE_TYPES = {'.pdf', '.docx', '.doc', '.txt'}
MAX_FILE_SIZE_MB = 5

# -----------------------------
# API Endpoints
# -----------------------------
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
    Analyze resume against job description with comprehensive matching report.
    Returns overall score, keyword matches, strengths/weaknesses, section analysis, and recommendations.
    """
    start_time = datetime.now()
    try:
        file_extension = Path(resume_file.filename).suffix.lower()
        if file_extension not in ALLOWED_FILE_TYPES:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_file_type",
                    "message": f"Unsupported file type. Allowed: {', '.join(ALLOWED_FILE_TYPES)}"
                }
            )
        file_content = await resume_file.read()
        if len(file_content) > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "file_too_large",
                    "message": f"File exceeds maximum size of {MAX_FILE_SIZE_MB}MB"
                }
            )
        resume_text = matcher.extract_text_from_file(file_content, resume_file.filename)
        if not resume_text.strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "empty_resume",
                    "message": "No readable text found in resume"
                }
            )
        analysis = matcher.analyze_resume_vs_jd(resume_text, job_description)
        
        # Ensure analysis has required fields
        if not isinstance(analysis, dict):
            analysis = {"overall_match_score": 5.0, "matching_keywords": [], "missing_keywords": [], "strengths": [], "weaknesses": [], "section_analysis": {}}
        
        if "overall_match_score" not in analysis:
            analysis["overall_match_score"] = 5.0
        if "matching_keywords" not in analysis:
            analysis["matching_keywords"] = []
        if "missing_keywords" not in analysis:
            analysis["missing_keywords"] = []
        if "strengths" not in analysis:
            analysis["strengths"] = []
        if "weaknesses" not in analysis:
            analysis["weaknesses"] = []
        if "section_analysis" not in analysis:
            analysis["section_analysis"] = {}
            
        recommendations = matcher.generate_recommendations(analysis, resume_text)
        ats_scan = None
        # Temporarily disabled ATS scan due to method issues
        # if include_ats_scan:
        #     try:
        #         ats_scan = {"ats_tips": matcher._get_ats_tips(resume_text)}
        #     except Exception as e:
        #         logger.warning(f"ATS scan failed: {str(e)}")
        #         ats_scan = {"ats_tips": ["Use standard section headings", "Include relevant keywords", "Avoid complex formatting"]}
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
    """Comprehensive health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.isoformat(datetime.now()),
        "services": {
            "gemini_api": matcher is not None,
            "chatbot": chatbot is not None
        }
    }

@fastapi_app.post("/api/extract-and-analyze", response_model=Dict[str, Any],
                  responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def extract_and_analyze(
    resume_file: UploadFile = File(..., description="Resume file (PDF, DOCX, or TXT)"),
    job_description: str = Form(..., description="Job description text", min_length=50),
    analyze_with_agent: bool = Form(True, description="Also analyze extracted text with analyzer agent")
):
    """
    Extract resume text using LLM and (optionally) analyze it for the job description.
    Returns extracted text and analysis.
    """
    try:
        if not extraction_agent:
            raise HTTPException(status_code=503, detail="Extraction agent service not available")
        file_content = await resume_file.read()
        extraction_result = extraction_agent.extract_and_analyze(file_content, resume_file.filename, job_description)
        if analyze_with_agent and extraction_result.get("extracted_text"):
            analysis = chatbot.resume_tool_function(extraction_result["extracted_text"], job_description)
            extraction_result["full_analysis"] = analysis
        return extraction_result
    except Exception as e:
        logger.error(f"Extraction/analysis error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "extract_analyze_error",
                "message": "Failed to extract/analyze resume",
                "details": str(e)
            }
        )

# -----------------------------
# Chat API Models
# -----------------------------
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ResumeAnalysisRequest(BaseModel):
    resume_markdown: str
    job_description: str
    session_id: str = "default"

class EnhanceResumeChatRequest(BaseModel):
    message: str
    resume_markdown: str = ""
    session_id: str = "default"

@fastapi_app.post("/api/chat", response_model=ChatResponse,
                  responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def chat_with_bot(request: ChatRequest):
    """
    Chat with the resume analysis bot. Maintains session history.
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
    Directly analyze a resume with job description through the chatbot.
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

@fastapi_app.post("/api/enhance-resume-chat", response_model=ChatResponse,
                  responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def enhance_resume_chat(request: EnhanceResumeChatRequest):
    """
    Continuous resume enhancement chat with the AI resume coach.
    """
    try:
        if not enhancer_chatbot:
            raise HTTPException(status_code=503, detail="Enhancer chatbot service not available")
        user_message = request.message
        if request.resume_markdown:
            user_message = f"Resume Content:\n{request.resume_markdown}\n\n{request.message}"
        response = enhancer_chatbot.chat(user_message, request.session_id)
        return ChatResponse(
            response=response,
            session_id=request.session_id,
            processed_at=datetime.isoformat(datetime.now())
        )
    except Exception as e:
        logger.error(f"Enhance chat error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "enhance_chat_error",
                "message": "Failed to process enhancement chat message",
                "details": str(e)
            }
        )

@fastapi_app.get("/")
async def root():
    """API information endpoint."""
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

# -----------------------------
# Main Entrypoint
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    import sys
    from dotenv import load_dotenv
    load_dotenv()
    port = int(os.getenv("PORT", 8000))
    for i, arg in enumerate(sys.argv):
        if arg in ("--port", "-p") and i + 1 < len(sys.argv):
            port = int(sys.argv[i + 1])
    uvicorn.run(
        "run:fastapi_app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=port,
        reload=os.getenv("DEBUG", "false").lower() == "true",
        workers=int(os.getenv("WORKERS", 1)),
        log_level="info"
    )