# Resume Matcher API - Project Overview

## 🎯 Core Purpose
AI-powered resume analysis service that matches resumes against job descriptions using Google's Gemini 2.5 Flash LLM with intelligent chatbot capabilities.

## 📁 Project Structure

### **run.py** (Main FastAPI Application)
- **FastAPI server** with CORS middleware
- **Lifespan management** for service initialization
- **5 main endpoints** for different use cases
- **Pydantic models** for request/response validation
- **Error handling** and logging throughout

### **resume_matcher.py** (Core Analysis Engine)
- **ResumeKeywordMatcher class** - the heart of analysis
- **Multi-format file processing** (PDF, DOCX, TXT)
- **Single LLM call optimization** for efficiency
- **Structured JSON analysis** with sections scoring
- **ATS optimization recommendations**

### **agent.py** (AI Chat Agents)
- **3 specialized chatbots** for different scenarios
- **LangChain integration** with tool calling
- **Session-based conversation memory**
- **Automatic model switching** (Gemini → GPT-3.5 after limits)

## 🚀 Key Features

### 1. **File Analysis** (`/api/analyze`)
```
📄 Upload Resume (PDF/DOCX/TXT) + Job Description
↓
🔍 Extract & analyze text
↓
🎯 Score matching (0-10 scale)
↓
📊 Detailed section breakdown
↓
💡 Actionable recommendations
```

### 2. **Interactive Chat** (`/api/chat`)
- **Smart detection**: Auto-analyzes when both resume & JD provided
- **Conversational memory**: Maintains session history
- **General career advice**: Beyond just analysis

### 3. **Direct Analysis** (`/api/analyze-with-chat`)
- **Structured input**: Resume markdown + job description
- **Chatbot interpretation**: Human-friendly analysis explanation

### 4. **Resume Enhancement** (`/api/enhance-resume-chat`)
- **Continuous coaching**: Iterative improvement suggestions
- **No JD required**: General resume enhancement
- **Section-specific feedback**: Targeted improvements

## 🧠 AI Architecture

### **LLM Strategy**
- **Primary**: Gemini 2.5 Flash (fast, cost-effective)
- **Fallback**: OpenAI GPT-3.5 Turbo (after 50 calls)
- **Usage tracking**: Automatic model switching

### **Analysis Approach**
- **Single comprehensive prompt** instead of multiple API calls
- **Structured JSON output** for consistent parsing
- **Keyword matching** with synonym awareness
- **Section-by-section scoring** (Experience, Education, Skills, etc.)

### **Chat Memory**
- **Per-session history** using LangChain
- **Tool integration** for resume analysis
- **Context awareness** across conversation

## 📊 Output Structure

### Analysis Response
```json
{
  "overall_score": 7.5,
  "match_percentage": 75.0,
  "matching_keywords": ["Python", "Machine Learning"],
  "missing_keywords": ["AWS", "Docker"],
  "strengths": ["Strong technical skills", "Relevant experience"],
  "weaknesses": ["Missing cloud expertise", "No metrics"],
  "section_analysis": {
    "WORK EXPERIENCE": {
      "match_score": 8.0,
      "matching_keywords": [...],
      "missing_keywords": [...]
    }
  },
  "recommendations": {
    "priority_actions": [...],
    "section_improvements": {...},
    "ats_optimization": [...]
  }
}
```

## 🔧 Technical Highlights

### **Error Handling**
- File size validation (5MB limit)
- Supported format checking
- Empty content detection
- Graceful LLM failure handling

### **Performance**
- Processing time tracking
- Efficient text extraction
- Minimal LLM calls
- Response caching potential

### **Security**
- File type validation
- Content size limits
- API key management
- CORS configuration

### **Scalability**
- Async FastAPI endpoints
- Configurable workers
- Environment-based config
- Health check monitoring

## 🎛️ Configuration Options
- **HOST/PORT**: Server binding
- **DEBUG**: Development mode
- **WORKERS**: Concurrent processing
- **ALLOWED_ORIGINS**: CORS settings
- **API Keys**: Gemini/OpenAI credentials

## 💡 Use Cases
1. **Job Seekers**: Optimize resumes for specific positions
2. **Recruiters**: Quick candidate screening
3. **Career Coaches**: Provide data-driven advice
4. **HR Teams**: Standardize resume evaluation
5. **Students**: Learn resume best practices

## ⚡ Quick Start
```bash
python run.py --port 8000
# Server starts at http://localhost:8000
# API docs at http://localhost:8000/api/docs
```