import os
import uuid
import time
import glob
import threading
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import your actual agent system
try:
    from config import Config
    from agents.agent import process_query
    config = Config()
    AGENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import agent system: {e}")
    config = None
    AGENTS_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================
app = FastAPI(title="Medical AI Assistant Backend", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_FOLDER = "uploads/backend"
SKIN_LESION_OUTPUT = "uploads/skin_lesion_output"
SPEECH_DIR = "uploads/speech"

for directory in [UPLOAD_FOLDER, SKIN_LESION_OUTPUT, SPEECH_DIR]:
    os.makedirs(directory, exist_ok=True)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# ============================================================================
# MODELS
# ============================================================================
class Message(BaseModel):
    role: str
    content: str
    has_image: Optional[bool] = False
    is_followup: Optional[bool] = False

class AnalysisContext(BaseModel):
    """Stores previous analysis results to avoid re-processing"""
    agent: str
    classification: Optional[str] = None
    confidence: Optional[float] = None
    response: str
    result_image: Optional[str] = None
    timestamp: float

class ChatRequest(BaseModel):
    query: str
    conversation_history: List[Message] = []
    last_analysis: Optional[AnalysisContext] = None  # NEW: Previous analysis context

class ChatResponse(BaseModel):
    status: str
    response: str
    agent: str
    timestamp: float
    classification: Optional[str] = None  # NEW: Include classification in response
    confidence: Optional[float] = None    # NEW: Include confidence in response

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def extract_response_from_agent(response_data: dict) -> tuple:
    """
    Extract readable response, agent name, and metadata from agent system output.
    
    Args:
        response_data: Dict returned from process_query
        
    Returns:
        tuple: (response_text, agent_name, classification, confidence)
    """
    try:
        # Get the output from the response
        output = response_data.get('output', None)
        agent_name = response_data.get('agent_name', 'Medical Assistant')
        
        # Extract classification and confidence if available
        classification = response_data.get('classification', None)
        confidence = response_data.get('confidence', None)
        
        # Handle different output types
        if output is None:
            return "No response generated", agent_name, classification, confidence
        
        # If output has a 'content' attribute (AIMessage)
        if hasattr(output, 'content'):
            response_text = output.content
        # If output is a string
        elif isinstance(output, str):
            response_text = output
        # If output is a dict
        elif isinstance(output, dict):
            response_text = output.get('content', str(output))
        else:
            response_text = str(output)
        
        # Check if response is empty
        if not response_text or response_text == "None":
            response_text = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
        
        return response_text, agent_name, classification, confidence
        
    except Exception as e:
        print(f"Error extracting response: {e}")
        return f"Error processing response: {str(e)}", "System", None, None

def build_context_prompt(query: str, last_analysis: AnalysisContext, conversation_history: List[Message]) -> str:
    """
    Build a context-aware prompt for follow-up questions.
    This ensures the agent uses previous analysis results instead of re-analyzing.
    
    Args:
        query: Current user query
        last_analysis: Previous analysis results
        conversation_history: Full conversation history
        
    Returns:
        str: Context-enriched prompt
    """
    context_parts = [
        "Previous Analysis Context:",
        f"- Agent: {last_analysis.agent}",
        f"- Analysis Response: {last_analysis.response}",
    ]
    
    if last_analysis.classification:
        context_parts.append(f"- Classification: {last_analysis.classification}")
    
    if last_analysis.confidence:
        context_parts.append(f"- Confidence: {last_analysis.confidence:.2%}")
    
    context_parts.extend([
        "",
        "The user has a follow-up question about the previous analysis:",
        f"Question: {query}",
        "",
        "Important: DO NOT re-analyze any images. Use the previous analysis results above to answer the follow-up question.",
        "Provide additional insights, explanations, or detailed information based on the existing analysis."
    ])
    
    return "\n".join(context_parts)

def is_followup_question(query: str, last_analysis: Optional[AnalysisContext]) -> bool:
    """
    Determine if the query is a follow-up question about previous analysis.
    
    Args:
        query: User's query
        last_analysis: Previous analysis context if available
        
    Returns:
        bool: True if this is a follow-up question
    """
    if not last_analysis:
        return False
    
    # Keywords that indicate follow-up questions
    followup_keywords = [
        "explain", "tell me more", "details", "elaborate", "what does",
        "why", "how", "report", "summary", "meaning", "significance",
        "تقرير", "اشرح", "تفاصيل", "ماذا يعني", "وضح"
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in followup_keywords)

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    return {
        "message": "Medical AI Assistant Backend",
        "version": "3.1",  # Updated version
        "status": "operational",
        "agents_available": AGENTS_AVAILABLE,
        "features": [
            "Context-aware conversations",
            "Image analysis without re-processing",
            "Multi-turn dialogue support"
        ],
        "endpoints": ["/health", "/chat", "/upload", "/validate"]
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "chat": "operational" if AGENTS_AVAILABLE else "limited",
            "upload": "operational" if AGENTS_AVAILABLE else "limited",
            "validation": "operational",
            "agents": "connected" if AGENTS_AVAILABLE else "mock_mode",
            "context_management": "enabled"  # NEW feature
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Handle text-only chat queries with context awareness.
    If last_analysis is provided, uses it for follow-up questions instead of re-analyzing.
    """
    try:
        if not AGENTS_AVAILABLE:
            return ChatResponse(
                status="error",
                response="Agent system not available. Please check config.py and agents/agent.py",
                agent="System",
                timestamp=time.time()
            )
        
        # Check if this is a follow-up question
        if request.last_analysis and is_followup_question(request.query, request.last_analysis):
            # Build context-aware prompt
            context_prompt = build_context_prompt(
                request.query, 
                request.last_analysis, 
                request.conversation_history
            )
            
            print(f"[FOLLOW-UP] Detected follow-up question. Using previous analysis context.")
            print(f"[FOLLOW-UP] Previous agent: {request.last_analysis.agent}")
            
            # Process with context
            response_data = process_query(context_prompt)
            response_text, agent_name, classification, confidence = extract_response_from_agent(response_data)
            
            # Return response with previous analysis metadata
            return ChatResponse(
                status="success",
                response=response_text,
                agent=f"{agent_name} (Follow-up)",
                classification=request.last_analysis.classification,
                confidence=request.last_analysis.confidence,
                timestamp=time.time()
            )
        else:
            # Process as new query
            print(f"[NEW QUERY] Processing new text query")
            response_data = process_query(request.query)
            response_text, agent_name, classification, confidence = extract_response_from_agent(response_data)
            
            return ChatResponse(
                status="success",
                response=response_text,
                agent=agent_name,
                classification=classification,
                confidence=confidence,
                timestamp=time.time()
            )
    
    except Exception as e:
        print(f"Error in /chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/upload")
async def upload_image(
    image: UploadFile = File(...),
    text: str = Form(""),
    conversation_history: str = Form("")  # NEW: Accept conversation history
):
    """
    Handle image upload with optional text query.
    This always performs a new analysis since a new image is provided.
    """
    
    if not AGENTS_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Agent system not available. Please check config.py and agents/agent.py"
        )
    
    # Validate file type
    if not image.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_extension = image.filename.rsplit('.', 1)[-1].lower()
    if file_extension not in ['png', 'jpg', 'jpeg']:
        raise HTTPException(status_code=400, detail="Invalid image format. Use PNG, JPG, or JPEG")
    
    # Save uploaded file
    filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    
    try:
        content = await image.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        print(f"[IMAGE UPLOAD] New image uploaded: {filename}")
        print(f"[IMAGE UPLOAD] Query: {text if text else 'Analyze this medical image'}")
        
        # Prepare input for agent system
        query_input = {
            "text": text if text else "Analyze this medical image",
            "image": file_path
        }
        
        # Process through your agent system
        response_data = process_query(query_input)
        
        # Extract readable response with metadata
        response_text, agent_name, classification, confidence = extract_response_from_agent(response_data)
        
        result = {
            "status": "success",
            "response": response_text,
            "agent": agent_name,
            "classification": classification,  # NEW: Include in response
            "confidence": confidence,          # NEW: Include in response
            "image_path": f"/uploads/backend/{filename}",
            "timestamp": time.time()
        }
        
        # Check for result images (e.g., skin lesion segmentation)
        if "SKIN_LESION" in agent_name.upper():
            segmentation_path = "uploads/skin_lesion_output/segmentation_plot.png"
            if os.path.exists(segmentation_path):
                result["result_image"] = f"/{segmentation_path}?v={int(time.time())}"
        
        print(f"[IMAGE UPLOAD] Analysis complete: {agent_name}")
        if classification:
            print(f"[IMAGE UPLOAD] Classification: {classification} ({confidence:.2%})")
        
        return result
    
    except Exception as e:
        print(f"Error in /upload endpoint: {str(e)}")
        # Clean up on error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")

@app.post("/validate")
async def validate(
    validation_result: str = Form(...),
    comments: str = Form(None)
):
    """Human-in-the-loop validation endpoint"""
    
    if validation_result not in ["approved", "rejected", "yes", "no"]:
        raise HTTPException(
            status_code=400, 
            detail="validation_result must be 'approved', 'rejected', 'yes', or 'no'"
        )
    
    # Normalize input
    is_approved = validation_result.lower() in ["approved", "yes"]
    
    if is_approved:
        message = "Diagnosis approved and saved to medical record."
    else:
        message = "Diagnosis rejected and flagged for manual review."
    
    return {
        "status": "success",
        "message": message,
        "comments": comments if comments else "No additional comments",
        "timestamp": time.time()
    }

# ============================================================================
# CLEANUP THREAD
# ============================================================================
def cleanup_old_files():
    """Remove old uploaded files every 10 minutes"""
    while True:
        try:
            # Clean uploaded images older than 1 hour
            for folder in [UPLOAD_FOLDER, SKIN_LESION_OUTPUT]:
                if os.path.exists(folder):
                    files = glob.glob(f"{folder}/*")
                    for file in files:
                        if os.path.isfile(file) and time.time() - os.path.getmtime(file) > 3600:
                            os.remove(file)
                            print(f"[CLEANUP] Removed old file: {file}")
        except Exception as e:
            print(f"[CLEANUP] Error: {e}")
        
        time.sleep(600)  # Run every 10 minutes

threading.Thread(target=cleanup_old_files, daemon=True).start()

# ============================================================================
# RUN SERVER
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Starting Medical AI Assistant Backend...")
    print("=" * 60)
    print(f"Version: 3.1 (Context-Aware)")
    print(f"Agents Available: {AGENTS_AVAILABLE}")
    if AGENTS_AVAILABLE:
        print("✓ Agent system loaded successfully")
    else:
        print("✗ Agent system not available - running in limited mode")
    print(f"✓ Context management: Enabled")
    print(f"✓ Follow-up question detection: Enabled")
    print(f"API Documentation: http://localhost:8001/docs")
    print(f"Interactive API: http://localhost:8001/redoc")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8001)