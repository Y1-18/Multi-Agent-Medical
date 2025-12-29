import streamlit as st
import requests
from PIL import Image
import io
import time
from typing import Dict, Any, Optional

# ============================================================================
# CONFIGURATION
# ============================================================================
BACKEND_URL = "http://localhost:8001"
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']

# ============================================================================
# PAGE SETUP
# ============================================================================
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background-color: #f5f7fa;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        padding: 2rem 1rem;
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown h1 {
        color: #ffffff !important;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    [data-testid="stSidebar"] .stMarkdown p {
        color: #b8c1cc !important;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: #2d3748;
        margin: 1.5rem 0;
    }
    
    /* Main title */
    h1 {
        color: #1a202c;
        font-weight: 700;
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: transparent !important;
        padding: 1rem 0 !important;
    }
    
    [data-testid="stChatMessageContent"] {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        max-width: 85%;
    }
    
    /* User messages - right aligned */
    .stChatMessage[data-testid="user"] [data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: auto;
    }
    
    .stChatMessage[data-testid="user"] [data-testid="stChatMessageContent"] p {
        color: white !important;
    }
    
    /* Assistant messages - left aligned */
    .stChatMessage[data-testid="assistant"] [data-testid="stChatMessageContent"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        margin-right: auto;
    }
    
    /* Agent label */
    .agent-label {
        display: inline-block;
        background-color: #edf2f7;
        color: #2d3748;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        margin-bottom: 0.75rem;
    }
    
    /* Validation warning */
    .validation-warning {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 0.75rem 1rem;
        margin-top: 1rem;
        border-radius: 6px;
        font-size: 0.9rem;
        color: #92400e;
    }
    
    /* Success message */
    .success-banner {
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 0.75rem 1rem;
        margin: 1rem 0;
        border-radius: 6px;
        color: #065f46;
    }
    
    /* Error message */
    .error-banner {
        background-color: #fee2e2;
        border-left: 4px solid #ef4444;
        padding: 0.75rem 1rem;
        margin: 1rem 0;
        border-radius: 6px;
        color: #991b1b;
    }
    
    /* Image container */
    .uploaded-image-container {
        margin: 0.75rem 0;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Chat input */
    .stChatInputContainer {
        border-top: 1px solid #e2e8f0;
        background-color: #ffffff;
        padding: 1rem 0;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 1rem;
        border: 1px dashed rgba(255,255,255,0.2);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Footer */
    .footer-text {
        text-align: center;
        color: #718096;
        font-size: 0.85rem;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid #e2e8f0;
        margin-top: 2rem;
    }
    
    /* Status indicator */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    
    .status-connected {
        background-color: #10b981;
        box-shadow: 0 0 8px #10b981;
    }
    
    .status-disconnected {
        background-color: #ef4444;
        box-shadow: 0 0 8px #ef4444;
    }
    
    /* Context indicator */
    .context-indicator {
        background-color: #e0e7ff;
        border-left: 3px solid #6366f1;
        padding: 0.5rem 0.75rem;
        margin-bottom: 0.75rem;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #4338ca;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "current_image" not in st.session_state:
    st.session_state.current_image = None

if "backend_status" not in st.session_state:
    st.session_state.backend_status = {"status": "unknown", "agents": "unknown"}

# NEW: Store structured analysis data to prevent re-analysis on follow-up questions
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def check_backend_health():
    """Check if backend is running and get status"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            st.session_state.backend_status = {
                "status": "connected",
                "agents": data.get("services", {}).get("agents", "unknown")
            }
            return True
        return False
    except:
        st.session_state.backend_status = {
            "status": "disconnected",
            "agents": "unavailable"
        }
        return False

def send_chat_request(query: str):
    """Send text-only chat request with full conversation history"""
    try:
        payload = {
            "query": query,
            "conversation_history": st.session_state.conversation_history,  # Send full history
            "last_analysis": st.session_state.last_analysis  # Send previous analysis to avoid re-processing
        }
        
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json=payload,
            timeout=90
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "status": "error",
                "response": f"Backend error ({response.status_code}): {response.text}",
                "agent": "System",
                "timestamp": time.time()
            }
    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "response": "Request timed out. The agent is processing your query. Please try again.",
            "agent": "System",
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "error",
            "response": f"Connection error: {str(e)}. Make sure the backend is running.",
            "agent": "System",
            "timestamp": time.time()
        }

def send_image_request(image_file, text: str):
    """Send image upload request"""
    try:
        files = {"image": (image_file.name, image_file.getvalue(), image_file.type)}
        data = {
            "text": text if text else "Analyze this medical image",
            "conversation_history": str(st.session_state.conversation_history)  # Include conversation context
        }
        
        response = requests.post(
            f"{BACKEND_URL}/upload",
            files=files,
            data=data,
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "status": "error",
                "response": f"Upload failed ({response.status_code}): {response.text}",
                "agent": "System",
                "timestamp": time.time()
            }
    except Exception as e:
        return {
            "status": "error",
            "response": f"Upload error: {str(e)}",
            "agent": "System",
            "timestamp": time.time()
        }

def save_analysis_result(result: Dict[str, Any]):
    """
    Save analysis result in session state for use in follow-up questions.
    This prevents re-analyzing the same image when user asks follow-up questions.
    """
    if result.get("status") == "success":
        st.session_state.last_analysis = {
            "agent": result.get("agent", "Unknown"),
            "classification": result.get("classification"),
            "confidence": result.get("confidence"),
            "response": result.get("response"),
            "result_image": result.get("result_image"),
            "timestamp": time.time()
        }

def check_validation_required(response_text: str, agent_name: str) -> bool:
    """Check if human validation is required"""
    validation_keywords = [
        "human validation required",
        "validation required",
        "please validate",
        "requires validation",
        "human-in-the-loop"
    ]
    
    validation_agents = [
        "HUMAN_VALIDATION",
        "BRAIN_TUMOR_AGENT",
        "CHEST_XRAY_AGENT", 
        "SKIN_LESION_AGENT"
    ]
    
    response_lower = response_text.lower()
    
    has_validation_keyword = any(keyword in response_lower for keyword in validation_keywords)
    is_validation_agent = any(agent in agent_name.upper() for agent in validation_agents)
    
    return has_validation_keyword or is_validation_agent

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("# Medical AI Assistant")
    st.markdown("Advanced multi-agent diagnostic system")
    
    st.markdown("---")
    
    # Backend status
    backend_connected = check_backend_health()
    status = st.session_state.backend_status
    
    if backend_connected:
        st.markdown(
            f'<div><span class="status-indicator status-connected"></span>'
            f'Backend: <strong>Connected</strong></div>',
            unsafe_allow_html=True
        )
        if status["agents"] == "connected":
            st.markdown("Agents: Active")
        elif status["agents"] == "mock_mode":
            st.markdown("Agents: Mock Mode")
    else:
        st.markdown(
            f'<div><span class="status-indicator status-disconnected"></span>'
            f'Backend: <strong>Disconnected</strong></div>',
            unsafe_allow_html=True
        )
        st.markdown("Start: `python backend.py`")
    
    st.markdown("---")
    
    # Display current context if an analysis has been performed
    if st.session_state.last_analysis:
        st.markdown("### 📋 Current Context")
        st.markdown(f"**Agent:** {st.session_state.last_analysis['agent']}")
        if st.session_state.last_analysis.get('classification'):
            st.markdown(f"**Classification:** {st.session_state.last_analysis['classification']}")
        if st.session_state.last_analysis.get('confidence'):
            st.markdown(f"**Confidence:** {st.session_state.last_analysis['confidence']:.1%}")
        st.markdown("---")
    
    st.markdown("### Capabilities")
    st.markdown("""
    **Text Analysis**
    - Medical knowledge queries (RAG)
    - Current medical news (Web Search)
    - Symptom evaluation
    - Drug information
    
    **Image Analysis**
    - Brain tumor detection (MRI)
    - COVID-19 detection (Chest X-ray)
    - Skin lesion classification
    """)
    
    st.markdown("---")
    
    st.markdown("### Upload Medical Image")
    uploaded_file = st.file_uploader(
        "Supported: PNG, JPG, JPEG",
        type=ALLOWED_EXTENSIONS,
        help="Upload MRI, X-ray, or skin lesion images",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        st.session_state.current_image = uploaded_file
        st.image(uploaded_file, caption="Ready to analyze", use_column_width=True)
        
        if st.button("Clear Image", use_container_width=True):
            st.session_state.current_image = None
            st.rerun()
    
    st.markdown("---")
    
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.session_state.current_image = None
        st.session_state.last_analysis = None  # Clear analysis context as well
        st.rerun()

# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================
st.title("Medical AI Assistant")
st.markdown("Ask medical questions or upload images for AI-powered analysis")
st.markdown("---")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Show context indicator for follow-up questions
        if message.get("is_followup") and message["role"] == "user":
            st.markdown(
                '<div class="context-indicator">📎 Follow-up question - Using previous analysis results</div>',
                unsafe_allow_html=True
            )
        
        # Show agent label for assistant messages
        if message["role"] == "assistant" and "agent" in message:
            st.markdown(
                f'<div class="agent-label">{message["agent"]}</div>',
                unsafe_allow_html=True
            )
        
        # Show uploaded image for user messages
        if "image" in message and message["image"] is not None:
            st.image(message["image"], width=400)
        
        # Show message content
        st.markdown(message["content"])
        
        # Show result image if available
        if "result_image" in message and message["result_image"]:
            try:
                result_url = f"{BACKEND_URL}{message['result_image']}"
                img_response = requests.get(result_url, timeout=10)
                if img_response.status_code == 200:
                    img = Image.open(io.BytesIO(img_response.content))
                    st.image(img, caption="Analysis Result", width=500)
            except Exception as e:
                st.caption(f"Could not load result image: {str(e)}")
        
        # Show validation warning if needed
        if message.get("requires_validation"):
            st.markdown(
                '<div class="validation-warning">⚠️ This diagnosis requires human validation. '
                'Healthcare professionals should review this result before clinical use.</div>',
                unsafe_allow_html=True
            )

# Chat input
user_input = st.chat_input("Type your medical question or describe the uploaded image...")

if user_input:
    # Check backend connection
    if not check_backend_health():
        st.error("Backend is not connected. Please start the backend server: `python backend.py`")
        st.stop()
    
    # Determine request type: new image analysis or follow-up question
    has_image = st.session_state.current_image is not None
    is_followup = st.session_state.last_analysis is not None and not has_image
    
    # Prepare user message
    user_message: Dict[str, Any] = {
        "role": "user",
        "content": user_input,
        "is_followup": is_followup
    }
    
    if has_image:
        user_message["image"] = st.session_state.current_image
    
    # Add to messages
    st.session_state.messages.append(user_message)
    
    # Display user message
    with st.chat_message("user"):
        if is_followup:
            st.markdown(
                '<div class="context-indicator">📎 Follow-up question - Using previous analysis results</div>',
                unsafe_allow_html=True
            )
        
        if user_message.get("image") is not None: 
            img = Image.open(user_message["image"])
            st.image(img, width=400)

        st.markdown(user_input)
    
    # Process request
    with st.chat_message("assistant"):
        with st.spinner("Processing your request through multi-agent system..."):
            # Send to appropriate endpoint
            if has_image:
                result = send_image_request(st.session_state.current_image, user_input)
                # Save the new analysis result for future reference
                save_analysis_result(result)
                st.session_state.current_image = None  # Clear after sending
            else:
                result = send_chat_request(user_input)
            
            # Extract response
            response_text = result.get("response", "No response received")
            agent_name = result.get("agent", "Medical Assistant")
            result_image = result.get("result_image")
            status = result.get("status", "success")
            
            # Handle errors
            if status == "error":
                st.error(response_text)
                agent_name = "System Error"
            
            # Display agent label
            st.markdown(
                f'<div class="agent-label">{agent_name}</div>',
                unsafe_allow_html=True
            )
            
            # Display response
            st.markdown(response_text)
            
            # Display result image if available
            if result_image:
                try:
                    result_url = f"{BACKEND_URL}{result_image}"
                    img_response = requests.get(result_url, timeout=10)
                    if img_response.status_code == 200:
                        img = Image.open(io.BytesIO(img_response.content))
                        st.image(img, caption="Analysis Result", width=500)
                except Exception as e:
                    st.caption(f"Could not load result image: {str(e)}")
            
            # Check for validation requirement
            requires_validation = check_validation_required(response_text, agent_name)
            
            if requires_validation:
                st.markdown(
                    '<div class="validation-warning">⚠️ This diagnosis requires human validation. '
                    'Healthcare professionals should review this result before clinical use.</div>',
                    unsafe_allow_html=True
                )
            
            # Save assistant message
            assistant_message = {
                "role": "assistant",
                "content": response_text,
                "agent": agent_name
            }
            
            if result_image:
                assistant_message["result_image"] = result_image
            
            if requires_validation:
                assistant_message["requires_validation"] = True
            
            st.session_state.messages.append(assistant_message)
            
            # Update conversation history with structured data
            st.session_state.conversation_history.append({
                "role": "user",
                "content": user_input,
                "has_image": has_image,
                "is_followup": is_followup
            })
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": response_text,
                "agent": agent_name
            })
    
    st.rerun()

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    '<div class="footer-text">'
    'Medical AI Assistant | Multi-Agent System with RAG, Web Search & Computer Vision<br>'
    'For informational and educational purposes only | Always consult qualified healthcare professionals'
    '</div>',
    unsafe_allow_html=True
)