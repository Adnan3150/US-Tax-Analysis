import os
import json
import time
import uuid
import boto3
import nest_asyncio
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dotenv import load_dotenv
from llama_index.core import Settings, Document, VectorStoreIndex
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from src import aws_extract_tool, field_mapping, vector_store, config, recommendation_generator,advanced_extraction_tool

# ========== INIT ==========
nest_asyncio.apply()
load_dotenv()

# Page config with custom styling
st.set_page_config(
    page_title="Spsoft AI 1040 Tax Extractor", 
    page_icon="SP",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved UI with draggable chat panel
st.markdown("""
<style>
/* Main content area */
.main {
    padding-top: 1rem;
    padding-bottom: 80px;
}

/* Sidebar styling */
.sidebar .sidebar-content {
    background-color: #f8f9fa;
    padding: 1rem;
}

/* Cards and containers */
.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
    border: 1px solid #e9ecef;
}

.tax-section-header {
    background: linear-gradient(135deg, #87CEFA, #B0E0E6);
    color: #003366;
    padding: 0.75rem 1.5rem;
    border-radius: 8px;
    margin: 1rem 0 0.5rem 0;
    font-weight: 600;
    font-size: 1.1rem;
}

.success-message {
    background-color: #d9f3ff;
    color: #004c66;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #b3e0f2;
    font-weight: 500;
    margin: 15px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

.upload-section {
    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    padding: 3rem;
    border-radius: 12px;
    border: 2px dashed #add8e6;
    text-align: center;
    margin-bottom: 2rem;
    transition: all 0.3s ease;
}

.upload-section:hover {
    border-color: #87CEFA;
    background: linear-gradient(135deg, #e6f7ff, #e0f4ff);
}

.data-table {
    background: white;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    border: 1px solid #e9ecef;
}

.section-divider {
    border-top: 2px solid #e9ecef;
    margin: 2.5rem 0;
}

/* Chat Panel Styles */
.chat-panel {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: white;
    border-radius: 20px 20px 0 0;
    box-shadow: 0 -10px 30px rgba(0,0,0,0.2);
    z-index: 1000;
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    border: 2px solid #e9ecef;
    border-bottom: none;
    max-height: 80vh;
    display: flex;
    flex-direction: column;
}

.chat-panel.minimized {
    transform: translateY(calc(100% - 70px));
}

.chat-panel.hidden {
    transform: translateY(100%);
}

.chat-drag-handle {
    background: linear-gradient(135deg, #87CEFA, #B0E0E6);
    color: #003366;
    padding: 15px 20px;
    border-radius: 20px 20px 0 0;
    cursor: grab;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-weight: 600;
    user-select: none;
    min-height: 40px;
}

.chat-drag-indicator {
    width: 40px;
    height: 4px;
    background: rgba(255,255,255,0.6);
    border-radius: 2px;
    margin: 0 auto 10px auto;
}

.chat-controls {
    display: flex;
    gap: 10px;
    align-items: center;
}

.chat-close-btn, .chat-minimize-btn {
    background: rgba(255,255,255,0.2);
    border: none;
    color: white;
    padding: 8px 12px;
    border-radius: 20px;
    cursor: pointer;
    font-weight: 600;
    transition: all 0.2s ease;
    font-size: 12px;
}

.chat-close-btn:hover, .chat-minimize-btn:hover {
    background: rgba(255,255,255,0.3);
    transform: scale(1.05);
}

.chat-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    max-height: calc(80vh - 70px);
}

.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    background: #fafafa;
    max-height: 400px;
}

.chat-message {
    margin-bottom: 15px;
    padding: 12px 16px;
    border-radius: 12px;
    line-height: 1.5;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    animation: fadeInUp 0.3s ease;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.user-message {
    background: linear-gradient(135deg, #e3f2fd, #d0ecff);
    border-left: 4px solid #87CEFA;
    margin-left: 40px;
}

.assistant-message {
    background: linear-gradient(135deg, #f0faff, #d8eff7);
    border-left: 4px solid #87CEFA;
    margin-right: 40px;
}

.chat-input-container {
    padding: 20px;
    background: white;
    border-top: 1px solid #e9ecef;
    border-radius: 0 0 20px 20px;
}

.chat-input-form {
    display: flex;
    gap: 10px;
    align-items: flex-end;
}

.chat-input {
    flex: 1;
    border: 2px solid #e9ecef;
    border-radius: 25px;
    padding: 12px 20px;
    font-size: 14px;
    resize: none;
    font-family: inherit;
    max-height: 100px;
    min-height: 45px;
    transition: all 0.3s ease;
}

.chat-input:focus {
    outline: none;
    border-color: #87CEFA;
    box-shadow: 0 0 0 3px rgba(135, 206, 250, 0.2);
}

.chat-send-btn {
    background: linear-gradient(135deg, #87CEFA, #B0E0E6);
    color: #003366;
    border: none;
    border-radius: 50%;
    width: 45px;
    height: 45px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    transition: all 0.3s ease;
    flex-shrink: 0;
}

.chat-send-btn:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 12px rgba(135, 206, 250, 0.4);
}

.chat-send-btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
}

/* Chat Toggle Button */
.chat-toggle-btn {
    position: fixed;
    bottom: 20px;
    right: 20px;
    width: 60px;
    height: 60px;
    background: linear-gradient(135deg, #87CEFA, #B0E0E6);
    color: white;
    border: none;
    border-radius: 50%;
    cursor: pointer;
    font-size: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 20px rgba(135, 206, 250, 0.3);
    z-index: 999;
    transition: all 0.3s ease;
}

.chat-toggle-btn:hover {
    transform: scale(1.1);
    box-shadow: 0 6px 25px rgba(135, 206, 250, 0.4);
}

.chat-toggle-btn.hidden {
    display: none;
}

.chat-status-indicator {
    position: absolute;
    top: -2px;
    right: -2px;
    width: 18px;
    height: 18px;
    background: #00bfff;
    border: 2px solid white;
    border-radius: 50%;
    font-size: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
}

.chat-status-indicator.active {
    background: #00ced1;
}

/* Form input enhancements */
.stTextInput > div > div > input {
    border-radius: 25px;
    border: 2px solid #e9ecef;
    padding: 0.75rem 1rem;
    font-size: 0.95rem;
    transition: all 0.3s ease;
}

.stTextInput > div > div > input:focus {
    border-color: #87CEFA;
    box-shadow: 0 0 0 3px rgba(135, 206, 250, 0.2);
}

.stButton > button {
    border-radius: 25px;
    border: none;
    padding: 0.75rem 2rem;
    font-weight: 600;
    transition: all 0.3s ease;
    background: linear-gradient(135deg, #87CEFA, #B0E0E6);
    color: #003366;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(135, 206, 250, 0.3);
}

.streamlit-expanderHeader {
    background-color: #f8f9fa;
    border-radius: 8px;
    border: 1px solid #e9ecef;
    font-weight: 600;
}

.streamlit-expanderContent {
    border: 1px solid #e9ecef;
    border-top: none;
    border-radius: 0 0 8px 8px;
    background-color: white;
}

[data-testid="metric-container"] {
    background: white;
    border: 1px solid #e9ecef;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Responsive adjustments */
@media (max-width: 768px) {
    .upload-section {
        padding: 2rem 1rem;
    }

    .chat-panel {
        border-radius: 15px 15px 0 0;
    }

    .chat-toggle-btn {
        width: 55px;
        height: 55px;
        bottom: 15px;
        right: 15px;
    }

    .user-message {
        margin-left: 20px;
    }

    .assistant-message {
        margin-right: 20px;
    }
}
</style>

<script>
document.addEventListener('DOMContentLoaded', function() {
    // Chat panel functionality will be handled by Streamlit components
});
</script>
""", unsafe_allow_html=True)

# ========== SESSION STATE INITIALIZATION ==========
def initialize_session_state():
    """Initialize all session state variables"""
    if 'session_id' not in st.session_state:
        print("session_id initialized..!")
        st.session_state.session_id=None
    if 'extracted_data' not in st.session_state:
        st.session_state.extracted_data = None
    if 'recommendation_data' not in st.session_state:
        st.session_state.recommendation_data=None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    if 'uploaded_filename' not in st.session_state:
        st.session_state.uploaded_filename = None
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = "ready"
    if 'sectioned_data' not in st.session_state:
        st.session_state.sectioned_data=None
    
    # Chat-related session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'chat_panel_open' not in st.session_state:
        st.session_state.chat_panel_open = False
    if 'chat_panel_minimized' not in st.session_state:
        st.session_state.chat_panel_minimized = False
    if 'chat_engine' not in st.session_state:
        st.session_state.chat_engine = None
    if 'sectioned_data' not in st.session_state:
        st.session_state.sectioned_data = None

def process_uploaded_file(pdf_file):
    """Process the uploaded PDF file and extract tax data"""
    try:
        st.session_state.processing_status = "processing"
        session_id=st.session_state.session_id
        print("session-id:",session_id)
        # Save the PDF file locally
        print("extracting fields...!")
        # output_text_file_path=aws_extract_tool.extract_data(pdf_file,session_id)
        # output_text_file_path=advanced_extraction_tool.extract_data(pdf_file,session_id)
        # Show progress messages
        output_text_file_path=r'F:\adnan\Adnan\1040_tax_analysis\aws_textract\Tax-architecture-dev\extracted_raw_data\new_optimized_llm_optimized.txt'
        with st.spinner("Processing your tax document..."):
            progress_container = st.container()
            
            with progress_container:
                st.markdown(
                    '<div class="success-message">🔄 Processing with AWS Textract...</div>',
                    unsafe_allow_html=True
                )
                
                # Simulate processing time
                time.sleep(2)
                
                st.markdown(
                    '<div class="success-message">✅ AWS Textract extraction complete</div>',
                    unsafe_allow_html=True
                )
                
                st.markdown(
                    '<div class="success-message">🔄 Extracting tax fields...</div>',
                    unsafe_allow_html=True
                )
                output_json,section_data=field_mapping.field_map(output_text_file_path,session_id)
                st.session_state.sectioned_data=section_data
                # Load dummy data (replace with actual processing)
                # path = config.PATHS.get("json_data_path","")
                # # path=os.path.join(path,f'{st.session_sate.session_id}_fields.json')
                # path=os.path.join(path,'dummy_data_aishwarya.json')
                # with open(path, 'r') as file:
                #     output_json = json.load(file)
                # Ensure JSON object  
                 
                if isinstance(output_json, str):
                    extracted_json = json.loads(output_json)
                else:
                    extracted_json = output_json
                recommendation_json=recommendation_generator.generate_recommendations(extracted_json,st.session_state.session_id)
                # Store in session state
                st.session_state.extracted_data = extracted_json
                st.session_state.processing_complete = True
                st.session_state.processing_status = "complete"
                st.session_state.uploaded_filename = pdf_file.name
                st.session_state.recommendation_data=recommendation_json
                
                st.markdown(
                    '<div class="success-message">✅ Tax data extraction complete!</div>',
                    unsafe_allow_html=True
                )
                
                time.sleep(1)
                progress_container.empty()
        
        return True
        
    except Exception as e:
        st.session_state.processing_status = "error"
        st.error(f"Failed to extract fields: {str(e)}")
        return False

def initialize_chat_engine():
    """Initialize the chat engine with extracted tax data"""
    if st.session_state.chat_engine is None:
        try:
            index = vector_store.build_vector_chat_engine_from_text(st.session_state.sectioned_data)
            st.session_state.chat_engine = index.as_chat_engine(
                chat_mode="context",
                system_prompt="""You are a helpful tax assistant AI with access to the user's tax document data.
                Please provide helpful, accurate information about their tax situation. You can answer questions about:
                - Their income, deductions, and tax liability
                - Tax planning advice
                - Explanations of tax terms and concepts
                - Analysis of their tax situation
                Always be helpful and provide specific information when possible based on their tax data."""
            )
            return True
        except Exception as e:
            st.error(f"Failed to initialize chat engine: {str(e)}")
            return False
    return True

def render_chat_panel():
    """Render the draggable chat panel"""
    if not st.session_state.chat_panel_open:
        return
    
    # Chat panel state classes
    panel_classes = ["chat-panel"]
    if st.session_state.chat_panel_minimized:
        panel_classes.append("minimized")
    
    # Chat panel HTML
    chat_panel_html = f"""
    <div class="{' '.join(panel_classes)}" id="chatPanel">
        <div class="chat-drag-handle" id="chatDragHandle">
            <div>
                <div class="chat-drag-indicator"></div>
                <span>🤖 AI Tax Assistant</span>
            </div>
            <div class="chat-controls">
                <button class="chat-minimize-btn" onclick="toggleChatMinimize()">
                    {'▼' if not st.session_state.chat_panel_minimized else '▲'}
                </button>
                <button class="chat-close-btn" onclick="closeChatPanel()">✕</button>
            </div>
        </div>
        <div class="chat-content" style="{'display: none;' if st.session_state.chat_panel_minimized else ''}">
            <div class="chat-messages" id="chatMessages">
                <!-- Chat messages will be populated by Streamlit -->
            </div>
            <div class="chat-input-container">
                <div class="chat-input-form">
                    <!-- Chat input will be handled by Streamlit form -->
                </div>
            </div>
        </div>
    </div>
    
    <script>
        function toggleChatMinimize() {{
            // This will be handled by Streamlit callback
            console.log('Toggle minimize');
        }}
        
        function closeChatPanel() {{
            // This will be handled by Streamlit callback
            console.log('Close panel');
        }}
    </script>
    """
    
    st.markdown(chat_panel_html, unsafe_allow_html=True)

def render_chat_toggle_button():
    """Render the floating chat toggle button"""
    if st.session_state.chat_panel_open:
        return
    
    # Determine button status
    status_class = "active" if st.session_state.chat_engine else ""
    message_count = len(st.session_state.chat_history)
    
    button_html = f"""
    <div class="chat-toggle-btn" onclick="openChatPanel()">
        💬
        {f'<div class="chat-status-indicator {status_class}">{message_count}</div>' if message_count > 0 else ''}
    </div>
    
    <script>
        function openChatPanel() {{
            // This will be handled by Streamlit callback
            console.log('Open chat panel');
        }}
    </script>
    """
    
    st.markdown(button_html, unsafe_allow_html=True)

# Initialize session state
initialize_session_state()
st.session_state.session_id=uuid.uuid4().hex

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("### SP AI Tax Extractor")
    st.markdown("---")
    
    # Status indicator
    status_container = st.container()
    with status_container:
        if st.session_state.processing_complete:
            st.success("✅ Processing Complete")
            if st.session_state.uploaded_filename:
                st.info(f"📄 {st.session_state.uploaded_filename}")
        elif st.session_state.processing_status == "processing":
            st.warning("🔄 Processing...")
        elif st.session_state.processing_status == "error":
            st.error("❌ Processing Failed")
        else:
            st.info("📤 Ready to Process")
    
    st.markdown("---")
    
    # Navigation
    menu_option = st.selectbox(
        "📋 Navigation",
        ["📊 Overview", "📄 Documents", "📈 Analysis", "💡 Insights"],
        index=0
    )
    
    st.markdown("---")
    
    # Chat status
    if st.session_state.chat_engine:
        st.success("🤖 Chat Assistant Active")
        st.caption(f"💬 {len(st.session_state.chat_history)} messages")
    else:
        st.info("🤖 Chat Assistant Inactive")
        st.caption("Process a document to enable")
    
    # Chat controls in sidebar
    if st.session_state.processing_complete:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💬 Open Chat", use_container_width=True):
                st.session_state.chat_panel_open = True
                st.session_state.chat_panel_minimized = False
                # Initialize chat engine if not already done
                # raw_text_path = config.PATHS.get("raw_text_path","")

                # raw_text_path=os.path.join(raw_text_path, f'{st.session_state.session_id}_text.txt')
                # raw_text_path=r'F:\adnan\Adnan\1040_tax_analysis\aws_textract\Tax-architecture-dev\sectioned_data\e7213e5c39c94b238b8f14d616e6549a_sectioned_data.json'
                # raw_text_path=os.path.join(raw_text_path, 'output_precise_positioning.txt')
                if not st.session_state.chat_engine:
                    with st.spinner("Initializing AI assistant..."):
                        initialize_chat_engine()
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
    
    st.markdown("---")
    
    # Actions
    if st.button("🔄 Reset Session", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    st.markdown("---")
    st.markdown("**📞 Support**")
    st.markdown("shaik.adnan@spsoft.in")

# ========== MAIN CONTENT ==========
st.title("SP AI-Powered IRS Form 1040 Extractor")
st.markdown("Transform your tax documents into structured data with AI-powered extraction")

# ========== FILE UPLOAD SECTION ==========
if not st.session_state.processing_complete and st.session_state.processing_status != "processing":
    with st.container():
        st.markdown("### 📄 Upload Your Tax Document")
        st.markdown("Drop your IRS Form 1040 PDF here or click to browse")

        # Move uploader OUTSIDE custom <div> (or don't wrap it at all)
        pdf_file = st.file_uploader(
            "Choose a PDF file", 
            type="pdf", 
            label_visibility="collapsed",
            key="pdf_uploader"
        )

        st.markdown("**Supported:** PDF files up to 200MB")

        # Process file if uploaded
        if pdf_file is not None and not st.session_state.get("file_uploaded", False):
            st.session_state.file_uploaded = True
            if process_uploaded_file(pdf_file):
                st.rerun()

  
# ========== PROCESSING STATUS ==========
elif st.session_state.processing_status == "processing":
    st.info("🔄 Processing your tax document... This may take a few moments.")
    st.progress(0.5)

# ========== CHAT PANEL ==========
if st.session_state.chat_panel_open:
    # Chat panel controls
    col1, col2, col3 = st.columns([1, 1, 8])
    
    with col1:
        if st.button("🔻 Minimize", key="minimize_chat"):
            st.session_state.chat_panel_minimized = not st.session_state.chat_panel_minimized
            st.rerun()
    
    with col2:
        if st.button("✕ Close", key="close_chat"):
            st.session_state.chat_panel_open = False
            st.session_state.chat_panel_minimized = False
            st.rerun()
    
    # Only show chat content if not minimized
    if not st.session_state.chat_panel_minimized:
        # Chat container with custom styling
        chat_container = st.container()
        with chat_container:
            st.markdown("""
            <div style="
                background: white;
                border-radius: 15px;
                box-shadow: 0 -5px 20px rgba(0,0,0,0.15);
                border: 2px solid #e9ecef;
                overflow: hidden;
                margin-top: 1rem;
            ">
                <div style="
                    background: linear-gradient(135deg, #87CEFA, #B0E0E6);
                    color: white;
                    padding: 15px 20px;
                    font-weight: 600;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                ">
                    🤖 AI Tax Chat Assistant
                    <span style="
                        background: rgba(30,144,255,0.2);
                        color: #1E90FF;
                        padding: 4px 10px;
                        border-radius: 12px;
                        font-size: 12px;
                        margin-left: 10px;
                        font-weight: 600;
                    ">
                        Active
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Chat messages display
            if st.session_state.chat_history:
                st.markdown('<div style="max-height: 300px; overflow-y: auto; padding: 15px; background: #fafafa; border-left: 2px solid #e9ecef; border-right: 2px solid #e9ecef;">', unsafe_allow_html=True)
                
                for msg in st.session_state.chat_history[-6:]:  # Show last 6 messages
                    if msg['role'] == 'user':
                        st.markdown(f"""
                        <div style="
                            margin: 10px 40px 10px 0;
                            padding: 12px 16px;
                            background: linear-gradient(135deg, #e3f2fd, #bbdefb);
                            border-left: 4px solid #2196f3;
                            border-radius: 12px;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                        ">
                            <strong>You:</strong> {msg['content']}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="
                            margin: 10px 0 10px 40px;
                            padding: 12px 16px;
                            background: linear-gradient(135deg, #f1f8e9, #dcedc8);
                            border-left: 4px solid #4caf50;
                            border-radius: 12px;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                        ">
                            <strong>TaxBot:</strong> {msg['content']}
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Chat input form
            with st.form("chat_form", clear_on_submit=True):
                st.markdown('<div style="padding: 15px; background: white; border-left: 2px solid #e9ecef; border-right: 2px solid #e9ecef; border-bottom: 2px solid #e9ecef; border-radius: 0 0 15px 15px;">', unsafe_allow_html=True)
                
                col1, col2 = st.columns([5, 1])
                with col1:
                    user_input = st.text_input(
                        "Ask about your tax data:",
                        placeholder="e.g., What's my effective tax rate? How much did I pay in taxes?",
                        label_visibility="collapsed",
                        key="chat_input"
                    )
                with col2:
                    submitted = st.form_submit_button("Send", use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                if submitted and user_input.strip():
                    # Add user message
                    st.session_state.chat_history.append({
                        "role": "user", 
                        "content": user_input
                    })
                    
                    # Get AI response
                    if st.session_state.chat_engine:
                        with st.spinner("TaxBot is thinking..."):
                            try:
                                response = st.session_state.chat_engine.chat(user_input)
                                st.session_state.chat_history.append({
                                    "role": "assistant", 
                                    "content": str(response)
                                })
                            except Exception as e:
                                st.session_state.chat_history.append({
                                    "role": "assistant", 
                                    "content": f"I apologize, but I encountered an error: {str(e)}"
                                })
                    else:
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "content": "Chat engine is not available. Please try uploading a document first."
                        })                 
                    st.rerun()

# ========== NAVIGATION CONTENT ==========
if menu_option == "📊 Overview":
    if st.session_state.extracted_data:
        extracted_json = st.session_state.extracted_data
        print(extracted_json)
        # Key metrics
        st.markdown("### 📊 Tax Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        income_data = extracted_json.get('income_section', {})
        tax_data = extracted_json.get('tax_section', {})
        payment_data = extracted_json.get('payment_section', {})
        
        with col1:
            st.metric("Total Income", f"${income_data.get('total_income', 0):,}")
        with col2:
            st.metric("Total Tax", f"${tax_data.get('total_tax', 0):,}")
        with col3:
            effective_rate = (tax_data.get('total_tax', 0) / income_data.get('total_income', 1)) * 100 if income_data.get('total_income', 0) > 0 else 0
            st.metric("Effective Rate", f"{effective_rate:.1f}%")
        with col4:
            refund = payment_data.get('refund', 0)
            owed = payment_data.get('amount_owed', 0)
            if refund > 0:
                st.metric("Refund", f"${refund:,}", delta="Refund")
            elif owed > 0:
                st.metric("Amount Owed", f"${owed:,}", delta="Owed")
            else:
                st.metric("Balance", "$0", delta="Even")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Income vs Tax comparison
            fig = go.Figure()
            categories = ['2024']
            
            fig.add_trace(go.Bar(
                name='Income',
                x=categories,
                y=[income_data.get('total_income', 0)],
                marker_color='#87CEFA',  # Light Sky Blue
                text=[f'${income_data.get("total_income", 0)/1000:.0f}K'],
                textposition='auto',
            ))
            
            fig.add_trace(go.Bar(
                name='Tax',
                x=categories,
                y=[tax_data.get('total_tax', 0)],
                marker_color='#AFEEEE',   # Red for tax remains the same
                text=[f'${tax_data.get("total_tax", 0)/1000:.0f}K'],
                textposition='auto',
            ))
            
            fig.update_layout(
                title="Income vs Tax Comparison",
                barmode='group',
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Income breakdown
            income_breakdown = {
                'Wages': income_data.get('wages', 0),
                'Interest': income_data.get('taxable_interest', 0),
                'Dividends': income_data.get('qualified_dividends', 0),
                'Capital Gains': income_data.get('capital_gains_or_loss', 0)
            }
            
            # Filter out zero values
            income_breakdown = {k: v for k, v in income_breakdown.items() if v > 0}
            
            if income_breakdown:
                fig_pie = go.Figure(data=[go.Pie(
                    labels=list(income_breakdown.keys()),
                    values=list(income_breakdown.values()),
                    hole=0.4,
                    marker_colors = ['#1E90FF', '#00BFFF', '#87CEFA', '#B0E0E6']
                )])
                
                fig_pie.update_layout(
                    title="Income Sources",
                    height=400
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
        
        # Detailed sections
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        with st.expander("👤 Taxpayer Information", expanded=True):
            basic_info = extracted_json.get('basic_info', {})
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Name:** {basic_info.get('taxpayer_name', 'N/A')}")
                st.write(f"**SSN:** {basic_info.get('ssn', 'N/A')}")
                st.write(f"**Filing Status:** {basic_info.get('filing_status', 'N/A')}")
            
            with col2:
                st.write(f"**Spouse:** {basic_info.get('spouse_name', 'N/A')}")
                st.write(f"**Address:** {basic_info.get('address', 'N/A')}")
                st.write(f"**City, State ZIP:** {basic_info.get('city', 'N/A')}, {basic_info.get('state', 'N/A')} {basic_info.get('zip_code', 'N/A')}")
        
        with st.expander("💰 Income Details"):
            income_df = pd.DataFrame([
                {"Description": "Wages, Salaries, Tips", "Amount": f"${income_data.get('wages', 0):,}"},
                {"Description": "Taxable Interest", "Amount": f"${income_data.get('taxable_interest', 0):,}"},
                {"Description": "Qualified Dividends", "Amount": f"${income_data.get('qualified_dividends', 0):,}"},
                {"Description": "Capital Gains/Losses", "Amount": f"${income_data.get('capital_gains_or_loss', 0):,}"},
                {"Description": "Total Income", "Amount": f"${income_data.get('total_income', 0):,}"},
                {"Description": "Adjusted Gross Income", "Amount": f"${income_data.get('adjusted_gross_income', 0):,}"},
            ])
            st.dataframe(income_df, hide_index=True, use_container_width=True)
        
        with st.expander("🧾 Tax & Payments"):
            tax_payment_df = pd.DataFrame([
                {"Description": "Federal Income Tax", "Amount": f"${tax_data.get('income_tax', 0):,}"},
                {"Description": "Total Tax Liability", "Amount": f"${tax_data.get('total_tax', 0):,}"},
                {"Description": "Federal Tax Withheld", "Amount": f"${payment_data.get('federal_total_withholding', 0):,}"},
                {"Description": "Total Payments", "Amount": f"${payment_data.get('total_payments', 0):,}"},
                {"Description": "Refund/Amount Owed", "Amount": f"${payment_data.get('refund', 0) - payment_data.get('amount_owed', 0):,}"},
            ])
            st.dataframe(tax_payment_df, hide_index=True, use_container_width=True)


        # with st.expander("📋 Schedule Information"):
        #     tab1, tab2, tab3 = st.tabs(["Schedule A", "Schedule D", "Schedule 1"])
            
        #     with tab1:
        #         schedule_a = extracted_json.get('schedule_a', {})
        #         if schedule_a:
        #             schedule_a_df = pd.DataFrame([
        #                 {"Deduction Type": "Medical & Dental", "Amount": f"${schedule_a.get('medical_dental', 0):,}"},
        #                 {"Deduction Type": "State & Local Taxes", "Amount": f"${schedule_a.get('state_local_income_tax', 0):,}"},
        #                 {"Deduction Type": "Real Estate Tax", "Amount": f"${schedule_a.get('real_estate_tax', 0):,}"},
        #                 {"Deduction Type": "Mortgage Interest", "Amount": f"${schedule_a.get('mortgage_interest', 0):,}"},
        #                 {"Deduction Type": "Total Itemized", "Amount": f"${schedule_a.get('total_itemized', 0):,}"},
        #             ])
        #             st.dataframe(schedule_a_df, hide_index=True, use_container_width=True)
        #         else:
        #             st.info("No Schedule A data available")

        #     with tab2:
        #         schedule_d = extracted_json.get('schedule_d', {})
        #         if schedule_d:
        #             schedule_d_df = pd.DataFrame([
        #                 {"Investment Type": "Short-term Gains/Losses", "Amount": f"${schedule_d.get('short_term_gain_loss', 0):,}"},
        #                 {"Investment Type": "Long-term Gains/Losses", "Amount": f"${schedule_d.get('long_term_gain_loss', 0):,}"},
        #                 {"Investment Type": "Total Investment Gain/Loss", "Amount": f"${schedule_d.get('total_investmen_gain_loss', 0):,}"},
        #             ])
        #             st.dataframe(schedule_d_df, hide_index=True, use_container_width=True)
        #         else:
        #             st.info("No Schedule D data available")
            
        #     with tab3:
        #         schedule_1 = extracted_json.get('schedule_1', {})
        #         if schedule_1:
        #             schedule_1_df = pd.DataFrame([
        #                 {"Item": "Additional Income", "Amount": f"${schedule_1.get('additional_income', 0):,}"},
        #                 {"Item": "Adjustments to Income", "Amount": f"${schedule_1.get('adjustments_to_income', 0):,}"},
        #             ])
        #             st.dataframe(schedule_1_df, hide_index=True, use_container_width=True)
        #         else:
        #             st.info("No Schedule 1 data available")

        with st.expander("📋 Schedules & Forms Summary"):
            # Filter out non-schedules/forms (like 'basic_info', 'income_section', etc.)
            schedule_data = {
                key: value
                for key, value in extracted_json.items()
                if key.startswith("schedule_") or key.startswith("form_")
            }

            if schedule_data:
                tab_objs = st.tabs([key.replace("_", " ").title() for key in schedule_data.keys()])

                for tab, (key, section_data) in zip(tab_objs, schedule_data.items()):
                    with tab:
                        if isinstance(section_data, dict) and section_data:
                            df = pd.DataFrame([
                                {
                                    "Field": field.replace("_", " ").title(),
                                    "Amount": f"${value:,}" if isinstance(value, (int, float)) else (value if value is not None else "—")
                                }
                                for field, value in section_data.items()
                            ])
                            st.dataframe(df, hide_index=True, use_container_width=True)
                        else:
                            st.info(f"No data available for {key.replace('_', ' ').title()}.")
            else:
                st.warning("No schedule or form data available.")
elif menu_option == "📄 Documents":
    st.markdown("### 📄 Document Management")
    
    if st.session_state.extracted_data:
        st.success("✅ Document processed successfully!")
        
        # Document info
        basic_info = st.session_state.extracted_data.get('basic_info', {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Document Information:**")
            st.write(f"• **Taxpayer:** {basic_info.get('taxpayer_name', 'N/A')}")
            st.write(f"• **Tax Year:** {st.session_state.extracted_data.get('tax_year', '2024')}")
            st.write(f"• **File:** {st.session_state.uploaded_filename or 'N/A'}")
        
        with col2:
            st.markdown("**Processing Details:**")
            st.write(f"• **Date:** {time.strftime('%Y-%m-%d')}")
            st.write(f"• **Time:** {time.strftime('%H:%M:%S')}")
            st.write(f"• **Status:** Complete ✅")
        
        # Document stats
        st.markdown("---")
        st.markdown("### 📊 Document Statistics")
        
        income_data = st.session_state.extracted_data.get('income_section', {})
        tax_data = st.session_state.extracted_data.get('tax_section', {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Income", f"${income_data.get('total_income', 0):,}")
        with col2:
            st.metric("Total Tax", f"${tax_data.get('total_tax',0):,}")
        with col3:
            effective_rate = (tax_data.get('total_tax', 0) / income_data.get('total_income', 1)) * 100 if income_data.get('total_income', 0) > 0 else 0
            st.metric("Effective Tax Rate", f"{effective_rate:.1f}%")
            
    else:
        st.info("📤 No documents processed yet. Please upload a tax document in the Overview section.")

elif menu_option == "📈 Analysis":
    st.title("Tax Analysis")
    if st.session_state.extracted_data:
        # st.success("✅ Tax data available for analysis!")
        
        # Advanced analysis with charts
        income_data = st.session_state.extracted_data.get('income_section', {})
        tax_data = st.session_state.extracted_data.get('tax_section', {})
        payment_data = st.session_state.extracted_data.get('payment_section', {})
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Income", f"${income_data.get('total_income', 0):,}")
        
        with col2:
            st.metric("Total Tax", f"${tax_data.get('total_tax', 0):,}")
        
        with col3:
            effective_rate = (tax_data.get('total_tax', 0) / income_data.get('total_income', 1)) * 100 if income_data.get('total_income', 0) > 0 else 0
            st.metric("Effective Tax Rate", f"{effective_rate:.1f}%")
            
        with col4:
            # Calculate marginal tax rate based on 2024 tax brackets (single filer)
            total_income = income_data.get('total_income', 0)
            if total_income <= 11000:
                marginal_rate = 10.0
            elif total_income <= 44725:
                marginal_rate = 12.0
            elif total_income <= 95375:
                marginal_rate = 22.0
            elif total_income <= 182050:
                marginal_rate = 24.0
            elif total_income <= 231250:
                marginal_rate = 32.0
            elif total_income <= 578125:
                marginal_rate = 35.0
            else:
                marginal_rate = 37.0
            
            st.metric("Marginal Tax Rate", f"{marginal_rate:.1f}%")
        
        st.markdown("---")
        
        # Income breakdown chart
        st.subheader("Income Breakdown")
        income_breakdown = {
            'Wages': income_data.get('wages', 0),
            'Interest': income_data.get('taxable_interest', 0),
            'Dividends': income_data.get('qualified_dividends', 0),
            'Capital Gains': income_data.get('capital_gains_or_loss', 0)
        }
        
        # Filter out zero values
        income_breakdown = {k: v for k, v in income_breakdown.items() if v > 0}
        
        if income_breakdown:
            fig_income = go.Figure(data=[go.Pie(
                labels=list(income_breakdown.keys()),
                values=list(income_breakdown.values()),
                hole=0.3
            )])
            fig_income.update_layout(title="Income Sources", height=400)
            st.plotly_chart(fig_income, use_container_width=True)
        
        # Tax efficiency analysis
        st.subheader("Tax Efficiency Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Tax Burden Analysis:**")
            total_income = income_data.get('total_income', 0)
            total_tax = tax_data.get('total_tax', 0)
            
            if total_income > 0:
                tax_rate = (total_tax / total_income) * 100
                if tax_rate < 15:
                    st.success(f"✅ Your effective tax rate ({tax_rate:.1f}%) is relatively low")
                elif tax_rate < 25:
                    st.info(f"ℹ️ Your effective tax rate ({tax_rate:.1f}%) is moderate")
                else:
                    st.warning(f"⚠️ Your effective tax rate ({tax_rate:.1f}%) is high")
        
        with col2:
            st.write("**Deduction Optimization:**")
            deductions = income_data.get('total_deductions', 0)
            standard_deduction = 27700  # 2024 married filing jointly
            
            if deductions > standard_deduction:
                st.success(f"✅ Itemizing saves you ${(deductions - standard_deduction):,}")
            else:
                st.info("ℹ️ Standard deduction is optimal for you")
        
    else:
        st.info("📤 No tax data available for analysis. Please upload a tax document in the Overview section.")

elif menu_option == "💡 Insights":
    st.title("Tax Insights & Recommendations")
    if st.session_state.extracted_data:
        st.success("✅ Tax data available for insights!")
        
        # Generate insights based on the data
        income_data = st.session_state.extracted_data.get('income_section', {})
        tax_data = st.session_state.extracted_data.get('tax_section', {})
        payment_data = st.session_state.extracted_data.get('payment_section', {})
        
        st.subheader("Key Insights")
        
        # Insight 1: Refund or Owe
        amount_owed = payment_data.get('amount_owed', 0)
        refund = payment_data.get('refund', 0)
        
        if amount_owed > 0:
            st.warning(f"💰 **Tax Liability:** You owe ${amount_owed:,} in taxes")
            st.write("💡 **Recommendation:** Consider increasing withholdings or making quarterly payments next year to avoid owing.")
        elif refund > 0:
            st.success(f"🎉 **Tax Refund:** You're getting a refund of ${refund:,}!")
            st.write("💡 **Recommendation:** Consider adjusting withholdings to get more money in your paycheck throughout the year.")
        else:
            st.success("✅ **Perfect Balance:** Your tax payments are balanced")
            st.write("💡 **Great job!** You've optimized your tax withholdings perfectly.")
        
        st.markdown("---")
        
        # Insight 2: Deduction analysis
        st.subheader("Deduction Insights")
        deductions = income_data.get('total_deductions', 0)
        standard_deduction = 27700  # 2024 married filing jointly (adjust based on filing status)
        
        col1, col2 = st.columns(2)
        with col1:
            if deductions > standard_deduction:
                savings = deductions - standard_deduction
                st.success(f"✅ **Itemizing Benefits:** You saved ${savings:,} by itemizing")
                st.write("💡 Keep detailed records of deductible expenses for next year.")
            else:
                potential_savings = standard_deduction - deductions
                st.info(f"ℹ️ **Standard Deduction Optimal:** You're using the standard deduction")
                st.write(f"💡 You'd need ${potential_savings:,} more in itemized deductions to benefit from itemizing.")
        
        with col2:
            # Schedule A analysis if available
            schedule_a = st.session_state.extracted_data.get('schedule_a', {})
            if schedule_a:
                mortgage_interest = schedule_a.get('mortgage_interest', 0)
                charitable = schedule_a.get('charitable_contributions', 0)
                
                st.write("**Top Deductions:**")
                if mortgage_interest > 0:
                    st.write(f"🏠 Mortgage Interest: ${mortgage_interest:,}")
                if charitable > 0:
                    st.write(f"❤️ Charitable Giving: ${charitable:,}")
        
        st.markdown("---")
        
        # Insight 3: Tax planning recommendations
        st.subheader("Tax Planning Recommendations" )

        # Load JSON recommendations from file (only once)
        if not  st.session_state.recommendation_data:
            st.warning("⚠️ Recommendations not generated yet.")

        recommendations = st.session_state.recommendation_data

        # Display “Needs Attention” section
        if recommendations.get("needs_attention"):
            st.markdown("### ⚠️ Needs Immediate Attention")
            for item in recommendations["needs_attention"]:
                with st.expander(f"❗ {item['title']}"):
                    st.markdown(f"**🔎 Insight:** {item['description']}")
                    st.markdown("**✅ Actionable Steps:**")
                    for step in item["actionable_steps"]:
                        st.markdown(f"- {step}")

        st.markdown("---")

        # Display “Opportunities” section
        if recommendations.get("opportunities"):
            st.markdown("### 🌟 Optimization Opportunities")
            for item in recommendations["opportunities"]:
                with st.expander(f"💡 {item['title']}"):
                    st.markdown(f"**🔎 Insight:** {item['description']}")
                    st.markdown("**📋 Suggested Actions:**")
                    for step in item["actionable_steps"]:
                        st.markdown(f"- {step}")
    else:
        st.info("📤 No tax data available for insights. Please upload a tax document in the Overview section.")

# ========== FOOTER ==========
if st.session_state.extracted_data:
    st.markdown("---")