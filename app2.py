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
import aws_extract_tool
import field_mapping
import vector_store

# ========== INIT ==========
nest_asyncio.apply()
load_dotenv()

# Page config with custom styling
st.set_page_config(
    page_title="AI 1040 Tax Extractor", 
    page_icon="🇺🇸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved UI
st.markdown("""
<style>
    /* Main content area */
    .main {
        padding-top: 1rem;
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
        background: linear-gradient(135deg, #2E8B57, #3CB371);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0 0.5rem 0;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
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
        border: 2px dashed #6c757d;
        text-align: center;
        margin-bottom: 2rem;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #2E8B57;
        background: linear-gradient(135deg, #f0f8f0, #e8f5e8);
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
    
    /* Chat interface improvements */
    .chat-container {
        background: white;
        border-radius: 12px;
        padding: 0;
        margin-top: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
        overflow: hidden;
    }
    
    .chat-header {
        background: linear-gradient(135deg, #2E8B57, #3CB371);
        color: white;
        padding: 1rem 1.5rem;
        font-weight: 600;
        font-size: 1.1rem;
        border-bottom: 1px solid rgba(255,255,255,0.2);
    }
    
    .chat-history {
        max-height: 400px;
        overflow-y: auto;
        padding: 1rem;
        background-color: #fafafa;
    }
    
    .chat-message {
        margin-bottom: 1rem;
        padding: 1rem;
        border-radius: 12px;
        line-height: 1.5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .user-message {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border-left: 4px solid #2196f3;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f1f8e9, #dcedc8);
        border-left: 4px solid #4caf50;
        margin-right: 2rem;
    }
    
    /* Form improvements */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e9ecef;
        padding: 0.75rem 1rem;
        font-size: 0.95rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #2E8B57;
        box-shadow: 0 0 0 3px rgba(46, 139, 87, 0.1);
    }
    
    .stButton > button {
        border-radius: 25px;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        background: linear-gradient(135deg, #2E8B57, #3CB371);
        color: white;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(46, 139, 87, 0.3);
    }
    
    /* Expander improvements */
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
    
    /* Metrics styling */
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
        
        .chat-container {
            margin: 1rem -1rem;
            border-radius: 0;
        }
    }
</style>
""", unsafe_allow_html=True)

# ========== SESSION STATE INITIALIZATION ==========
def initialize_session_state():
    """Initialize all session state variables"""
    if 'extracted_data' not in st.session_state:
        st.session_state.extracted_data = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    if 'uploaded_filename' not in st.session_state:
        st.session_state.uploaded_filename = None
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = "ready"
    
    # Chat-related session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'show_chat' not in st.session_state:
        st.session_state.show_chat = False
    if 'chat_engine' not in st.session_state:
        st.session_state.chat_engine = None
    if 'sectioned_data' not in st.session_state:
        st.session_state.sectioned_data = None

def process_uploaded_file(pdf_file):
    """Process the uploaded PDF file and extract tax data"""
    try:
        st.session_state.processing_status = "processing"
        
        # Save the PDF file locally
        pdf_path = aws_extract_tool.save_file_to_local(pdf_file)
        
        # Show progress messages
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
                
                # Load dummy data (replace with actual processing)
                path = r"F:\adnan\Adnan\1040_tax_analysis\aws_textract\Tax-architecture-dev\dummy_data.json"
                with open(path, 'r') as file:
                    output_json = json.load(file)
                
                # Ensure JSON object   
                if isinstance(output_json, str):
                    extracted_json = json.loads(output_json)
                else:
                    extracted_json = output_json
                
                # Store in session state
                st.session_state.extracted_data = extracted_json
                st.session_state.processing_complete = True
                st.session_state.processing_status = "complete"
                st.session_state.uploaded_filename = pdf_file.name
                
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

def initialize_chat_engine(raw_text_path):
    """Initialize the chat engine with extracted tax data"""
    if st.session_state.chat_engine is None:
        try:
            index = vector_store.build_vector_chat_engine_from_text(raw_text_path)
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

# Initialize session state
initialize_session_state()

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("### 🇺🇸 AI Tax Extractor")
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
st.title("🇺🇸 AI-Powered IRS Form 1040 Extractor")
st.markdown("Transform your tax documents into structured data with AI-powered extraction")

# ========== FILE UPLOAD SECTION ==========
if not st.session_state.processing_complete and st.session_state.processing_status != "processing":
    upload_container = st.container()
    with upload_container:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("### 📄 Upload Your Tax Document")
        st.markdown("Drop your IRS Form 1040 PDF here or click to browse")
        
        pdf_file = st.file_uploader(
            "Choose a PDF file", 
            type="pdf", 
            label_visibility="collapsed",
            key="pdf_uploader"
        )
        
        st.markdown("**Supported:** PDF files up to 200MB")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process file if uploaded
        if pdf_file is not None and not st.session_state.file_uploaded:
            st.session_state.file_uploaded = True
            if process_uploaded_file(pdf_file):
                st.rerun()

# ========== PROCESSING STATUS ==========
elif st.session_state.processing_status == "processing":
    st.info("🔄 Processing your tax document... This may take a few moments.")
    st.progress(0.5)

# ========== CHAT INTERFACE ==========
if st.session_state.processing_complete:
    st.markdown("---")
    
    # Chat toggle
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("💬 Chat Assistant", use_container_width=True):
            st.session_state.show_chat = not st.session_state.show_chat

    # Chat interface
    if st.session_state.show_chat:
        chat_container = st.container()
        with chat_container:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            st.markdown('<div class="chat-header">🤖 AI Tax Chat Assistant</div>', unsafe_allow_html=True)
            
            # Initialize chat engine
            raw_text_path = r"F:\adnan\Adnan\1040_tax_analysis\aws_textract\Tax-architecture-dev\extracted_raw_data\extracted_text_2pages.txt"
            if not st.session_state.chat_engine:
                with st.spinner("Initializing AI assistant..."):
                    initialize_chat_engine(raw_text_path)
            
            # Chat input
            with st.form("chat_form", clear_on_submit=True):
                col1, col2 = st.columns([4, 1])
                with col1:
                    user_input = st.text_input(
                        "Ask about your tax data:",
                        placeholder="e.g., What's my effective tax rate?",
                        label_visibility="collapsed"
                    )
                with col2:
                    submitted = st.form_submit_button("Send", use_container_width=True)
                
                if submitted and user_input.strip():
                    # Add user message
                    st.session_state.chat_history.append({
                        "role": "user", 
                        "content": user_input
                    })
                    
                    # Generate response
                    if st.session_state.chat_engine:
                        try:
                            with st.spinner("Thinking..."):
                                response = st.session_state.chat_engine.chat(user_input)
                                st.session_state.chat_history.append({
                                    "role": "assistant", 
                                    "content": str(response)
                                })
                        except Exception as e:
                            st.session_state.chat_history.append({
                                "role": "assistant", 
                                "content": f"I encountered an error: {str(e)}. Please try again."
                            })
                    
                    st.rerun()
            
            # Chat history
            if st.session_state.chat_history:
                st.markdown('<div class="chat-history">', unsafe_allow_html=True)
                for msg in st.session_state.chat_history[-8:]:  # Show last 8 messages
                    if msg['role'] == 'user':
                        st.markdown(f"""
                        <div class='chat-message user-message'>
                            <strong>You:</strong> {msg['content']}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='chat-message assistant-message'>
                            <strong>TaxBot:</strong> {msg['content']}
                        </div>
                        """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

# ========== NAVIGATION CONTENT ==========
if menu_option == "📊 Overview":
    if st.session_state.extracted_data:
        extracted_json = st.session_state.extracted_data
        
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
                marker_color='#2E8B57',
                text=[f'${income_data.get("total_income", 0)/1000:.0f}K'],
                textposition='auto',
            ))
            
            fig.add_trace(go.Bar(
                name='Tax',
                x=categories,
                y=[tax_data.get('total_tax', 0)],
                marker_color='#FF6B6B',
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
                    marker_colors=['#2E8B57', '#3CB371', '#90EE90', '#98FB98']
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
        
        with st.expander("📋 Schedule Information"):
            tab1, tab2, tab3 = st.tabs(["Schedule A", "Schedule D", "Schedule 1"])
            
            with tab1:
                schedule_a = extracted_json.get('schedule_a', {})
                if schedule_a:
                    schedule_a_df = pd.DataFrame([
                        {"Deduction Type": "Medical & Dental", "Amount": f"${schedule_a.get('medical_dental', 0):,}"},
                        {"Deduction Type": "State & Local Taxes", "Amount": f"${schedule_a.get('state_local_income_tax', 0):,}"},
                        {"Deduction Type": "Real Estate Tax", "Amount": f"${schedule_a.get('real_estate_tax', 0):,}"},
                        {"Deduction Type": "Mortgage Interest", "Amount": f"${schedule_a.get('mortgage_interest', 0):,}"},
                        {"Deduction Type": "Total Itemized", "Amount": f"${schedule_a.get('total_itemized', 0):,}"},
                    ])
                    st.dataframe(schedule_a_df, hide_index=True, use_container_width=True)
                else:
                    st.info("No Schedule A data available")
            
            with tab2:
                schedule_d = extracted_json.get('schedule_d', {})
                if schedule_d:
                    schedule_d_df = pd.DataFrame([
                        {"Investment Type": "Short-term Gains/Losses", "Amount": f"${schedule_d.get('short_term_gain_loss', 0):,}"},
                        {"Investment Type": "Long-term Gains/Losses", "Amount": f"${schedule_d.get('long_term_gain_loss', 0):,}"},
                        {"Investment Type": "Total Investment Gain/Loss", "Amount": f"${schedule_d.get('total_investmen_gain_loss', 0):,}"},
                    ])
                    st.dataframe(schedule_d_df, hide_index=True, use_container_width=True)
                else:
                    st.info("No Schedule D data available")
            
            with tab3:
                schedule_1 = extracted_json.get('schedule_1', {})
                if schedule_1:
                    schedule_1_df = pd.DataFrame([
                        {"Item": "Additional Income", "Amount": f"${schedule_1.get('additional_income', 0):,}"},
                        {"Item": "Adjustments to Income", "Amount": f"${schedule_1.get('adjustments_to_income', 0):,}"},
                    ])
                    st.dataframe(schedule_1_df, hide_index=True, use_container_width=True)
                else:
                    st.info("No Schedule 1 data available")
        
        # Raw data view
        with st.expander("🔍 Raw JSON Data"):
            st.json(extracted_json, expanded=False)
    
    else:
        st.info("👆 Upload a tax document to view the extracted data and analysis.")

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
        st.success("✅ Tax data available for analysis!")
        
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
        st.subheader("Tax Planning Recommendations")
        
        total_income = income_data.get('total_income', 0)
        effective_rate = (tax_data.get('total_tax', 0) / total_income) * 100 if total_income > 0 else 0
        
        recommendations = []
        
        # Income-based recommendations
        if total_income > 100000:
            recommendations.append("💼 Consider maximizing 401(k) contributions to reduce taxable income")
            recommendations.append("🏥 Look into HSA contributions if you have a high-deductible health plan")
        
        # Tax rate recommendations
        if effective_rate > 20:
            recommendations.append("📊 Consider tax-loss harvesting for investment accounts")
            recommendations.append("🎯 Look into tax-advantaged investment strategies")
        
        # Capital gains analysis
        capital_gains = income_data.get('capital_gains_or_loss', 0)
        if capital_gains > 0:
            recommendations.append("📈 Review your investment holding periods for better tax treatment")
        
        # Display recommendations
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        else:
            st.info("💡 Your tax situation looks optimized! Keep up the good work.")
        
        st.markdown("---")
        
        # Next year planning
        st.subheader("Next Year Planning")
        st.write("**Key Dates to Remember:**")
        st.write("📅 **April 15, 2025:** Tax filing deadline")
        st.write("📅 **January 31, 2025:** W-2 and 1099 forms available")
        st.write("📅 **Throughout 2025:** Track deductible expenses")
        
        # Action items
        st.subheader("Action Items")
        st.write("**Before Next Tax Season:**")
        st.checkbox("📁 Organize tax documents in a dedicated folder")
        st.checkbox("💳 Track business expenses and charitable donations")
        st.checkbox("📊 Review and adjust tax withholdings if needed")
        st.checkbox("💰 Consider increasing retirement contributions")
        st.checkbox("🏥 Maximize HSA contributions if applicable")
        
    else:
        st.info("📤 No tax data available for insights. Please upload a tax document in the Overview section.")

# ========== FOOTER ==========
if st.session_state.extracted_data:
    st.markdown("---")
    st.markdown("*💡 This analysis is for informational purposes only. Consult a tax professional for personalized advice.*")