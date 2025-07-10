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
# from llama_index.core import Settings, Document, VectorStoreIndex
# from llama_index.llms.groq import Groq
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import aws_extract_tool
import field_mapping

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

# Custom CSS for Fifteenth-style UI
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 1rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .tax-section-header {
        background: #2E8B57;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        margin: 1rem 0 0.5rem 0;
        font-weight: 600;
    }
    
    .success-message {
            background-color: #d4edda;
            color: #155724;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #c3e6cb;
            font-weight: bold;
            margin-top: 10px;
    }
    
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 8px;
        border: 2px dashed #dee2e6;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .data-table {
        background: white;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .section-divider {
        border-top: 1px solid #dee2e6;
        margin: 2rem 0;
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
        st.session_state.processing_status = "ready"  # ready, processing, complete, error

def process_uploaded_file(pdf_file):
    """Process the uploaded PDF file and extract tax data"""
    try:
        st.session_state.processing_status = "processing"
        
        # Save the PDF file locally
        pdf_path = aws_extract_tool.save_file_to_local(pdf_file)
        
        # Show progress messages
        progress_placeholder = st.empty()
        
        # AWS Textract extraction
        progress_placeholder.markdown(
            '<div class="success-message">🔄 Processing with AWS Textract...</div>',
            unsafe_allow_html=True
        )
        
        raw_text_path = r"F:\adnan\Adnan\1040_tax_analysis\aws_textract\Tax-architecture-dev\extracted_raw_data\extracted_text_2pages.txt"
        
        # Simulate textract processing (replace with actual call)
        time.sleep(1)
        
        progress_placeholder.markdown(
            '<div class="success-message">✅ AWS Textract extraction complete.</div>',
            unsafe_allow_html=True
        )
        time.sleep(1)
        
        # Field extraction
        progress_placeholder.markdown(
            '<div class="success-message">🔄 Extracting tax fields...</div>',
            unsafe_allow_html=True
        )
        # output_json=field_mapping.field_map(raw_text_path)
        # # Load dummy data (replace with actual field mapping)
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
        
        # Final success message
        progress_placeholder.markdown(
            '<div class="success-message">✅ Tax data extraction complete!</div>',
            unsafe_allow_html=True
        )
        time.sleep(2)
        progress_placeholder.empty()
        
        return True
        
    except Exception as e:
        st.session_state.processing_status = "error"
        st.error("Failed to extract fields.")
        st.exception(e)
        return False

# Initialize session state
initialize_session_state()

# ========== SIDEBAR ==========
with st.sidebar:
    st.title("🇺🇸 Tax Extractor")
    st.markdown("---")
    
    # Show current status
    if st.session_state.processing_complete:
        st.success("✅ Data processed successfully")
        if st.session_state.uploaded_filename:
            st.write(f"📄 File: {st.session_state.uploaded_filename}")
    elif st.session_state.processing_status == "processing":
        st.info("🔄 Processing in progress...")
    elif st.session_state.processing_status == "error":
        st.error("❌ Processing failed")
    else:
        st.info("📤 Ready to upload file")
    
    st.markdown("---")
    
    # Navigation menu
    menu_option = st.selectbox(
        "Navigation",
        ["📊 Overview", "📄 Documents", "📈 Analysis", "💡 Insights"],
        index=0
    )
    
    st.markdown("---")
    
    # Reset button to clear session
    if st.button("🔄 Reset Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    st.markdown("---")
    st.markdown("**Need help?**")
    st.markdown("📧 shaik.adnan@spsoft.in")

# ========== MAIN CONTENT ==========
st.title("AI-Powered IRS Form 1040 Extractor")
st.markdown("Upload your tax documents and extract key information automatically")

# ========== FILE UPLOAD SECTION (Only show if not processed) ==========
if not st.session_state.processing_complete and st.session_state.processing_status != "processing":
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### Upload your IRS Form 1040 (PDF)")
    pdf_file = st.file_uploader("", type="pdf", label_visibility="collapsed", key="pdf_uploader")
    st.markdown("Limit 200MB per file • PDF")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process file if uploaded
    if pdf_file is not None and not st.session_state.file_uploaded:
        st.session_state.file_uploaded = True
        process_uploaded_file(pdf_file)
        st.rerun()

# ========== PROCESSING STATUS ==========
elif st.session_state.processing_status == "processing":
    st.info("🔄 Processing your tax document... Please wait.")

# ========== NAVIGATION BASED CONTENT ==========
if menu_option == "📊 Overview":
    if st.session_state.extracted_data:
        extracted_json = st.session_state.extracted_data
        
        # Create two columns for charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Income vs Taxes Chart
            income_data = extracted_json.get('income_section', {})
            tax_data = extracted_json.get('tax_section', {})
            payment_data = extracted_json.get('payment_section', {})
            
            # Create bar chart similar to Fifteenth
            fig = go.Figure()
            
            categories = ['2024']
            taxes = [tax_data.get('total_tax', 0)]
            income = [income_data.get('total_income', 0)]
            
            fig.add_trace(go.Bar(
                name='Taxes',
                x=categories,
                y=taxes,
                marker_color='#404040',
                text=[f'${x/1000:.0f}K' for x in taxes],
                textposition='auto',
            ))
            
            fig.add_trace(go.Bar(
                name='Income',
                x=categories,
                y=income,
                marker_color='#90EE90',
                text=[f'${x/1000:.0f}K' for x in income],
                textposition='auto',
            ))
            
            fig.update_layout(
                title="Prior Year Tax Returns",
                barmode='group',
                height=400,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                plot_bgcolor='white',
                paper_bgcolor='black'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Tax breakdown pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Federal Tax', 'State Tax', 'Other'],
                values=[
                    tax_data.get('total_tax', 0) * 0.7,
                    tax_data.get('total_tax', 0) * 0.25,
                    tax_data.get('total_tax', 0) * 0.05
                ],
                hole=0.3,
                marker_colors=['#404040', '#90EE90', '#FFB6C1']
            )])
            
            fig_pie.update_layout(
                title="Tax Breakdown 2024",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # ========== TAX SUMMARY TABLE ==========
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Create summary table similar to Fifteenth
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Federal")
            federal_data = {
                "Category": ["Income", "Deductions", "Credits", "Taxes"],
                "Amount": [
                    f"${income_data.get('total_income', 0):,}",
                    f"${income_data.get('total_deductions', 0):,}",
                    f"${tax_data.get('total_credits', 0) or 0:,}",
                    f"${tax_data.get('total_tax', 0):,}"
                ]
            }
            df_federal = pd.DataFrame(federal_data)
            st.dataframe(df_federal, hide_index=True, use_container_width=True)
        
        # ========== DETAILED SECTIONS ==========
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Expandable sections for detailed data
        with st.expander("👤 Basic Information", expanded=True):
            basic_info = extracted_json.get('basic_info', {})
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Taxpayer:** {basic_info.get('taxpayer_name', 'N/A')}")
                st.write(f"**SSN:** {basic_info.get('ssn', 'N/A')}")
                st.write(f"**Address:** {basic_info.get('address', 'N/A')}")
            
            with col2:
                st.write(f"**Spouse:** {basic_info.get('spouse_name', 'N/A')}")
                st.write(f"**Spouse SSN:** {basic_info.get('spouse_ssn', 'N/A')}")
                st.write(f"**City, State:** {basic_info.get('city', 'N/A')}, {basic_info.get('state', 'N/A')} {basic_info.get('zip_code', 'N/A')}")
        
        with st.expander("💰 Income Section"):
            income_df = pd.DataFrame([
                {"Item": "Wages", "Amount": f"${income_data.get('wages', 0):,}"},
                {"Item": "Taxable Interest", "Amount": f"${income_data.get('taxable_interest', 0):,}"},
                {"Item": "Qualified Dividends", "Amount": f"${income_data.get('qualified_dividends', 0):,}"},
                {"Item": "Capital Gains/Loss", "Amount": f"${income_data.get('capital_gains_or_loss', 0):,}"},
                {"Item": "Total Income", "Amount": f"${income_data.get('total_income', 0):,}"},
                {"Item": "Adjusted Gross Income", "Amount": f"${income_data.get('adjusted_gross_income', 0):,}"},
            ])
            st.dataframe(income_df, hide_index=True, use_container_width=True)
        
        with st.expander("🧾 Tax & Payment Section"):
            tax_payment_df = pd.DataFrame([
                {"Item": "Income Tax", "Amount": f"${tax_data.get('income_tax', 0):,}"},
                {"Item": "Total Tax", "Amount": f"${tax_data.get('total_tax', 0):,}"},
                {"Item": "Federal Withholding", "Amount": f"${payment_data.get('federal_total_withholding', 0):,}"},
                {"Item": "Total Payments", "Amount": f"${payment_data.get('total_payments', 0):,}"},
                {"Item": "Amount Owed", "Amount": f"${payment_data.get('amount_owed', 0):,}"},
            ])
            st.dataframe(tax_payment_df, hide_index=True, use_container_width=True)
        
        with st.expander("📋 Schedule Details"):
            # Schedule tabs
            tab1, tab2, tab3 = st.tabs(["Schedule A", "Schedule D", "Schedule 1"])
            
            with tab1:
                schedule_a = extracted_json.get('schedule_a', {})
                schedule_a_df = pd.DataFrame([
                    {"Item": "Medical & Dental", "Amount": f"${schedule_a.get('medical_dental', 0):,}"},
                    {"Item": "State & Local Income Tax", "Amount": f"${schedule_a.get('state_local_income_tax', 0):,}"},
                    {"Item": "Real Estate Tax", "Amount": f"${schedule_a.get('real_estate_tax', 0):,}"},
                    {"Item": "Mortgage Interest", "Amount": f"${schedule_a.get('mortgage_interest', 0):,}"},
                    {"Item": "Total Itemized", "Amount": f"${schedule_a.get('total_itemized', 0):,}"},
                ])
                st.dataframe(schedule_a_df, hide_index=True, use_container_width=True)
            
            with tab2:
                schedule_d = extracted_json.get('schedule_d', {})
                schedule_d_df = pd.DataFrame([
                    {"Item": "Short-term Gain/Loss", "Amount": f"${schedule_d.get('short_term_gain_loss', 0):,}"},
                    {"Item": "Long-term Gain/Loss", "Amount": f"${schedule_d.get('long_term_gain_loss', 0):,}"},
                    {"Item": "Total Investment Gain/Loss", "Amount": f"${schedule_d.get('total_investmen_gain_loss', 0):,}"},
                ])
                st.dataframe(schedule_d_df, hide_index=True, use_container_width=True)
            
            with tab3:
                schedule_1 = extracted_json.get('schedule_1', {})
                schedule_1_df = pd.DataFrame([
                    {"Item": "Additional Income", "Amount": f"${schedule_1.get('additional_income', 0):,}"},
                    {"Item": "Adjustments to Income", "Amount": f"${schedule_1.get('adjustments_to_income', 0):,}"},
                ])
                st.dataframe(schedule_1_df, hide_index=True, use_container_width=True)
        
        # Raw JSON view
        with st.expander("🔍 Raw JSON Data"):
            st.json(extracted_json)
    
    elif st.session_state.processing_status == "error":
        st.error("❌ There was an error processing your tax document. Please try uploading again.")
    elif not st.session_state.processing_complete and st.session_state.processing_status == "ready":
        st.info("👆 Please upload a tax document to get started.")

elif menu_option == "📄 Documents":
    st.title("Documents")
    if st.session_state.extracted_data:
        st.success("✅ Tax data available - processed successfully!")
        basic_info = st.session_state.extracted_data.get('basic_info', {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Document for:** {basic_info.get('taxpayer_name', 'N/A')}")
            st.write(f"**Tax Year:** {st.session_state.extracted_data.get('tax_year', 'N/A')}")
            st.write(f"**File Name:** {st.session_state.uploaded_filename or 'N/A'}")
        
        with col2:
            st.write(f"**Processing Date:** {time.strftime('%Y-%m-%d %H:%M')}")
            st.write(f"**Status:** Complete")
            
        st.markdown("---")
        st.subheader("Document Summary")
        
        # Quick stats
        income_data = st.session_state.extracted_data.get('income_section', {})
        tax_data = st.session_state.extracted_data.get('tax_section', {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Income", f"${income_data.get('total_income', 0):,}")
        with col2:
            st.metric("Total Tax", f"${tax_data.get('total_tax', 0):,}")
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