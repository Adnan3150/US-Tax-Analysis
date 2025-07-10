from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from src import config
import json
import os


def generate_recommendations(json_file,session_id):
    if not isinstance(json_file, (dict, list)):
        # === Load JSON Knowledge Base ===
        json_path = r"f:\adnan\Adnan\1040_tax_analysis\aws_textract\Tax-architecture-dev\extracted_json_data\dummy_data_aishwarya.json"
        if not os.path.exists(json_path):
            raise RuntimeError("Knowledge base file not found.")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data=json_file
    json_text = json.dumps(data, indent=2)
    docs = [Document(page_content=json_text)]

    # === Split Text into Chunks ===
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # === Embedding + Vector Store ===
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedder)

    # === LLM via Groq ===
    llm = ChatGroq(
        temperature=0.3,
        model_name="llama3-8b-8192",
        groq_api_key="s"
    )

    # === Prompt Template with Description + Instructions ===
    prompt_template = PromptTemplate.from_template("""
        You are a professional tax advisor that analyzes taxpayer data retrieved from the knowledge base.

        Description:
            -"You are a professional tax advisor that analyzes taxpayer data retrieved from the knowledge base. "
            -"Use ONLY the specific taxpayer information found in the knowledge base - never create hypothetical examples. "
            -"Search the knowledge base for the taxpayer's actual tax data and financial information. "
            -"Return your analysis in valid JSON format with structured sections for tax issues and opportunities. "
            -"Always reference exact dollar amounts and tax situations from the retrieved taxpayer data."

        Instructions:
                -"CRITICAL: Search the knowledge base for the specific taxpayer's actual tax data - never use hypothetical examples",
                -"Return response as valid JSON format only - no additional text or markdown",
                -"Structure JSON with two main sections: 'needs_attention' and 'opportunities'",
                -"MUST identify AT LEAST 3-4 items for 'needs_attention' and 4-5 items for 'opportunities'",
                -"Perform comprehensive analysis of ALL tax data including: withholding vs tax owed, deduction strategies, AGI thresholds, investment activity, charitable giving, tax credits",
                -"For 'needs_attention': analyze withholding adequacy, penalty risks, missed deductions, incorrect filing strategies",
                -"For 'opportunities': analyze Roth IRA strategies, charitable giving optimization, deduction bundling, tax credit maximization, investment tax strategies",
                -"Search multiple aspects of taxpayer data: income_section, tax_section, payment_section, schedule_a, schedule_d, schedule_1",
                -"Compare standard deduction vs itemized deductions from actual data",
                -"Analyze AGI against ALL relevant thresholds (Roth IRA, tax brackets, phase-outs, SALT limits)",
                -"Examine ALL schedules for optimization opportunities",
                -"Always include specific dollar amounts from the taxpayer's actual data in descriptions",
                -"Reference actual tax forms and schedules found in the knowledge base",
        **Strictly return json format**
        JSON Output Format:
        {{
            "needs_attention": [
                {{
                    "title": "Descriptive issue title",
                    "description": "Detailed analysis with specific dollar amounts",
                    "actionable_steps": ["Step 1", "Step 2", "Step 3"]
                }}
            ],
            "opportunities": [
                {{
                    "title": "Descriptive strategy title",
                    "description": "Detailed explanation with taxpayer's data",
                    "actionable_steps": ["Step 1", "Step 2", "Step 3"]
                }}
            ]
        }}

        Context:
        {context}

        Question:
        {question}
        """)

    # === Retrieval-based Chain ===
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=False,
    )

    # === Request Model ===
    query = (
        "Analyze ALL aspects of the user's tax data comprehensively and provide optimization strategies in JSON format. "
        "Provide at least 3 'needs_attention' items and 4 'opportunities' based on actual tax details."
    )

    print("🔍 Running analysis...\n")

    response = qa_chain.run(query)

    try:
        # Try parsing response text
        parsed = json.loads(response)
        
        # Print nicely
        # print(json.dumps(parsed, indent=2, ensure_ascii=False))
        json_path=config.PATHS.get('recommendation_json_path',"")
        json_name=f'{session_id}_recommendations.json'
        json_path=os.path.join(json_path,json_name)
        # Save to file
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=4, ensure_ascii=False)
        return parsed
    except json.JSONDecodeError:
        print("⚠️ Model returned invalid JSON:\n")
        return None