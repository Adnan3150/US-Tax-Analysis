from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from phi.agent import Agent, RunResponse
from phi.model.groq import Groq
from phi.knowledge.json import JSONKnowledgeBase
from phi.vectordb.chroma import ChromaDb
from phi.embedder.sentence_transformer import SentenceTransformerEmbedder
import json
import os
 
app = FastAPI()
 
# Load Embedder and Knowledge Base
embedder = SentenceTransformerEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
 
knowledge_base = JSONKnowledgeBase(
    path=r"C:\Users\SPSOFT\Documents\Arshad\Agents\whole.json",
    vector_db=ChromaDb(collection="recommendations_1", embedder=embedder)
)
 
# Remove OpenAI API Key if set
if "OPENAI_API_KEY" in os.environ:
    del os.environ["OPENAI_API_KEY"]
 
# Agent Definition
agent = Agent(
    model=Groq(
        id="llama-3.1-8b-instant",
        max_tokens=2000,
        temperature=0.3,
        api_key="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"
    ),
    description=(
        "You are a professional tax advisor that analyzes taxpayer data retrieved from the knowledge base. "
        "Use ONLY the specific taxpayer information found in the knowledge base - never create hypothetical examples. "
        "Search the knowledge base for the taxpayer's actual tax data and financial information. "
        "Return your analysis in valid JSON format with structured sections for tax issues and opportunities. "
        "Always reference exact dollar amounts and tax situations from the retrieved taxpayer data."
    ),
    instructions=[
        "CRITICAL: Search the knowledge base for the specific taxpayer's actual tax data - never use hypothetical examples",
        "Return response as valid JSON format only - no additional text or markdown",
        "Structure JSON with two main sections: 'needs_attention' and 'opportunities'",
        "MUST identify AT LEAST 3-4 items for 'needs_attention' and 4-5 items for 'opportunities'",
        "Perform comprehensive analysis of ALL tax data including: withholding vs tax owed, deduction strategies, AGI thresholds, investment activity, charitable giving, tax credits",
        "For 'needs_attention': analyze withholding adequacy, penalty risks, missed deductions, incorrect filing strategies",
        "For 'opportunities': analyze Roth IRA strategies, charitable giving optimization, deduction bundling, tax credit maximization, investment tax strategies",
        "Search multiple aspects of taxpayer data: income_section, tax_section, payment_section, schedule_a, schedule_d, schedule_1",
        "Compare standard deduction vs itemized deductions from actual data",
        "Analyze AGI against ALL relevant thresholds (Roth IRA, tax brackets, phase-outs, SALT limits)",
        "Examine ALL schedules for optimization opportunities",
        "Always include specific dollar amounts from the taxpayer's actual data in descriptions",
        "Reference actual tax forms and schedules found in the knowledge base",
        "Strictly return the output in JSON Format only",
        "Use this exact JSON structure:",
        '''
            "needs_attention": [
                {
                    "title": "Descriptive issue title",
                    "description": "Detailed analysis with specific dollar amounts",
                    "actionable_steps": ["Step 1", "Step 2", "Step 3"],
                }
            ],
            "opportunities": [
                {
                    "title": "Descriptive strategy title", 
                    "description": "Detailed explanation with taxpayer's data",
                    "actionable_steps": ["Step 1", "Step 2", "Step 3"],
                }
            ]
        }'''
    ],
    markdown=False,
    knowledge=knowledge_base,
    search_knowledge=True,
    prevent_hallucinations=True,
)
 
# Load or reload the knowledge base
agent.knowledge.load(recreate=True)
 
# Request model (optional, in case you want to take user prompt later)
class TaxAnalysisRequest(BaseModel):
    prompt: str = (
        "Analyze ALL aspects of the user's tax data comprehensively and provide optimization strategies in JSON format. "
        "Review these specific areas: Provide at least 2-3 'needs_attention' items and 3-4 'opportunities' based on the taxpayer's actual data."
    )
 
@app.post("/analyze-tax/")
async def analyze_tax(request: TaxAnalysisRequest):
    try:
        run: RunResponse = agent.run(request.prompt)
        response_json = json.loads(run.content)
        return response_json
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON response from the agent.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))