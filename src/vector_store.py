import os
from llama_index.core import Settings, VectorStoreIndex, Document
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import re
import json
from src import config

def extract_sections_from_text(text:str ,save_path:str) -> dict:
    def extract_by_headers(start: str, end_pattern: str) -> str:
        end_matches = re.findall(end_pattern, text)
        if start in text and end_matches:
            end_text = end_matches[-1]
            return text[text.find(start):text.rfind(end_text) + len(end_text)]
        return ""

    def extract_match_or_empty(pattern: str) -> str:
        match = re.search(pattern, text, re.DOTALL)
        return match.group(0).strip() if match else ""

    year_pattern = r"\(Form 1040\)\s*\d{4}"

    sections = {
        "basic_info": extract_match_or_empty(r'(?s)U.S. Individual Income Tax Return.*?(?=1a\s+)'),
        "income": extract_match_or_empty(r'(?s)(?:1[az]|W-2).*?15\s+[\d,\-]+'),
        "tax": extract_match_or_empty(r'(?s)Tax and\s+16.*?24\s+[\d,\-]+'),
        "payments": extract_match_or_empty(r'(?s)Payments\s+25.*?37\s+[\d,\-]+'),
        "schedule_a": extract_by_headers("SCHEDULE A", f"Schedule A {year_pattern}"),
        "schedule_b": extract_by_headers("SCHEDULE B", f"Schedule B {year_pattern}"),
        "schedule_d": extract_by_headers("SCHEDULE D", f"Schedule D {year_pattern}"),
        "schedule_1": extract_by_headers("SCHEDULE 1", f"Schedule 1 {year_pattern}")
    }

    print("sectioned_data saving")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2, ensure_ascii=False)

    return sections

def build_vector_chat_engine_from_text(sectioned_data):
    print("initializing cha bot...!")
    # save_path = config.PATHS.get("section_save_path","")
    # save_name=f'{session_id}_sectioned_data.json'
    # save_path=os.path.join(save_path,save_name)
    # Extract and chunk sections
    # with open(raw_text, 'r') as file:
    #     sectioned_data = json.load(file)
    # sectioned_data = extract_sections_from_text(content,save_path)
    
    documents = []
    for section_name, content in sectioned_data.items():
        if content.strip():
            documents.append(Document(text=content.strip(), metadata={"section": section_name}))
    
    # Set embedding + LLM
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = Groq(model="llama-3.1-8b-instant", api_key="")  # Update if needed

    Settings.llm = llm
    Settings.embed_model = embed_model

    # Create vector index
    vector_index = VectorStoreIndex.from_documents(documents, show_progress=True)
    print("data ingestion completed")
    return vector_index
