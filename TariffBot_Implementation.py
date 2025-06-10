import os
import re
import sqlite3
import pandas as pd
from typing import List, Dict, Any, Optional
from fpdf import FPDF
from datetime import datetime
import PyPDF2
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import logging
import streamlit as st

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    DB_NAME = r"C:\Users\sshas\Downloads\HTSAgent\data\hts_data.db"
    VECTOR_STORE_PATH = r"C:\Users\sshas\Downloads\HTSAgent\faiss_index"
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    LLM_MODEL = "google/flan-t5-large"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    CSV_PATH = r"C:\Users\sshas\Downloads\HTSAgent\data\hts_2025_basic_edition_csv.csv"
    PDF_PATH = r"C:\Users\sshas\Downloads\HTSAgent\data\General Notes.pdf"
    ALL_CSV_PATHS = [
        r"C:\Users\sshas\Downloads\HTSAgent\data\htsdata.csv",
        r"C:\Users\sshas\Downloads\HTSAgent\data\htsdata (1).csv"
        r"C:\Users\sshas\Downloads\HTSAgent\data\htsdata (2).csv",
        r"C:\Users\sshas\Downloads\HTSAgent\data\htsdata (3).csv",
        r"C:\Users\sshas\Downloads\HTSAgent\data\htsdata (4).csv",
    ]
    ENABLE_MULTI_SECTION = True

class HTSValidator:
    @staticmethod
    def validate_hts_code(code: str) -> bool:
        """Validate HTS code format"""
        if not code:
            return False
        code = code.replace(" ", "").upper()
        pattern = r'^\d{4}\.\d{2}\.\d{2}\.\d{2}$'
        return bool(re.match(pattern, code))
    
    @staticmethod
    def normalize_hts_code(code: str) -> str:
        """Normalize HTS code format"""
        if not code:
            return ""
        return code.replace(" ", "").upper().strip()
    
    @staticmethod
    def suggest_corrections(code: str, all_codes: List[str]) -> List[str]:
        """Suggest close HTS code matches"""
        if not code or not all_codes:
            return []
        code = HTSValidator.normalize_hts_code(code)
        suggestions = []
        for existing_code in all_codes[:100]:
            if existing_code.startswith(code[:4]):
                suggestions.append(existing_code)
            if len(suggestions) >= 5:
                break
        return suggestions

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(
    model_name=Config.EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu', 'trust_remote_code': True}
)

# Initialize LLM
model_name = Config.LLM_MODEL
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    device_map=None,
    torch_dtype=torch.float32,
    trust_remote_code=True
)
hf_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=1024,
    device=-1
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Custom Prompt Template
CUSTOM_PROMPT_TEMPLATE = """
You are TariffBot — an intelligent assistant trained on U.S. International Trade Commission data. 
You exist to help importers, analysts, and trade professionals quickly understand tariff rules, duty rates, and policy agreements.
You always provide clear, compliant, and factual answers grounded in official HTS documentation.

When given an HTS code and product information, explain all applicable duties and cost components thoroughly.

When asked about trade agreements (e.g., NAFTA, Israel FTA), reference the relevant General Notes with citations.

If a query is ambiguous or unsupported, politely defer or recommend reviewing the relevant HTS section manually.

Do not speculate or make policy interpretations — clarify with precision and data.

Always provide at least 2 relevant sentences in your response.

Context: {context}
Question: {question}
Helpful Answer:
"""

# Data Ingestion
def create_database_schema():
    logger.info(f"Creating database schema at {Config.DB_NAME}")
    os.makedirs(os.path.dirname(Config.DB_NAME), exist_ok=True)
    try:
        conn = sqlite3.connect(Config.DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS hts_codes (
            hts_code TEXT PRIMARY KEY,
            indent INTEGER,
            description TEXT,
            unit_of_quantity TEXT,
            general_rate_of_duty TEXT,
            special_rate_of_duty TEXT,
            column_2_rate_of_duty TEXT,
            quota_quantity TEXT,
            additional_duties TEXT,
            section TEXT,
            chapter TEXT,
            section_name TEXT,
            is_heading INTEGER DEFAULT 0,
            parent_heading TEXT,
            statistical_suffix TEXT
        )
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS country_codes (
            code TEXT PRIMARY KEY,
            name TEXT,
            region TEXT,
            is_fta_member INTEGER,
            fta_name TEXT,
            special_program TEXT,
            notes TEXT
        )
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS citations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            source_document TEXT,
            page_number INTEGER,
            section_reference TEXT,
            citation_text TEXT,
            timestamp TEXT
        )
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS query_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            query_type TEXT,
            query_text TEXT,
            response_text TEXT
        )
        """)
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to create database schema: {e}")
        raise
    finally:
        conn.close()

def load_country_codes():
    logger.info("Loading enhanced country codes")
    countries = [
        ("US", "United States", "North America", 0, "", "", ""),
        ("CA", "Canada", "North America", 1, "USMCA", "GSP", "NAFTA successor"),
        ("MX", "Mexico", "North America", 1, "USMCA", "GSP", "NAFTA successor"),
        ("IL", "Israel", "Middle East", 1, "USFTA", "", "United States-Israel FTA"),
        ("AU", "Australia", "Oceania", 1, "AUSFTA", "", "Australia-US FTA"),
        ("GB", "United Kingdom", "Europe", 0, "", "", "Post-Brexit status"),
        ("CN", "China", "Asia", 0, "", "", "MFN status"),
        ("JP", "Japan", "Asia", 1, "USJTA", "", "US-Japan Trade Agreement"),
        ("KR", "South Korea", "Asia", 1, "KORUS", "", "Korea-US FTA"),
        ("SG", "Singapore", "Asia", 1, "USSFTA", "GSP", "US-Singapore FTA"),
        ("JO", "Jordan", "Middle East", 1, "USJFTA", "", "US-Jordan FTA"),
        ("CL", "Chile", "South America", 1, "USCFTA", "GSP", "US-Chile FTA"),
        ("PE", "Peru", "South America", 1, "USPFTA", "GSP", "US-Peru FTA"),
        ("CO", "Colombia", "South America", 1, "USCTPA", "GSP", "US-Colombia TPA"),
        ("PA", "Panama", "Central America", 1, "USPTPA", "GSP", "US-Panama TPA"),
    ]
    try:
        conn = sqlite3.connect(Config.DB_NAME)
        cursor = conn.cursor()
        cursor.executemany("INSERT OR REPLACE INTO country_codes VALUES (?, ?, ?, ?, ?, ?, ?)", countries)
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to load country codes: {e}")
        raise
    finally:
        conn.close()

def ingest_hts_csv(csv_path: str):
    logger.info(f"Ingesting HTS CSV from {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        df = df.where(pd.notnull(df), None)
        expected_headers = [
            'HTS Number', 'Indent', 'Description', 'Unit of Quantity',
            'General Rate of Duty', 'Special Rate of Duty', 'Column 2 Rate of Duty',
            'Quota Quantity', 'Additional Duties'
        ]
        if not all(h in df.columns for h in expected_headers):
            missing = [h for h in expected_headers if h not in df.columns]
            raise ValueError(f"Missing CSV headers: {missing}")
        df = df.rename(columns={
            'HTS Number': 'hts_code',
            'Indent': 'indent',
            'Description': 'description',
            'Unit of Quantity': 'unit_of_quantity',
            'General Rate of Duty': 'general_rate_of_duty',
            'Special Rate of Duty': 'special_rate_of_duty',
            'Column 2 Rate of Duty': 'column_2_rate_of_duty',
            'Quota Quantity': 'quota_quantity',
            'Additional Duties': 'additional_duties'
        })
        df['section'] = df['hts_code'].str.split('.').str[0].str[:2]
        df['chapter'] = df['hts_code'].str.split('.').str[0]
        conn = sqlite3.connect(Config.DB_NAME)
        df.to_sql('hts_codes', conn, if_exists='append', index=False)
        logger.info("HTS CSV ingested successfully")
    except Exception as e:
        logger.error(f"Failed to ingest HTS CSV: {e}")
        raise
    finally:
        conn.close()

def ingest_all_csvs():
    logger.info("Ingesting all CSV files")
    if Config.ENABLE_MULTI_SECTION:
        for csv_path in Config.ALL_CSV_PATHS:
            if os.path.exists(csv_path):
                ingest_hts_csv(csv_path)
            else:
                logger.warning(f"CSV file not found: {csv_path}")
    else:
        if os.path.exists(Config.CSV_PATH):
            ingest_hts_csv(Config.CSV_PATH)
        else:
            raise FileNotFoundError(f"HTS CSV file not found at {Config.CSV_PATH}")

def ingest_pdf_documents(pdf_path: str):
    logger.info(f"Ingesting PDF from {pdf_path}")
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text
        if not text:
            raise ValueError("No text extracted from PDF. Ensure the PDF contains readable text.")
        chunks = text_splitter.split_text(text)
        vectorstore = FAISS.from_texts(chunks, embeddings)
        vectorstore.save_local(Config.VECTOR_STORE_PATH)
        logger.info("PDF documents processed successfully")
    except Exception as e:
        logger.error(f"Failed to ingest PDF: {e}")
        raise

# Initialize database and vector store
def initialize_system(force_reingest: bool = False):
    logger.info("Initializing system")
    try:
        create_database_schema()
        load_country_codes()
        ingest_all_csvs()
        vector_store_path = os.path.normpath(Config.VECTOR_STORE_PATH)
        faiss_index_path = os.path.normpath(os.path.join(Config.VECTOR_STORE_PATH, "index.faiss"))
        if not force_reingest and os.path.exists(vector_store_path) and os.path.exists(faiss_index_path):
            try:
                with open(faiss_index_path, 'rb') as f:
                    pass
                logger.info(f"FAISS index found and accessible at {vector_store_path}")
            except (IOError, PermissionError) as e:
                logger.error(f"FAISS index exists but is not accessible: {e}")
                raise RuntimeError(f"Cannot access FAISS index: {str(e)}")
        else:
            logger.info(f"FAISS index not found or force re-ingestion requested")
            if os.path.exists(Config.PDF_PATH):
                ingest_pdf_documents(Config.PDF_PATH)
            else:
                raise FileNotFoundError(f"PDF document not found at {Config.PDF_PATH}")
        return True, "Initialization successful."
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        return False, f"Initialization failed: {str(e)}"

# Tools
def parse_duty_advanced(duty_str: str, unit_weight: float = None, 
                       quantity: int = None, cif_value: float = None) -> Dict[str, Any]:
    """Advanced duty parser handling all formats"""
    if not duty_str or pd.isna(duty_str) or str(duty_str).strip().lower() == "free":
        return {"rate": 0.0, "amount": 0.0, "type": "free", "original": duty_str}
    
    duty_str = str(duty_str).strip().lower()
    result = {"rate": 0.0, "amount": 0.0, "type": "unknown", "original": duty_str}
    
    if "free" in duty_str:
        result["type"] = "conditional_free"
        return result
    
    match = re.search(r"([\d.]+)\s*%", duty_str)
    if match:
        rate = float(match.group(1)) / 100
        result.update({
            "rate": rate,
            "amount": rate * cif_value if cif_value else 0,
            "type": "percentage"
        })
        return result
    
    match = re.search(r"([\d.]+)\s*¢/kg", duty_str)
    if match and unit_weight and quantity:
        cents_per_kg = float(match.group(1))
        total_amount = (cents_per_kg * unit_weight * quantity) / 100
        result.update({
            "rate": total_amount / cif_value if cif_value else 0,
            "amount": total_amount,
            "type": "weight_based"
        })
        return result
    
    match = re.search(r"\$([\d.]+)(?:/unit|/each|\s+each)", duty_str)
    if match and quantity:
        dollars_per_unit = float(match.group(1))
        total_amount = dollars_per_unit * quantity
        result.update({
            "rate": total_amount / cif_value if cif_value else 0,
            "amount": total_amount,
            "type": "unit_based"
        })
        return result
    
    if "+" in duty_str:
        parts = duty_str.split("+")
        total_amount = 0
        for part in parts:
            sub_result = parse_duty_advanced(part.strip(), unit_weight, quantity, cif_value)
            total_amount += sub_result["amount"]
        result.update({
            "amount": total_amount,
            "rate": total_amount / cif_value if cif_value else 0,
            "type": "compound"
        })
        return result
    
    return result

class QueryRouter:
    @staticmethod
    def classify_query(query: str) -> str:
        """Classify query type for routing"""
        query_lower = query.lower()
        if re.search(r'\d{4}\.\d{2}\.\d{2}\.\d{2}', query):
            if any(word in query_lower for word in ['calculate', 'duty', 'duties', 'cost']):
                return "duty_calculation"
            else:
                return "hts_lookup"
        if any(word in query_lower for word in ['calculate', 'duty', 'duties', 'cost', 'freight', 'insurance', '$', 'usd']):
            return "duty_calculation"
        if any(phrase in query_lower for phrase in ['hts code for', 'tariff code for', 'classification for']):
            return "product_search"
        if any(word in query_lower for word in ['agreement', 'fta', 'nafta', 'usmca', 'israel', 'policy', 'gsp', 'quota']):
            return "trade_policy"
        if any(word in query_lower for word in ['find', 'search', 'look for']):
            return "description_search"
        return "trade_policy"

class HTSDatabase:
    @staticmethod
    def lookup_hts_code(hts_code: str) -> Dict[str, Any]:
        try:
            conn = sqlite3.connect(Config.DB_NAME)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM hts_codes WHERE hts_code = ?", (hts_code,))
            result = cursor.fetchone()
            if not result:
                return {"error": "HTS code not found"}
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, result))
        except Exception as e:
            logger.error(f"Failed to lookup HTS code {hts_code}: {e}")
            return {"error": f"Database error: {str(e)}"}
        finally:
            conn.close()
    
    @staticmethod
    def search_hts_description(search_term: str) -> List[Dict[str, Any]]:
        try:
            conn = sqlite3.connect(Config.DB_NAME)
            cursor = conn.cursor()
            cursor.execute("""
            SELECT hts_code, description 
            FROM hts_codes 
            WHERE description LIKE ? 
            LIMIT 10
            """, (f"%{search_term}%",))
            results = cursor.fetchall()
            return [{"hts_code": r[0], "description": r[1]} for r in results]
        except Exception as e:
            logger.error(f"Failed to search HTS description for '{search_term}': {e}")
            return [{"error": f"Database error: {str(e)}"}]
        finally:
            conn.close()
    
    @staticmethod
    def calculate_duties(hts_code: str, product_cost: float, freight: float, 
                        insurance: float, quantity: int, unit_weight: float,
                        origin_country: Optional[str] = None) -> Dict[str, Any]:
        try:
            if not HTSValidator.validate_hts_code(hts_code):
                return {"error": f"Invalid HTS code format: {hts_code}. Expected format: XXXX.XX.XX.XX"}
            
            hts_data = HTSDatabase.lookup_hts_code(hts_code)
            if "error" in hts_data:
                return hts_data
            
            cif_value = product_cost + freight + insurance
            duty_breakdown = {}
            
            for duty_type in ["general_rate_of_duty", "special_rate_of_duty", "column_2_rate_of_duty"]:
                duty_str = hts_data.get(duty_type)
                parsed = parse_duty_advanced(duty_str, unit_weight, quantity, cif_value)
                duty_breakdown[duty_type] = parsed
            
            total_duties = sum(d["amount"] for d in duty_breakdown.values())
            landed_cost = cif_value + total_duties
            vat_rate = 0.1
            vat_amount = landed_cost * vat_rate
            total_cost = landed_cost + vat_amount
            
            fta_status = ""
            if origin_country:
                conn = sqlite3.connect(Config.DB_NAME)
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM country_codes WHERE code = ?", (origin_country,))
                result = cursor.fetchone()
                conn.close()
                
                if result:
                    country_data = dict(zip([desc[0] for desc in cursor.description], result))
                    if country_data.get('is_fta_member') == 1:
                        fta_status = f"✅ {country_data['name']} ({origin_country}) has preferential trade status under {country_data['fta_name']}. Special rates may apply."
                    else:
                        fta_status = f"❌ {country_data['name']} ({origin_country}) has MFN (Most Favored Nation) status only."
            
            return {
                "hts_code": hts_code,
                "description": hts_data.get("description"),
                "cif_value": cif_value,
                "duty_breakdown": duty_breakdown,
                "total_duties": total_duties,
                "landed_cost": landed_cost,
                "vat_rate": vat_rate,
                "vat_amount": vat_amount,
                "total_cost": total_cost,
                "fta_status": fta_status,
                "calculation_details": {
                    "product_cost": product_cost,
                    "freight": freight,
                    "insurance": insurance,
                    "quantity": quantity,
                    "unit_weight": unit_weight,
                    "origin_country": origin_country
                }
            }
        except Exception as e:
            logger.error(f"Failed to calculate duties for HTS code {hts_code}: {e}")
            return {"error": f"Calculation error: {str(e)}"}
    
    @staticmethod
    def log_query(query_type: str, query_text: str, response_text: str):
        try:
            conn = sqlite3.connect(Config.DB_NAME)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO query_history (timestamp, query_type, query_text, response_text) VALUES (?, ?, ?, ?)",
                (datetime.now().isoformat(), query_type, query_text, response_text)
            )
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to log query: {e}")
        finally:
            conn.close()

class ResponseFormatter:
    @staticmethod
    def format_hts_lookup(data: Dict[str, Any]) -> str:
        if "error" in data:
            return data["error"]
        
        response = f"**HTS Code: {data['hts_code']}**\n\n"
        response += f"**Classification:** Section {data['section']}, Chapter {data['chapter']}\n"
        response += f"**Description:** {data['description']}\n\n"
        response += f"**Duty Rates:**\n"
        response += f"- General Rate: {data['general_rate_of_duty']}\n"
        response += f"- Special Rate: {data['special_rate_of_duty']}\n"
        response += f"- Column 2 Rate: {data['column_2_rate_of_duty']}\n"
        
        if data.get('additional_duties'):
            response += f"- Additional Duties: {data['additional_duties']}\n"
        
        if data.get('unit_of_quantity'):
            response += f"\n**Unit of Quantity:** {data['unit_of_quantity']}\n"
            
        return response
    
    @staticmethod
    def format_duty_calculation(data: Dict[str, Any]) -> str:
        if "error" in data:
            return data["error"]
        
        response = f"**Duty Calculation for HTS {data['hts_code']}**\n\n"
        response += f"**Product:** {data['description']}\n\n"
        
        response += f"**Cost Breakdown:**\n"
        response += f"- Product Cost: ${data['calculation_details']['product_cost']:,.2f}\n"
        response += f"- Freight: ${data['calculation_details']['freight']:,.2f}\n"
        response += f"- Insurance: ${data['calculation_details']['insurance']:,.2f}\n"
        response += f"- **CIF Value: ${data['cif_value']:,.2f}**\n\n"
        
        response += f"**Duty Details:**\n"
        if data.get('duty_breakdown'):
            for duty_type, details in data['duty_breakdown'].items():
                response += f"- {duty_type.replace('_', ' ').title()}: ${details['amount']:,.2f} ({details['type']})\n"
        
        response += f"\n**Summary:**\n"
        response += f"- Total Duties: ${data['total_duties']:,.2f}\n"
        response += f"- Landed Cost: ${data['landed_cost']:,.2f}\n"
        response += f"- VAT ({data['vat_rate']*100}%): ${data['vat_amount']:,.2f}\n"
        response += f"- **Final Cost: ${data['total_cost']:,.2f}**\n"
        
        if data.get("fta_status"):
            response += f"\n**Trade Agreement Status:**\n{data['fta_status']}\n"
            
        return response
    
    @staticmethod
    def format_hts_search(results: List[Dict[str, Any]]) -> str:
        if not results or "error" in results[0]:
            return results[0].get("error", "No results found")
        
        response = "**Search Results:**\n\n"
        for result in results:
            response += f"- **HTS Code**: {result['hts_code']}\n"
            response += f"  **Description**: {result['description']}\n\n"
        return response

class RAGSystem:
    def __init__(self):
        try:
            self.vectorstore = FAISS.load_local(Config.VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            qa_prompt = PromptTemplate(
                template=CUSTOM_PROMPT_TEMPLATE,
                input_variables=["context", "question"]
            )
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(),
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": qa_prompt}
            )
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            self.qa_chain = None
    
    def query(self, question: str) -> str:
        if not self.qa_chain:
            return "The RAG system is not properly initialized. Please initialize the system."
        try:
            result = self.qa_chain({"question": question})
            HTSDatabase.log_query("RAG", question, result["answer"])
            return result["answer"]
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return f"Query error: {str(e)}"

# Export Functions
def export_to_pdf(data: Dict[str, Any], filename: str):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="TariffBot Duty Calculation Report", ln=1, align="C")
        pdf.cell(200, 10, txt=f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1, align="C")
        pdf.ln(10)
        pdf.set_font("Arial", size=10, style="B")
        pdf.cell(50, 10, txt="HTS Code:", ln=0)
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 10, txt=data.get("hts_code", ""), ln=1)
        pdf.set_font("Arial", size=10, style="B")
        pdf.cell(50, 10, txt="Description:", ln=0)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 10, txt=data.get("description", ""))
        pdf.ln(5)
        pdf.set_font("Arial", size=10, style="B")
        pdf.cell(0, 10, txt="Calculation Results:", ln=1)
        pdf.set_font("Arial", size=10)
        for key, value in data.items():
            if key not in ["description", "hts_code", "calculation_details", "fta_status"]:
                pdf.cell(50, 10, txt=f"{key.replace('_', ' ').title()}:", ln=0)
                pdf.cell(0, 10, txt=f"${value:,.2f}" if isinstance(value, (int, float)) else str(value), ln=1)
        if data.get("fta_status"):
            pdf.ln(5)
            pdf.set_font("Arial", size=10, style="I")
            pdf.multi_cell(0, 10, txt=data["fta_status"])
        pdf.output(filename)
        logger.info(f"Exported PDF to {filename}")
    except Exception as e:
        logger.error(f"Failed to export PDF: {e}")
        raise

def export_to_excel(data: Dict[str, Any], filename: str):
    try:
        result_data = {
            "Field": [],
            "Value": []
        }
        for key, value in data.items():
            if key not in ["calculation_details"]:
                result_data["Field"].append(key.replace('_', ' ').title())
                result_data["Value"].append(f"${value:,.2f}" if isinstance(value, (int, float)) else str(value))
        df = pd.DataFrame(result_data)
        writer = pd.ExcelWriter(filename, engine='xlsxwriter')
        df.to_excel(writer, sheet_name='Duty Calculation', index=False)
        if "calculation_details" in data:
            details_df = pd.DataFrame(list(data["calculation_details"].items()), columns=["Parameter", "Value"])
            details_df.to_excel(writer, sheet_name='Input Parameters', index=False)
        writer.close()
        logger.info(f"Exported Excel to {filename}")
    except Exception as e:
        logger.error(f"Failed to export Excel: {e}")
        raise

# Streamlit Application
@st.cache_data
def get_all_hts_codes():
    """Get all HTS codes for validation"""
    try:
        conn = sqlite3.connect(Config.DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT hts_code FROM hts_codes LIMIT 1000")
        codes = [row[0] for row in cursor.fetchall()]
        return codes
    except Exception as e:
        return []
    finally:
        conn.close()

def main():
    st.set_page_config(
        page_title="TariffBot - HTS AI Agent",
        page_icon="🌐",
        layout="wide"
    )

    # Initialize session state
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
        st.session_state.rag_system = None
        st.session_state.error_message = None
    if "force_reingest" not in st.session_state:
        st.session_state.force_reingest = False

    # Sidebar navigation
    st.sidebar.title("TariffBot")
    st.sidebar.markdown("""
    Welcome to TariffBot — your intelligent assistant for U.S. Harmonized Tariff Schedule (HTS) queries.  
    Select an option below to get started.
    """)
    option = st.sidebar.selectbox(
        "Choose an Action",
        ["Home", "Lookup HTS Code", "Search HTS Description", "Calculate Duties", "Ask Trade Question"]
    )
    st.sidebar.markdown("---")
    st.session_state.force_reingest = st.sidebar.checkbox(
        "Force PDF Re-ingestion",
        value=st.session_state.force_reingest,
        help="Enable to re-process General Notes.pdf on next initialization. Uncheck to use existing FAISS index."
    )

    # Initialize system
    @st.cache_resource
    def init_system(force_reingest):
        try:
            success, msg = initialize_system(force_reingest=force_reingest)
            if success:
                rag_system = RAGSystem()
                if rag_system.qa_chain is None:
                    return False, "Failed to initialize RAG system.", None
                return True, "System initialized successfully.", rag_system
            return False, msg, None
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            return False, f"Initialization failed: {str(e)}", None

    # Perform initialization
    if not st.session_state.initialized:
        with st.spinner("Initializing TariffBot..."):
            success, msg, rag_system = init_system(st.session_state.force_reingest)
            if success:
                st.session_state.initialized = True
                st.session_state.rag_system = rag_system
                st.success(msg)
            else:
                st.session_state.error_message = msg
                st.error(msg)

    # Main content
    st.title("TariffBot - HTS AI Agent")
    st.markdown("""
    TariffBot is powered by U.S. International Trade Commission data to assist importers, analysts, and trade professionals.  
    I provide clear, compliant, and factual answers grounded in official HTS documentation.  
    Use the sidebar to navigate through my capabilities.
    """)

    if st.session_state.error_message:
        st.error(f"System Error: {st.session_state.error_message}")
        st.stop()

    if st.session_state.initialized:
        if option == "Home":
            st.header("Welcome to TariffBot")
            quick_query = st.text_input("Quick Query (I'll route it automatically):", "")
            if quick_query:
                query_type = QueryRouter.classify_query(quick_query)
                st.info(f"Query classified as: {query_type}")
                
                if query_type == "hts_lookup":
                    hts_match = re.search(r'(\d{4}\.\d{2}\.\d{2}\.\d{2})', quick_query)
                    if hts_match:
                        result = HTSDatabase.lookup_hts_code(hts_match.group(1))
                        st.write(ResponseFormatter.format_hts_lookup(result))
                
                elif query_type == "product_search":
                    search_term = quick_query.replace("hts code for", "").replace("tariff code for", "").strip()
                    results = HTSDatabase.search_hts_description(search_term)
                    st.write(ResponseFormatter.format_hts_search(results))
                
                elif query_type in ["trade_policy", "description_search"]:
                    answer = st.session_state.rag_system.query(quick_query)
                    st.write(answer)
            
            st.markdown("""
            ### What I Can Do:
            - **Lookup HTS Codes**: Find detailed information about specific HTS codes.
            - **Search Descriptions**: Search for HTS codes by product description.
            - **Calculate Duties**: Compute duties, landed costs, and VAT based on HTS codes and shipment details.
            - **Answer Trade Questions**: Get insights on trade policies and agreements using official HTS General Notes.

            ### Sample Queries:
            - **RAG Agent**:
              - What is the United States-Israel Free Trade Agreement?
              - Can a product exceeding its tariff-rate quota qualify for duty-free entry under GSP or FTA?
              - How is classification determined for imported manufacturing parts?
            - **Tariff Agent**:
              - Calculate duties for HTS code 0101.30.00.00 with $10,000 product cost, 500 kg, 5 units.
              - Find the HTS code for donkeys.
              - What are the duty rates for female cattle?
            """)
        
        elif option == "Lookup HTS Code":
            st.header("Lookup HTS Code")
            hts_code = st.text_input("Enter HTS Code (e.g., 0101.30.00.00)", "").strip()
            if st.button("Lookup"):
                if hts_code:
                    with st.spinner("Fetching HTS code details..."):
                        result = HTSDatabase.lookup_hts_code(hts_code)
                        formatted_response = ResponseFormatter.format_hts_lookup(result)
                        st.write(formatted_response)
                        with st.expander("View raw data"):
                            if "error" not in result:
                                df = pd.DataFrame([result])
                                st.dataframe(df, use_container_width=True)
                else:
                    st.warning("Please enter an HTS code.")
        
        elif option == "Search HTS Description":
            st.header("Search HTS Description")
            search_term = st.text_input("Enter Search Term (e.g., donkeys)", "").strip()
            if st.button("Search"):
                if search_term:
                    with st.spinner("Searching HTS descriptions..."):
                        results = HTSDatabase.search_hts_description(search_term)
                        formatted_response = ResponseFormatter.format_hts_search(results)
                        st.write(formatted_response)
                        with st.expander("View raw data"):
                            if not results or "error" in results[0]:
                                st.error(results[0].get("error", "No results found"))
                            else:
                                df = pd.DataFrame(results)
                                st.dataframe(df, use_container_width=True)
                else:
                    st.warning("Please enter a search term.")
        
        elif option == "Calculate Duties":
            st.header("Calculate Duties")
            with st.form("duty_calculator_form"):
                hts_code = st.text_input("HTS Code (e.g., 0101.30.00.00)", "").strip()
                product_cost = st.number_input("Product Cost (USD)", min_value=0.0, step=100.0)
                freight = st.number_input("Freight Cost (USD)", min_value=0.0, step=50.0)
                insurance = st.number_input("Insurance Cost (USD)", min_value=0.0, step=10.0)
                quantity = st.number_input("Quantity", min_value=1, step=1, format="%d")
                unit_weight = st.number_input("Unit Weight (kg)", min_value=0.0, step=0.1)
                origin_country = st.text_input("Origin Country Code (e.g., US) [Optional]", "").strip().upper()
                origin_country = origin_country if origin_country else None
                export_option = st.selectbox("Export Results", ["None", "PDF", "Excel"])
                submitted = st.form_submit_button("Calculate")

            if submitted:
                if not hts_code:
                    st.warning("Please enter an HTS code.")
                else:
                    try:
                        with st.spinner("Calculating duties..."):
                            result = HTSDatabase.calculate_duties(
                                hts_code, product_cost, freight, insurance, quantity, unit_weight, origin_country
                            )
                            formatted_response = ResponseFormatter.format_duty_calculation(result)
                            st.write(formatted_response)
                            
                            if export_option != "None":
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename = f"duty_report_{hts_code}_{timestamp}.{export_option.lower()}"
                                if export_option == "PDF":
                                    export_to_pdf(result, filename)
                                    st.success(f"Exported PDF to {filename}")
                                    with open(filename, "rb") as f:
                                        st.download_button(
                                            label="Download PDF",
                                            data=f,
                                            file_name=filename,
                                            mime="application/pdf"
                                        )
                                elif export_option == "Excel":
                                    export_to_excel(result, filename)
                                    st.success(f"Exported Excel to {filename}")
                                    with open(filename, "rb") as f:
                                        st.download_button(
                                            label="Download Excel",
                                            data=f,
                                            file_name=filename,
                                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                        )
                    except Exception as e:
                        st.error(f"Error in calculation: {str(e)}")
        
        elif option == "Ask Trade Question":
            st.header("Ask Trade Question")
            question = st.text_area("Enter your question about trade policies or agreements", "").strip()
            if st.button("Submit Question"):
                if question:
                    with st.spinner("Processing your question..."):
                        answer = st.session_state.rag_system.query(question)
                        st.subheader("Answer")
                        st.write(answer)
                else:
                    st.warning("Please enter a question.")

    # Footer
    st.markdown("---")
    st.markdown("""
    Built to assist with U.S. HTS queries. For manual review, visit [hts.usitc.gov](https://hts.usitc.gov).  
    If a query is ambiguous, I'll recommend checking the relevant HTS section.
    """)

if __name__ == "__main__":
    main()