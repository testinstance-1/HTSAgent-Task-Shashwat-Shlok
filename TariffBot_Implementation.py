import streamlit as st
import duckdb
import pandas as pd
import re
import os
import logging
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime
import warnings
import json
import openpyxl
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM

# Set environment variables before any imports
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'  

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize LLM (Mistral 7B via Ollama)
llm = OllamaLLM(model="mistral:7b", base_url="http://localhost:11434")

# Initialize embeddings 
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        cache_folder=r"C:\Users\sshas\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    )
except Exception as e:
    logger.error(f"Failed to load embeddings: {str(e)}")
    st.error(f"Failed to load embeddings model: {str(e)}. Ensure the model files are present at the specified cache folder.")
    raise

# Paths to data
GENERAL_NOTES_PATH = r"C:\Users\sshas\Downloads\HTSAgent\data\General Notes.pdf"
CSV_PATHS = [
    r"C:\Users\sshas\Downloads\HTSAgent\data\htsdata.csv",
    r"C:\Users\sshas\Downloads\HTSAgent\data\htsdata (1).csv",
    r"C:\Users\sshas\Downloads\HTSAgent\data\htsdata (2).csv",
    r"C:\Users\sshas\Downloads\HTSAgent\data\htsdata (3).csv",
    r"C:\Users\sshas\Downloads\HTSAgent\data\htsdata (4).csv"
]

# Country mapping
COUNTRY_MAPPING = {
    "AU": "Australia",
    "CA": "Canada",
    "MX": "Mexico",
    "IL": "Israel",
    "IN": "India",
    "US": "United States",
    "DE": "Germany",
}

# Initialize DuckDB
conn = duckdb.connect("hts.db")

# Data Ingestion
def ingest_data():
    logger.info("Starting data ingestion...")
    
    # Ingest CSVs
    dfs = []
    missing_files = []
    for csv_path in CSV_PATHS:
        if os.path.exists(csv_path):
            logger.info(f"Loading CSV: {csv_path}")
            df = pd.read_csv(csv_path)
            for code, name in COUNTRY_MAPPING.items():
                df["Special Rate of Duty"] = df["Special Rate of Duty"].str.replace(code, name, regex=False)
            dfs.append(df)
        else:
            logger.warning(f"CSV file not found: {csv_path}")
            missing_files.append(csv_path)
    if missing_files:
        st.warning(f"Missing CSV files: {', '.join(missing_files)}. Download additional HTS data from https://hts.usitc.gov/export or https://catalog.data.gov/dataset/harmonized-tariff-schedule-of-the-united-states-2025.")
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df["HTS Number"] = combined_df["HTS Number"].str.strip().str.replace(r'[^\d.]', '', regex=True)
        logger.info("Writing combined CSV data to DuckDB")
        conn.execute("CREATE OR REPLACE TABLE hts_data AS SELECT * FROM combined_df")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hts_number ON hts_data(\"HTS Number\")")
    else:
        logger.error("No CSV files were loaded")
        st.error("No CSV files were loaded. Please ensure at least one HTS CSV file is available.")

    # Ingest General Notes PDF
    persist_dir = "./chroma_db"
    if os.path.exists(persist_dir) and os.path.isdir(persist_dir):
        logger.info("Chroma vector store already exists, skipping PDF ingestion")
    else:
        if os.path.exists(GENERAL_NOTES_PATH):
            logger.info(f"Loading PDF: {GENERAL_NOTES_PATH}")
            try:
                loader = PyMuPDFLoader(GENERAL_NOTES_PATH)
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(documents)
                logger.info("Creating Chroma vector store")
                Chroma.from_documents(
                    chunks, 
                    embeddings, 
                    collection_name="general_notes",
                    persist_directory=persist_dir
                )
                logger.info("PDF ingestion completed")
            except Exception as e:
                logger.error(f"Error ingesting PDF: {str(e)}")
                st.error(f"Error ingesting General Notes PDF: {str(e)}")
        else:
            logger.error(f"General Notes PDF not found at: {GENERAL_NOTES_PATH}")
            st.error(f"General Notes PDF not found at: {GENERAL_NOTES_PATH}")

# Load General Notes for RAG
def load_general_notes():
    try:
        vectorstore = Chroma(
            collection_name="general_notes", 
            embedding_function=embeddings, 
            persist_directory="./chroma_db"
        )
        return vectorstore.as_retriever()
    except Exception as e:
        logger.error(f"Error loading vector store: {str(e)}")
        st.error(f"Error loading vector store: {str(e)}")
        return None

# Enhanced RAG Tool
def rag_tool_func(query):
    retriever = load_general_notes()
    if retriever is None:
        return "Error: Unable to load General Notes data."
    try:
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant information found in General Notes."
        
        content = docs[0].page_content
        hts_codes = re.findall(r'\b\d{4}\.\d{2}\.\d{2}\.\d{2}\b', content)
        
        response = content[:2000]
        if hts_codes:
            response += f"\n\nRelated HTS codes: {', '.join(hts_codes[:3])}"
        
        return response
    except Exception as e:
        logger.error(f"Error invoking RAG tool: {str(e)}")
        return f"Error processing query: {str(e)}"

# HTS Code Lookup Tool
def hts_lookup_func(product_name):
    try:
        product_name = re.sub(r'[^\w\s]', '', product_name).strip().lower()
        query = "SELECT * FROM hts_data WHERE LOWER(\"Description\") LIKE ?"
        df = conn.execute(query, [f"%{product_name}%"]).fetch_df()
        
        if df.empty:
            return f"No HTS codes found for '{product_name}'"
        
        results = []
        for _, row in df.iterrows():
            results.append(f"{row['HTS Number']}: {row['Description']}")
        
        return "\n".join(results[:5])
    except Exception as e:
        logger.error(f"Error in HTS lookup: {str(e)}")
        return f"Error looking up HTS codes: {str(e)}"

# Duty Parser
def parse_duty_advanced(duty_str, unit_weight=None, quantity=None, cif_value=None, country=None):
    if pd.isna(duty_str) or duty_str.strip() == "":
        return 0.0
    duty_str = duty_str.strip().lower()
    if "free" in duty_str:
        return 0.0

    if country and "(" in duty_str:
        country = country.lower()
        matches = re.findall(r"([\w\s]+)\s*\(([\w\s,]+)\)", duty_str)
        for rate, countries in matches:
            rate = rate.strip()
            countries = [c.strip().lower() for c in countries.split(",")]
            if country in countries:
                if "free" in rate:
                    return 0.0
                match = re.search(r"([\d.]+)\s*%", rate)
                if match:
                    return float(match.group(1)) / 100
                match = re.search(r"([\d.]+)\s*¢/kg", rate)
                if match and unit_weight is not None:
                    return (float(match.group(1)) / 100) * unit_weight
                match = re.search(r"\$([\d.]+)/unit", rate)
                if match and quantity is not None:
                    return float(match.group(1)) * quantity

    match = re.search(r"([\d.]+)\s*%", duty_str)
    if match:
        return float(match.group(1)) / 100
    match = re.search(r"([\d.]+)\s*¢/kg", duty_str)
    if match and unit_weight is not None:
        return (float(match.group(1)) / 100) * unit_weight
    match = re.search(r"\$([\d.]+)/unit", duty_str)
    if match and quantity is not None:
        return float(match.group(1)) * quantity
    return 0.0

# Tariff Calculator Tool
def calculate_tariff(inputs):
    if isinstance(inputs, str):
        try:
            inputs = json.loads(inputs)
        except:
            params = {}
            pattern = r"(\w+)\s*[:=]\s*([\w.]+)"
            matches = re.findall(pattern, inputs)
            for key, value in matches:
                try:
                    params[key] = float(value) if '.' in value else value
                except:
                    params[key] = value
            inputs = params
    
    hts_code = inputs.get("hts_code", "")
    hts_code = re.sub(r'[^\d.]', '', hts_code).strip()
    
    product_cost = float(inputs.get("product_cost", 0))
    freight = float(inputs.get("freight", 0))
    insurance = float(inputs.get("insurance", 0))
    unit_weight = float(inputs.get("unit_weight", 0))
    quantity = float(inputs.get("quantity", 0))
    country = inputs.get("country", None)
    unit_of_quantity = inputs.get("unit_of_quantity", None)

    cif_value = product_cost + freight + insurance

    query = "SELECT * FROM hts_data WHERE \"HTS Number\" = ?"
    df = conn.execute(query, [hts_code]).fetch_df()
    if df.empty:
        alt_query = "SELECT * FROM hts_data WHERE \"HTS Number\" LIKE ?"
        df = conn.execute(alt_query, [f"%{hts_code}%"]).fetch_df()
        if df.empty:
            logger.warning(f"No data found for HTS code: {hts_code}")
            return f"No data found for HTS code {hts_code}. Please verify the code in the HTS database."

    expected_unit = df["Unit of Quantity"].iloc[0].strip().lower()
    if unit_of_quantity and unit_of_quantity.lower() != expected_unit:
        st.warning(f"Input unit ({unit_of_quantity}) does not match HTS unit ({expected_unit}).")

    quota_qty = df["Quota Quantity"].iloc[0]
    if pd.notna(quota_qty) and quantity > float(quota_qty):
        st.warning(f"Quantity ({quantity}) exceeds tariff-rate quota ({quota_qty}). Higher rates may apply.")

    result = {
        "HTS Number": hts_code,
        "Description": df["Description"].iloc[0],
        "Unit of Quantity": expected_unit,
        "CIF Value": cif_value,
        "Product Cost": product_cost,
        "Freight": freight,
        "Insurance": insurance,
        "Unit Weight": unit_weight,
        "Quantity": quantity,
        "Country": country if country else "N/A"
    }

    duty_calculations = []
    
    for col in ["General Rate of Duty", "Special Rate of Duty", "Column 2 Rate of Duty"]:
        duty_str = df[col].iloc[0]
        parsed_rate = parse_duty_advanced(
            duty_str,
            unit_weight=unit_weight,
            quantity=quantity,
            cif_value=cif_value,
            country=country
        )
        
        if "%" in str(duty_str).lower():
            duty_amount = parsed_rate * cif_value
            rate_percent = parsed_rate * 100
            result[f"{col} Rate (%)"] = rate_percent
            duty_calculations.append(f"- {col}: {rate_percent:.2f}% of CIF Value = ${duty_amount:.2f}")
        else:
            duty_amount = parsed_rate
            result[f"{col} Rate (%)"] = 0
            duty_calculations.append(f"- {col}: {duty_str} = ${duty_amount:.2f}")
            
        result[f"{col} Duty Amount"] = duty_amount

    add_duty_str = df["Additional Duties"].iloc[0]
    add_duty_amount = parse_duty_advanced(
        add_duty_str,
        unit_weight=unit_weight,
        quantity=quantity,
        cif_value=cif_value
    )
    result["Additional Duties Amount"] = add_duty_amount
    duty_calculations.append(f"- Additional Duties: {add_duty_str} = ${add_duty_amount:.2f}")

    vat_rate = 0.05
    vat_amount = cif_value * vat_rate
    result["VAT Rate (%)"] = vat_rate * 100
    result["VAT Amount"] = vat_amount
    duty_calculations.append(f"- VAT: {vat_rate*100:.2f}% of CIF Value = ${vat_amount:.2f}")

    total_duty = sum([result.get(f"{col} Duty Amount", 0) for col in ["General Rate of Duty", "Special Rate of Duty", "Column 2 Rate of Duty"]]) + add_duty_amount
    result["Total Landed Cost"] = cif_value + total_duty + vat_amount
    
    calculation_steps = [
        f"Calculation for HTS Code: {hts_code}",
        f"Description: {result['Description']}",
        "",
        "Input Parameters:",
        f"- Product Cost: ${product_cost:,.2f}",
        f"- Freight: ${freight:,.2f}",
        f"- Insurance: ${insurance:,.2f}",
        f"- Unit Weight: {unit_weight} kg",
        f"- Quantity: {quantity}",
        f"- Country: {country if country else 'Not specified'}",
        "",
        "Calculation Steps:",
        f"1. CIF Value = Product Cost + Freight + Insurance = ${cif_value:,.2f}",
        "",
        "2. Duty Calculations:"
    ]
    
    calculation_steps.extend(duty_calculations)
    
    calculation_steps.extend([
        "",
        f"3. Total Duties = ${total_duty:,.2f}",
        f"4. VAT (5%) = ${vat_amount:,.2f}",
        f"5. Total Landed Cost = CIF Value + Total Duties + VAT = ${result['Total Landed Cost']:,.2f}",
        "",
        "Note: This calculation assumes standard VAT rate of 5%. Actual rates may vary."
    ])
    
    st.session_state.detailed_calculation = "\n".join(calculation_steps)
    
    return result

# Tools for Agent
rag_tool = Tool(
    name="GeneralNotes",
    func=rag_tool_func,
    description="Answers questions about trade policies, agreements, or classification rules."
)

hts_lookup_tool = Tool(
    name="HTSLookup",
    func=hts_lookup_func,
    description="Finds HTS codes for products based on description."
)

tariff_tool = Tool(
    name="TariffCalculator",
    func=calculate_tariff,
    description="Calculates duties and landed costs for HTS codes."
)

# Initialize Agent with  error handling
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
tools = [rag_tool, hts_lookup_tool, tariff_tool]
agent_executor = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True,
    memory=memory,
    handle_parsing_errors=(
        "Check your output and make sure it conforms! "
        "Use the exact format: Action: [tool_name] without any extra characters, "
        "followed by Action Input: [input] on a new line."
    ),
    max_iterations=5,
    early_stopping_method="generate",
    return_intermediate_steps=True
)

# Streamlit Frontend
def generate_pdf(report_data, calculation_steps):
    pdf_path = f"hts_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "HTS Duty Report")
    c.setFont("Helvetica", 12)
    
    y = 700
    for key, value in report_data.items():
        c.drawString(100, y, f"{key}: {value}")
        y -= 20
    
    c.drawString(100, y-40, "Calculation Steps:")
    y -= 60
    
    lines = calculation_steps.split('\n')
    for line in lines:
        if y < 100:
            c.showPage()
            y = 750
            c.setFont("Helvetica", 12)
        c.drawString(110, y, line)
        y -= 15
    
    c.save()
    return pdf_path

def generate_excel(report_data, calculation_steps):
    excel_path = f"hts_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    report_df = pd.DataFrame([report_data])
    
    steps_df = pd.DataFrame({
        "Calculation Steps": calculation_steps.split('\n')
    })
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        report_df.to_excel(writer, sheet_name='Duty Calculation', index=False)
        steps_df.to_excel(writer, sheet_name='Calculation Steps', index=False)
    
    return excel_path

def format_tariff_result(result):
    if isinstance(result, dict):
        currency_fields = ["CIF Value", "Product Cost", "Freight", "Insurance", 
                          "General Rate of Duty Duty Amount", "Special Rate of Duty Duty Amount",
                          "Column 2 Rate of Duty Duty Amount", "Additional Duties Amount",
                          "VAT Amount", "Total Landed Cost"]
        for field in currency_fields:
            if field in result:
                result[field] = f"${float(result[field]):,.2f}"
        rate_fields = ["General Rate of Duty Rate (%)", "Special Rate of Duty Rate (%)", 
                      "Column 2 Rate of Duty Rate (%)", "VAT Rate (%)"]
        for field in rate_fields:
            if field in result:
                result[field] = f"{float(result[field]):.2f}%"
        if "Quantity" in result:
            result["Quantity"] = f"{float(result['Quantity']):,.0f} units"
        if "Unit Weight" in result:
            result["Unit Weight"] = f"{float(result['Unit Weight']):,.2f} kg"
    return result

def is_duty_related_query(query):
    duty_keywords = ["duty", "tariff", "cost", "calculate", "rate", "hts_code", "product_cost", "freight", "insurance", "quantity"]
    return any(keyword in query.lower() for keyword in duty_keywords)

def parse_query_params(query):
    params = {}
    query_lower = query.lower()
    
    # Extract HTS code
    hts_match = re.search(r'\b(\d{4}\.\d{2}\.\d{2}\.\d{2})\b', query)
    if hts_match:
        params["hts_code"] = hts_match.group(1)
    
    # Extract product cost
    cost_match = re.search(r'product cost.*?\b\$?([\d,]+(?:\.\d+)?)\b', query_lower, re.IGNORECASE)
    if cost_match:
        params["product_cost"] = float(cost_match.group(1).replace(",", ""))
    
    # Extract weight
    weight_match = re.search(r'(\d+\.?\d*)\s*kg(?:\s*weight)?', query_lower)
    if weight_match:
        params["unit_weight"] = float(weight_match.group(1))
    
    # Extract quantity
    quantity_match = re.search(r'(\d+\.?\d*)\s*units?', query_lower)
    if quantity_match:
        params["quantity"] = float(quantity_match.group(1))
    
    # Extract country
    country_match = re.search(r'country\s*[:=]\s*(\w+)', query_lower)
    if country_match:
        params["country"] = country_match.group(1)
    
    # Extract freight and insurance if provided
    freight_match = re.search(r'freight\s*[:=]\s*\$?([\d,]+(?:\.\d+)?)', query_lower)
    if freight_match:
        params["freight"] = float(freight_match.group(1).replace(",", ""))
    else:
        params["freight"] = 0.0
    
    insurance_match = re.search(r'insurance\s*[:=]\s*\$?([\d,]+(?:\.\d+)?)', query_lower)
    if insurance_match:
        params["insurance"] = float(insurance_match.group(1).replace(",", ""))
    else:
        params["insurance"] = 0.0
    
    return params

def classify_and_rewrite_query(query):
    query_lower = query.lower()
    if any(k in query_lower for k in ["what is", "explain", "how is", "classification", "trade agreement"]):
        return f"Use GeneralNotes to answer: {query}"
    elif "hts code for" in query_lower or "hts code of" in query_lower:
        product = re.search(r'hts code (?:for|of) (.+?)(?:\?|,|$)', query_lower, re.IGNORECASE)
        if product:
            product_name = product.group(1).strip()
            return f"Use HTSLookup to find the HTS code for '{product_name}'"
    elif "given hts code" in query_lower and any(k in query_lower for k in ["cost", "weight", "quantity"]):
        params = parse_query_params(query)
        if "hts_code" in params:
            # Validate required parameters
            required_params = ["hts_code", "product_cost", "quantity"]
            missing_params = [p for p in required_params if p not in params or params[p] == 0]
            if missing_params:
                logger.warning(f"Missing required parameters: {missing_params}")
                return f"Error: Missing required parameters: {', '.join(missing_params)}"
            return f"Use TariffCalculator with input: {json.dumps(params)}"
    elif "duty rates for" in query_lower:
        product = re.search(r'duty rates for (.+?)(?:\?|,|$)', query_lower, re.IGNORECASE)
        if product:
            product_name = product.group(1).strip()
            return f"Use HTSLookup to find HTS codes for '{product_name}'"
    return query

def main():
    st.set_page_config(page_title="TariffBot - HTS AI Agent", layout="wide")
    st.title("TariffBot 🌎 - Your HTS AI Assistant")
    
    if 'detailed_calculation' not in st.session_state:
        st.session_state.detailed_calculation = None
    
    if 'data_loaded' not in st.session_state:
        with st.spinner("Ingesting data..."):
            ingest_data()
            st.session_state.data_loaded = True
    
    query = st.text_input("Enter your query (e.g., 'What is United States-Israel Free Trade?', 'What’s the HTS code for donkeys?', 'What are the duty rates for veal?', or 'Given HTS code 0101.30.00.00, product cost of $10,000, 500 kg weight, and 5 units — what are all applicable duties?')")
    
    if st.button("Submit Query"):
        st.session_state.detailed_calculation = None
        
        if 'last_query' in st.session_state and st.session_state.last_query == query:
            st.warning("Query already processed. Please enter a new query.")
            return
            
        st.session_state.last_query = query
        
        with st.spinner("Processing query..."):
            try:
                logger.info(f"Processing query: {query}")
                rewritten_query = classify_and_rewrite_query(query)
                if rewritten_query.startswith("Error"):
                    st.error(rewritten_query)
                    return
                agent_output = agent_executor.invoke({"input": rewritten_query})
                
                result = None
                for step in agent_output.get("intermediate_steps", []):
                    if step[0].tool == "TariffCalculator":
                        try:
                            result = json.loads(step[1]) if isinstance(step[1], str) else step[1]
                            break
                        except:
                            continue
                
                if not result:
                    try:
                        result = json.loads(agent_output["output"]) if isinstance(agent_output["output"], str) else agent_output["output"]
                    except:
                        result = agent_output["output"]
                
                if isinstance(result, dict):
                    formatted = format_tariff_result(result.copy())
                    
                    st.subheader("Duty Calculation Results")
                    
                    if st.session_state.detailed_calculation:
                        with st.expander("View Detailed Calculation Steps"):
                            st.text(st.session_state.detailed_calculation)
                    
                    st.subheader("Summary")
                    df = pd.DataFrame([formatted])
                    st.dataframe(df)
                    
                    if st.session_state.detailed_calculation:
                        st.subheader("Download Report")
                        col1, col2 = st.columns(2)
                        with col1:
                            excel_path = generate_excel(formatted, st.session_state.detailed_calculation)
                            with open(excel_path, "rb") as f:
                                st.download_button(
                                    label="Download Excel Report",
                                    data=f,
                                    file_name=excel_path,
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key=f"excel_download_{datetime.now().timestamp()}"
                                )
                        with col2:
                            pdf_path = generate_pdf(formatted, st.session_state.detailed_calculation)
                            with open(pdf_path, "rb") as f:
                                st.download_button(
                                    label="Download PDF Report",
                                    data=f,
                                    file_name=pdf_path,
                                    mime="application/pdf",
                                    key=f"pdf_download_{datetime.now().timestamp()}"
                                )
                elif isinstance(result, str):
                    if "No data found for HTS code" in result:
                        st.error(result)
                    else:
                        st.subheader("Response")
                        st.write(result)
                else:
                    st.error("Unexpected response format from agent.")
                
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                st.error(f"Error processing query: {str(e)}")
                if is_duty_related_query(query):
                    params = parse_query_params(query)
                    if "hts_code" in params:
                        result = calculate_tariff(params)
                        if isinstance(result, dict):
                            formatted = format_tariff_result(result.copy())
                            
                            st.subheader("Duty Calculation Results (Fallback)")
                            if st.session_state.detailed_calculation:
                                with st.expander("View Detailed Calculation Steps"):
                                    st.text(st.session_state.detailed_calculation)
                            
                            st.subheader("Summary")
                            df = pd.DataFrame([formatted])
                            st.dataframe(df)
                            
                            st.subheader("Download Report")
                            col1, col2 = st.columns(2)
                            with col1:
                                excel_path = generate_excel(formatted, st.session_state.detailed_calculation)
                                with open(excel_path, "rb") as f:
                                    st.download_button(
                                        label="Download Excel Report",
                                        data=f,
                                        file_name=excel_path,
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        key=f"excel_download_fallback_{datetime.now().timestamp()}"
                                    )
                            with col2:
                                pdf_path = generate_pdf(formatted, st.session_state.detailed_calculation)
                                with open(pdf_path, "rb") as f:
                                    st.download_button(
                                        label="Download PDF Report",
                                        data=f,
                                        file_name=pdf_path,
                                        mime="application/pdf",
                                        key=f"pdf_download_fallback_{datetime.now().timestamp()}"
                                    )
                        else:
                            st.error(result)

if __name__ == "__main__":
    main()
