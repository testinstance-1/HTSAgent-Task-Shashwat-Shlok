# HTS AI Agent – TariffBot

TariffBot is an AI-powered tool designed to assist importers, analysts, and trade professionals with U.S. Harmonized Tariff Schedule (HTS) queries. It combines a Retrieval-Augmented Generation (RAG) agent for trade policy questions and a tariff calculator for computing duties, landed costs, and VAT. Built with open-source technologies like LangChain, SQLite, FAISS, and Streamlit, TariffBot provides clear, compliant, and factual answers grounded in official HTS documentation from [hts.usitc.gov](https://hts.usitc.gov).

## Features

- **RAG-Based Question Answering**: Answers trade policy and agreement questions (e.g., "What is the United States-Israel Free Trade Agreement?") using semantic search over the General Notes PDF, powered by LangChain and FAISS.
- **HTS Duty Calculator**: Computes duties, landed costs, and VAT for given HTS codes, product costs, freight, insurance, quantity, and unit weight, with support for multiple duty formats (%, ¢/kg, $/unit).
- **Multi-Section CSV Ingestion**: Supports ingestion of multiple HTS CSV files for comprehensive tariff data coverage.
- **Enhanced Country Data**: Maps country codes to full names (e.g., AU → Australia) with Free Trade Agreement (FTA) details.
- **Query History Logging**: Stores query details in an SQLite database for tracking.
- **Export Options**: Exports duty calculation results to PDF or Excel.
- **Streamlit Interface**: User-friendly web interface for querying, searching, and calculating duties.
- **Query Routing**: Automatically classifies user queries to route them to the appropriate tool (RAG or tariff calculator).
- **Performance Optimization**: Uses Streamlit caching for faster initialization and HTS code retrieval.

## Prerequisites

- Python 3.8+
- Required Python packages:
  ```bash
  pip install pandas sqlite3 langchain langchain-community langchain-huggingface streamlit fpdf PyPDF2 transformers torch xlsxwriter
  ```
- HTS CSV files from [hts.usitc.gov](https://hts.usitc.gov) (e.g., `htsdata.csv`, `hts_2025_basic_edition_csv.csv`).
- General Notes PDF from [hts.usitc.gov](https://hts.usitc.gov).

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/HTSAgent-Task-<YourName>.git
   cd HTSAgent-Task-<YourName>
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Alternatively, install the required packages listed above.

3. **Prepare Data**:
   - Place HTS CSV files in the `data/` directory (e.g., `C:\Users\sshas\Downloads\HTSAgent\data\`).
   - Place the `General Notes.pdf` file in the `data/` directory.
   - Update the `Config` class in `hts_agent.py` if your file paths differ:
     ```python
     class Config:
         DB_NAME = r"path/to/hts_data.db"
         VECTOR_STORE_PATH = r"path/to/faiss_index"
         CSV_PATH = r"path/to/hts_2025_basic_edition_csv.csv"
         PDF_PATH = r"path/to/General Notes.pdf"
         ALL_CSV_PATHS = [
             r"path/to/htsdata.csv",
             r"path/to/htsdata (1).csv",
             r"path/to/htsdata (2).csv",
             r"path/to/htsdata (3).csv",
             r"path/to/htsdata (4).csv",
         ]
     ```

4. **Run the Application**:
   ```bash
   streamlit run hts_agent.py
   ```
   This will launch the Streamlit web interface at `http://localhost:8501`.

## Usage

1. **Launch the Streamlit App**:
   - Open the app in your browser.
   - Use the sidebar to select an action: "Home", "Lookup HTS Code", "Search HTS Description", "Calculate Duties", or "Ask Trade Question".

2. **Sample Queries**:
   - **RAG Agent**:
     - "What is the United States-Israel Free Trade Agreement?"
     - "Can a product that exceeds its tariff-rate quota still qualify for duty-free entry under GSP or any FTA? Why or why not?"
     - "How is classification determined for an imported item that will be used as a part in manufacturing but isn’t itself a finished part?"
   - **Tariff Agent**:
     - Calculate duties for HTS code `0101.30.00.00` with $10,000 product cost, 500 kg weight, and 5 units.
     - Find the HTS code for "donkeys".
     - Get the applicable duty rates for "female cattle".

3. **Quick Query**:
   - On the "Home" page, enter a query (e.g., "HTS code for donkeys" or "What is USMCA?") to automatically route it to the appropriate tool.

4. **Export Results**:
   - In the "Calculate Duties" section, select "PDF" or "Excel" to export the duty calculation results.

## Project Structure

- `hts_agent.py`: Main script containing the TariffBot implementation, including data ingestion, RAG system, duty calculator, and Streamlit interface.
- `data/`: Directory for HTS CSV files and General Notes PDF.
- `faiss_index/`: Directory for FAISS vector store (generated during initialization).
- `hts_data.db`: SQLite database for HTS codes, country codes, and query history.

## Bonus Features

- **Multi-Section Ingestion**: Supports multiple HTS CSV files via `Config.ENABLE_MULTI_SECTION`.
- **Query History**: Logs all queries in the `query_history` table for auditing.
- **Country Code Enhancements**: Includes FTA details and mappings (e.g., CA → Canada, USMCA).
- **Advanced Duty Parsing**: Handles various duty formats (%, ¢/kg, $/unit, compound rates).
- **Streamlit Caching**: Optimizes performance with `@st.cache_data` and `@st.cache_resource`.
- **Error Handling**: Validates HTS codes and handles duplicate entries during CSV ingestion.

## Troubleshooting

- **UNIQUE Constraint Error**: If you encounter `UNIQUE constraint failed: hts_codes.hts_code`, ensure the `ingest_hts_csv` function uses `INSERT OR IGNORE` to skip duplicate HTS codes.
- **Missing Data**: Verify that all CSV files and the General Notes PDF are in the correct paths specified in `Config`.
- **RAG Issues**: If RAG responses are vague, check the FAISS index (`faiss_index/`) and force re-ingestion via the Streamlit checkbox.



## Submission Details

This project was developed for the HTS AI Agent coding task by Shashwat Shlok. For feedback or inquiries, contact sshashwat10@gmail.com.


## Acknowledgments

Built with open-source tools:
- [LangChain](https://github.com/langchain-ai/langchain) for RAG and agent framework.
- [FAISS](https://github.com/facebookresearch/faiss) for vector storage.
- [Streamlit](https://streamlit.io) for the web interface.
- [SQLite](https://www.sqlite.org) for data storage.
- Data sourced from [hts.usitc.gov](https://hts.usitc.gov).

---

Thank you for exploring TariffBot! 🌎🧠💼
