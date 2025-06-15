# HTS AI Agent – TariffBot

TariffBot is an AI-powered multi-tool agent designed to assist importers, analysts, and trade professionals with the U.S. Harmonized Tariff Schedule (HTS). It combines a **Retrieval-Augmented Generation (RAG)** agent for answering trade policy questions and a **Tariff Calculator** for computing duties, landed costs, and VAT. Built with open-source technologies like LangChain, DuckDB, Chroma, and Streamlit, TariffBot delivers clear, compliant, and factual responses grounded in official HTS documentation from [hts.usitc.gov](https://hts.usitc.gov).

## Features

- **RAG-Based Question Answering**: Answers trade policy and agreement questions (e.g., "What is the United States-Israel Free Trade Agreement?") using semantic search over the General Notes PDF, powered by LangChain and Chroma vector store.
- **HTS Duty Calculator**: Computes duties, landed costs, and VAT for given HTS codes, product costs, freight, insurance, quantity, and unit weight, supporting multiple duty formats (%, ¢/kg, $/unit).
- **Multi-CSV Ingestion**: Ingests multiple HTS CSV files for comprehensive tariff data coverage, with deduplication and indexing.
- **Country Code Enhancement**: Maps country codes to full names (e.g., AU → Australia) for accurate special rate parsing, including Free Trade Agreement (FTA) details.
- **Query Memory**: Retains conversation history using LangChain’s `ConversationBufferMemory` to avoid duplicate queries.
- **Export Options**: Generates PDF and Excel reports with detailed duty calculations.
- **Streamlit Interface**: User-friendly web interface for querying, calculating duties, and downloading results.
- **Query Routing**: Automatically classifies queries to route them to the appropriate tool (RAG, HTS Lookup, or Tariff Calculator).
- **Performance Optimization**: Leverages DuckDB indexing for fast HTS queries and Chroma for efficient vector retrieval.

## Prerequisites

- **Python**: 3.8 or higher
- **Required Python Packages**:
  ```bash
  pip install streamlit duckdb pandas reportlab openpyxl langchain langchain-chroma langchain-huggingface langchain-community langchain-ollama PyMuPDF
  ```
- **Data Files**:
  - HTS CSV files (e.g., `htsdata.csv`) from [hts.usitc.gov](https://hts.usitc.gov/export).
  - General Notes PDF from [hts.usitc.gov](https://hts.usitc.gov).
- **Ollama**: Install [Ollama](https://ollama.ai) and pull the Mistral 7B model:
  ```bash
  ollama pull mistral:7b
  ```
- **HuggingFace Cache**: Ensure the SentenceTransformer model (`all-MiniLM-L6-v2`) is cached at `C:\Users\sshas\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\c9745ed1d9f207416be6d2e6f8de32d1f16199bf`.

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ShashwatShlok/HTSAgent-Task-Shashwat-Shlok.git
   cd HTSAgent-Task-Shashwat-Shlok
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Or install the packages listed in the Prerequisites section.

3. **Prepare Data**:
   - Place HTS CSV files in `C:\Users\sshas\Downloads\HTSAgent\data\` (e.g., `htsdata.csv`).
   - Place `General Notes.pdf` in `C:\Users\sshas\Downloads\HTSAgent\data\`.
   - Verify file paths in `TariffBot_Implementation.py`:
     ```python
     GENERAL_NOTES_PATH = r"C:\Users\sshas\Downloads\HTSAgent\data\General Notes.pdf"
     CSV_PATHS = [
         r"C:\Users\sshas\Downloads\HTSAgent\data\htsdata.csv",
         r"C:\Users\sshas\Downloads\HTSAgent\data\htsdata (1).csv",
         r"C:\Users\sshas\Downloads\HTSAgent\data\htsdata (2).csv",
         r"C:\Users\sshas\Downloads\HTSAgent\data\htsdata (3).csv",
         r"C:\Users\sshas\Downloads\HTSAgent\data\htsdata (4).csv"
     ]
     ```

4. **Start Ollama Server**:
   ```bash
   ollama run mistral:7b
   ```
   Ensure the server is running at `http://localhost:11434`.

5. **Run the Application**:
   ```bash
   streamlit run TariffBot_Implementation.py
   ```
   This launches the Streamlit interface at `http://localhost:8501`.

## Usage

1. **Access the Streamlit App**:
   - Open `http://localhost:8501` in your browser.
   - Enter queries in the text input field on the main page.

2. **Sample Queries**:
   - **RAG Agent**:
     - "What is the United States-Israel Free Trade Agreement?"
     - "Can a product that exceeds its tariff-rate quota still qualify for duty-free entry under GSP or any FTA? Why or why not?"
     - "How is classification determined for an imported item that will be used as a part in manufacturing but isn’t itself a finished part?"
   - **Tariff Agent**:
     - "Given HTS code 0101.30.00.00, product cost of $10,000, 500 kg weight, and 5 units — what are all applicable duties?"
     - "What’s the HTS code for donkeys?"
     - "What are the applicable duty rates for female cattle?"

3. **View Results**:
   - For tariff calculations, view detailed steps in the expandable section.
   - Download results as PDF or Excel reports using the provided buttons.

4. **Error Handling**:
   - If an HTS code is invalid, TariffBot returns an error suggesting verification.
   - For ambiguous queries, it recommends reviewing the relevant HTS section.

## Project Structure

- `TariffBot_Implementation.py`: Core script implementing TariffBot, including data ingestion, RAG system, tariff calculator, and Streamlit interface.
- `data/`: Directory for HTS CSV files and `General Notes.pdf`.
- `chroma_db/`: Directory for Chroma vector store (generated during PDF ingestion).
- `hts.db`: DuckDB database for HTS tariff data.
- `requirements.txt`: List of required Python packages.

## Bonus Features

- **Multi-CSV Ingestion**: Supports multiple HTS CSV files with deduplication and indexing for comprehensive coverage.
- **Query Memory**: Uses LangChain’s `ConversationBufferMemory` to track query history and prevent duplicate processing.
- **Country Code Mapping**: Enhances special rate parsing by mapping codes (e.g., AU → Australia, IL → Israel) with FTA details.
- **VAT Calculation**: Applies a default 5% VAT, adjustable for country-specific rates.
- **Advanced Duty Parsing**: Handles %, ¢/kg, $/unit, and “Free” rates, with country-specific special rates.
- **Export Functionality**: Generates PDF and Excel reports with detailed calculation steps.
- **Error Handling**: Validates inputs, handles missing data, and provides clear error messages.
- **Performance Optimization**: Uses DuckDB for fast queries and Chroma for efficient vector retrieval.

## Troubleshooting

- **torch.classes Warning**: A known Streamlit-PyTorch issue. Mitigate by downgrading Streamlit:
  ```bash
  pip install streamlit==1.24.0
  ```
  Or run Python as an administrator to enable symlink support.
- **Missing Data Files**: Ensure `htsdata.csv` and `General Notes.pdf` are in `C:\Users\sshas\Downloads\HTSAgent\data\`. Download from [hts.usitc.gov](https://hts.usitc.gov) if missing.
- **Ollama Errors**: Verify the Ollama server is running and the Mistral 7B model is pulled.
- **Chroma Issues**: If RAG responses are vague, delete `chroma_db/` and re-run the app to rebuild the vector store.
- **DuckDB Errors**: Ensure CSV files are properly formatted and paths are correct in `CSV_PATHS`.

## Demo Video

A demo video showcasing TariffBot’s RAG and tariff calculator capabilities is available at [Insert Video Link]. It covers the sample queries:
- RAG: United States-Israel FTA, tariff-rate quotas, and classification rules.
- Tariff: Duty calculation for HTS 0101.30.00.00, HTS code for donkeys, and duty rates for female cattle.

## Submission Details

This project was developed by Shashwat Shlok for the HTS AI Agent coding task. For feedback or inquiries, contact:
- Email: sshashwat10@gmail.com
- GitHub: [ShashwatShlok/HTSAgent-Task-Shashwat-Shlok](https://github.com/ShashwatShlok/HTSAgent-Task-Shashwat-Shlok)
- Submission Email:
  - To: santosh@personaliz.ai
  - CC: aniketms42@gmail.com
  - Subject: HTS AI Task – Shashwat Shlok

## Acknowledgments

Built with open-source tools:
- [LangChain](https://github.com/langchain-ai/langchain) for RAG and agent framework.
- [Chroma](https://github.com/chroma-core/chroma) for vector storage.
- [DuckDB](https://duckdb.org) for efficient data storage and querying.
- [Streamlit](https://streamlit.io) for the web interface.
- [Ollama](https://ollama.ai) for local LLM inference.
- Data sourced from [hts.usitc.gov](https://hts.usitc.gov).

---

Thank you for exploring TariffBot! 🌎🧠💼