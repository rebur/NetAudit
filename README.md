Markdown# NetAudit: Automated Labeling-Quality Auditor for NIDS Datasets

**DSN 2025 Artifact** | Available • Functional • Reusable

This project contains complete, reproducible code from the paper **NetAudit: A Reasoning-Driven Automated Auditing Tool for Labeling Quality in Network Intrusion Detection Datasets**.

## Quick Start (< 5 minutes)

### 1. Clone the repository

```bash
git clone https://github.com/rebur/NetAudit.git

cd NetAudit

2. Install dependencies

Bashpip install -r requirements.txt

3. Configure your LLM (Required!)

Open enhanced_excel_traffic_analyzer_integrated.py, find the setup_llm function, and modify the following two lines:

Pythonapi_key="sk-your-real-key-here", # ← Enter your Tongyi Qianwen / OpenAI / Grok key

base_url="https://your-endpoint.com/v1" # ← Enter the compatibility address for the corresponding platform

Also, in all places where LLM is called, it should be written as:

Pythonmodel="your_model" # ← Change to the name of your actual model, for example:

# "qwen-turbo" / "qwen-plus" / "gpt-4o" / For examples like "claude-3-5-sonnet-20241022", we used qwen-turbo in our experiments, which offers the best cost-performance ratio. We recommend using the same model for the first reproduction.

4. One-Click Reproduction of Paper Results

Bash # Linux / macOS

bash run.sh

# Windows
run.bat
The script will automatically:

Load your CSV file

Perform a full three-tier audit

Output Final Annotated Review Report.jsonl

Run analyze_report_summary.py to automatically generate new error statistics

Reproduce Table 2 of the paper (483 new errors, 9 new defect categories)

Estimated total time: 12–15 hours (consistent with the paper)

File Structure
textNetAudit/

├── enhanced_excel_traffic_analyzer_integrated.py # Main program

├── analyze_report_summary.py # Novelty analysis

├── rag_data/ # 6 authoritative documents (2 papers + 4 RFCs)

├── vector_index/ # Pre-built FAISS index

├── requirements.txt

├── run.sh / run.bat

└── README.md



License: Apache-2.0
