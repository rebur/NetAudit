import pandas as pd
import os
import glob
import json
import re
import sys
import subprocess
from typing import List, Dict, Any, Optional
from datetime import datetime
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')


class LLMIntegration:
    """LLM Integration Classes - For intelligent analytics and RAG knowledge retrieval"""

    def __init__(self, api_key=None, base_url=None, index_path="vector_index"):
        self.llm_client = None
        self.embeddings_model = None
        self.vector_db = None
        self.rag_knowledge_base = []
        self.index_path = index_path
        self.is_vector_store_loaded = False
        self.setup_llm(api_key, base_url)

    def setup_llm(self, api_key=None, base_url=None):
        """Please note that you must use your own api_key!!!"""
        try:
            from openai import OpenAI
            self.llm_client = OpenAI(
                api_key="sk-your-key",  # Replace your api_key！！！
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # Replace your url！！！
            )
            return True
        except Exception as e:
            print(f"LLM initialization failed: {str(e)}")
            return False

    def setup_embeddings(self):
        """Configure the embedded model - Use the extracted model configuration"""
        try:
            from sentence_transformers import SentenceTransformer
            # Using an embedding model extracted from user code
            self.embeddings_model = SentenceTransformer('shibing624/text2vec-base-chinese')
            return True
        except Exception as e:
            print(f"Embedded model initialization failed: {str(e)}")
            return False

    def _smart_load_file(self, file_path: str) -> str:
        """Loader: Supports PDFs + Automatic Dependency Installation"""
        ext = os.path.splitext(file_path)[1].lower()

        # ====================== Dedicated processing for PDF files ======================
        if ext == '.pdf':
            try:
                # Delayed import + automatic installation completely avoid ImportError causing the entire program to crash.
                try:
                    from PyPDF2 import PdfReader
                except ImportError:
                    print("PyPDF2 is not installed; it is installing automatically (only once required)...")
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", "PyPDF2", "--quiet"
                    ])
                    from PyPDF2 import PdfReader

                text = ""
                with open(file_path, 'rb') as f:
                    reader = PdfReader(f)
                    page_count = len(reader.pages)
                    for i, page in enumerate(reader.pages):
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    print(f"PDF loaded successfully:{os.path.basename(file_path)}（{page_count}页，提取{text.count(' '):,}个字符）")
                return text

            except Exception as e:
                print(f"PDF failed to load (possibly due to scanning or corruption): {file_path} → {e}")
                return f"[PDF loading failed] Filename: {os.path.basename(file_path)}"

        # ====================== Plain text file processing ======================
        else:
            encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig', 'latin-1']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    print(f"Text loaded successfully: {os.path.basename(file_path)} ({encoding} encoding)")
                    return content
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"Text reading error: {file_path} → {e}")
                    break

            # As a last resort: force a read using binary mode (rarely triggered).
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                print(f"Forced loading successful (ignore garbled characters): {os.path.basename(file_path)}")
                return content
            except:
                print(f"All methods failed: {file_path}")
                return f"[File read failed.] {os.path.basename(file_path)}"

    def load_documents(self, knowledge_paths: List[str]) -> List[Dict]:
        """Load knowledge documents - Supports PDF + Super fault tolerance + Fixed bug where ext variable was undefined"""
        knowledge_docs = []
        seen_sources = set()

        for path in knowledge_paths:
            if not os.path.exists(path):
                print(f"Path does not exist: {path}")
                continue

            patterns = ["**/*.txt", "**/*.md", "**/*.json", "**/*.csv", "**/*.yaml", "**/*.yml", "**/*.pdf", "*.pdf"]
            files = []
            for p in patterns:
                files.extend(glob.glob(os.path.join(path, p), recursive=True))

            # Deduplication
            files = list(set(files))
            print(f"{len(files)} candidate files were scanned")

            for file_path in files:
                source_name = os.path.basename(file_path)
                if source_name in seen_sources:
                    continue
                seen_sources.add(source_name)

                file_ext = os.path.splitext(file_path)[1].lower()

                content = self._smart_load_file(file_path)
                if not content or not content.strip():
                    print(f"If the content is empty or loading failed, skip: {source_name}")
                    continue

                knowledge_docs.append({
                    'content': content.strip(),
                    'source': source_name,
                    'file_path': file_path,
                    'file_size': len(content),
                    'file_type': file_ext
                })
                print(f"Successfully loaded: {source_name} ({file_ext}, {len(content):,} characters)")

        print(f"A total of {len(knowledge_docs)} valid knowledge documents (including PDFs) were successfully loaded.")
        self.rag_knowledge_base = knowledge_docs
        return knowledge_docs

    def create_vector_store(self, documents: List[Dict]):
        """Creating a vector store - using FAISS and extracted embedding models"""
        if not self.embeddings_model:
            if not self.setup_embeddings():
                print("Embedded model setup failed; unable to create vector storage.")
                return None

        try:
            import faiss
            import numpy as np

            all_texts = []
            text_to_doc_mapping = []  # Mapping text snippets to the original document

            for doc_idx, doc in enumerate(documents):
                content = doc['content']
                # Smarter text segmentation
                sentences = re.split(r'[。！？\n；;]', content)
                for sentence in sentences:
                    clean_sentence = sentence.strip()
                    if len(clean_sentence) > 5:  # Lower the length threshold to capture more content.
                        all_texts.append(clean_sentence)
                        text_to_doc_mapping.append(doc_idx)  # Which document does the recorded text fragment belong to?

            print(f"Processed {len(all_texts)} text fragments from {len(documents)} documents.")

            if not all_texts:
                print("No valid text content available for processing.")
                return None

            # Generate Embedded
            print("Generate text embedding vectors...")
            embeddings = self.embeddings_model.encode(all_texts)
            dimension = embeddings.shape[1]

            # Create FAISS index
            self.vector_db = faiss.IndexFlatIP(dimension)
            self.vector_db.add(embeddings.astype('float32'))

            # Ensure the index directory exists
            if not os.path.exists(self.index_path):
                os.makedirs(self.index_path, exist_ok=True)
                print(f"Create an index directory: {self.index_path}")

            # Save vector index
            faiss.write_index(self.vector_db, os.path.join(self.index_path, "index.faiss"))

            # Save text mapping information
            index_data = {
                'texts': all_texts,
                'text_to_doc_mapping': text_to_doc_mapping,
                'documents': [{'source': doc['source'], 'file_path': doc['file_path']} for doc in documents],
                'embedding_dimension': dimension,
                'created_at': datetime.now().isoformat()
            }

            with open(os.path.join(self.index_path, "index_metadata.json"), 'w', encoding='utf-8') as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)

            self.is_vector_store_loaded = True
            print(f"Vector storage creation complete, containing {self.vector_db.ntotal} vectors.")
            print(f"The index has been saved to: {self.index_path}")
            return self.vector_db

        except ImportError as e:
            print(f"Missing necessary libraries: {str(e)}")
            print("Please install: pip install faiss-cpu sentence-transformers")
            return None
        except Exception as e:
            print(f"Vector storage creation failed: {str(e)}")
            return None

    def load_existing_vector_store(self):
        """Loading an existing vector storage"""
        try:
            import faiss
            import numpy as np

            if self.embeddings_model is None:
                if not self.setup_embeddings():
                    print("Embedded model setup failed")
                    return False

            index_file = os.path.join(self.index_path, "index.faiss")
            metadata_file = os.path.join(self.index_path, "index_metadata.json")

            if os.path.exists(index_file) and os.path.exists(metadata_file):
                print(f"Existing vector storage found, loading...")
                self.vector_db = faiss.read_index(index_file)

                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                # Rebuild document mapping
                self.rag_knowledge_base = []
                for doc_info in metadata.get('documents', []):
                    self.rag_knowledge_base.append({
                        'source': doc_info['source'],
                        'file_path': doc_info['file_path'],
                        'content': 'Loaded into the vector library'  # Do not reload the full content
                    })

                self.is_vector_store_loaded = True
                print(f"Successfully loaded the existing vector store, containing {self.vector_db.ntotal} vectors.")
                print(f"Link {len(self.rag_knowledge_base)} knowledge documents")
                return True
            else:
                print(f"No complete vector storage file found")
                if os.path.exists(self.index_path):
                    print(f"Index contents: {os.listdir(self.index_path)}")
                return False

        except ImportError as e:
            print(f"Missing necessary libraries: {str(e)}")
            return False
        except Exception as e:
            print(f"Loading vector storage failed: {str(e)}")
            return False

    def build_rag_knowledge_base(self, knowledge_paths: List[str]):
        """Building a RAG knowledge base - Optimized version, prioritizing loading existing vector storage."""
        print("Starting to build the RAG knowledge base...")

        # First, try loading the existing vector storage.
        if self.load_existing_vector_store():
            print("Use existing vector storage")
            return self.rag_knowledge_base

        print("No existing vector storage found, starting to build a new one...")

        # 加载文档
        knowledge_docs = self.load_documents(knowledge_paths)

        if not knowledge_docs:
            print("No available knowledge documentation found")
            # Create sample knowledge document
            self._create_sample_knowledge_base()
            knowledge_docs = self.rag_knowledge_base

        if knowledge_docs:
            # Create vector storage
            vector_store = self.create_vector_store(knowledge_docs)
            if vector_store:
                print("RAG knowledge base construction completed.")
                return knowledge_docs
            else:
                print("Vector storage creation failed")
        else:
            print("No knowledge documentation available")

        return []

    def _create_sample_knowledge_base(self):
        """Create a sample knowledge base - for use when no document is found."""
        sample_docs = [
            {
                'content': """Network traffic analysis basics：
1. The TCP protocol conforms to the RFC 793 standard and requires a complete three-way handshake and four-way handshake.
2. DDoS attack characteristics: high concurrency connections, a large number of data packets in a short period of time.
3. Port scanning characteristics: rapid and continuous access to multiple ports.
4. Normal traffic characteristics: protocol compliance, stable traffic, and a complete connection lifecycle.""",
                'source': 'Cybersecurity Fundamentals.md',
                'file_path': 'Built-in knowledge/cybersecurity basics.md',
                'file_size': 200,
                'file_type': '.md'
            },
            {
                'content': """Traffic data quality testing standards：
1. Data Integrity: All necessary fields should exist and be valid.
2. Protocol Compliance: Traffic should comply with relevant protocol standards.
3. Time Continuity: Timestamps should be continuous and reasonable.
4. Reasonable Numerical Range: Port numbers, packet sizes, etc., should be within valid ranges.""",
                'source': 'Data quality standards.txt',
                'file_path': 'Built-in knowledge/data quality standards.txt',
                'file_size': 150,
                'file_type': '.txt'
            }
        ]

        self.rag_knowledge_base = sample_docs
        print("A sample knowledge base has been created.")

    def retrieve_relevant_knowledge(self, query: str, k: int = 3) -> List[Dict]:
        """Search for relevant knowledge - Optimized version"""
        if not self.vector_db or not self.embeddings_model or not self.is_vector_store_loaded:
            print("Vector storage is not ready; retrieval is not possible.")
            return []

        try:
            import faiss
            import numpy as np

            # Generate query embeddings
            query_embedding = self.embeddings_model.encode([query])

            # Search similar vectors
            similarities, indices = self.vector_db.search(query_embedding.astype('float32'), k)

            relevant_docs = []
            for i, idx in enumerate(indices[0]):
                if idx < self.vector_db.ntotal:
                    # Load detailed document information
                    metadata_file = os.path.join(self.index_path, "index_metadata.json")
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)

                        if idx < len(metadata.get('texts', [])):
                            text_content = metadata['texts'][idx]
                            doc_idx = metadata['text_to_doc_mapping'][idx]
                            doc_info = metadata['documents'][doc_idx]

                            relevant_docs.append({
                                'content': text_content,
                                'source': doc_info['source'],
                                'file_path': doc_info['file_path'],
                                'similarity': float(similarities[0][i]),
                                'rank': i + 1
                            })

            print(f"{len(relevant_docs)} relevant documents were retrieved.")
            return relevant_docs

        except Exception as e:
            print(f"Knowledge retrieval failed: {str(e)}")
            return []

    def call_llm_analysis(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> str:
        """Calling LLM for analysis - using extracted model configuration"""
        if not self.llm_client:
            return "LLM not initialized"

        try:
            response = self.llm_client.chat.completions.create(
                model="your_model",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"LLM call failed: {str(e)}"

    def hierarchical_analysis_prompt(self, traffic_record: Dict[str, Any], context: str) -> str:
        """Building Hierarchical Analysis of Prompt Keywords - Integrating New User-Provided Prompt Keywords"""
        traffic_desc = self.convert_to_natural_language(traffic_record)

        prompt = f"""You are a cybersecurity expert responsible for systematically performing hierarchical error detection on network traffic datasets.

Relevant background information:
{context}

Traffic records to be analyzed:
{traffic_desc}

Please conduct a systematic analysis according to the following three levels:

## Layer 1: Data Acquisition Layer Detection
### TCP stream construction verification (RFC compliance)
- Check if the TCP stream construction conforms to RFC 793 standard
- Verify that the bidirectional FIN mechanism is implemented correctly
- Detect the existence of invalid "TCP appendix streams"
- Analyze whether the stream termination mechanism is compliant

### Feature extraction tool problem diagnosis  
- Assess the compliance of feature extraction tools (such as CICFlowMeter)
- Check for violations of RFC standards in feature extraction
- Verify the correctness of tool configuration parameters

## Level 2: Labeling Strategy Layer Detection
### Labeling Omission Identification (Time Window Analysis)
- Analyze the integrity of traffic labeling within the time window
- Detect whether there are labeling interruptions in continuous attack flows
- Verify the temporal consistency of labels

### Tag Contamination Detection (Protocol Behavior Verification)
- Check if normal protocol behavior is mislabeled as an attack
- Verify if idle traffic is incorrectly labeled
- Analyze whether protocol-compliant traffic has been contaminated

## Layer 3: Attack Validity Detection
### Tool Configuration Correctness Verification
- Verify the rationality of the attack tool parameter configuration
- Check whether the attack payload is effectively generated
- Analyze whether the attack execution conditions are met

### Attack Execution Success Rate Assessment
- Assess whether the attack successfully reached its target
- Analyze whether the attack payload was effectively executed
- Verify whether the attack flow produced the expected results

Please provide the following for each detection item:
1. Detection result (Pass/Fail/Suspicious)
2. Detailed description of the problem found
3. Problem severity (High/Medium/Low)
4. Remediation suggestions

Output JSON format:
{{
    "hierarchical_analysis": {{
        "level1_data_collection": {{
            "tcp_flow_validation": {{
                "result": "Passed/Failed/Suspicious",
                "issues": ["Problem Description 1", Problem Description 2"],
                "severity": "High/Medium/Low",
                "suggestions": ["Repair suggestion 1", "Repair suggestion 2""]
            }},
            "feature_extraction_diagnosis": {{
                "result": "Passed/Failed/Suspicious", 
                "issues": [],
                "severity": "High/Medium/Low",
                "suggestions": []
            }}
        }},
        "level2_labeling_strategy": {{
            "labeling_omission_detection": {{
                "result": "Passed/Failed/Suspicious",
                "issues": [],
                "severity": "High/Medium/Low",
                "suggestions": []
            }},
            "label_contamination_detection": {{
                "result": "Passed/Failed/Suspicious",
                "issues": [],
                "severity": "High/Medium/Low",
                "suggestions": []
            }}
        }},
        "level3_attack_effectiveness": {{
            "tool_configuration_validation": {{
                "result": "Passed/Failed/Suspicious",
                "issues": [],
                "severity": "High/Medium/Low",
                "suggestions": []
            }},
            "attack_success_assessment": {{
                "result": "Passed/Failed/Suspicious",
                "issues": [],
                "severity": "High/Medium/Low",
                "suggestions": []
            }}
        }}
    }},
    "overall_verdict": "Reasonable/Unreasonable",
    "comprehensive_reasoning": "Overall Analysis Reasons",
    "priority_recommendations": ["Priority Repair Recommendation 1", "Priority Repair Recommendation 2""]
}}

Please output the analysis results directly in JSON format:"""

        return prompt

    def perform_hierarchical_analysis(self, traffic_record: Dict[str, Any]) -> Dict[str, Any]:
        """Perform hierarchical analysis - Integrate user-provided methods"""
        if self.llm_client is None:
            return {"error": "LLM not initialized"}

        try:
            # Search for relevant knowledge
            query = f"Analyze network traffic records: Protocol {traffic_record.get('protocol', 'unknown')}, Label {traffic_record.get('label', 'unknown')}"
            relevant_knowledge = self.retrieve_relevant_knowledge(query, k=3)

            context = "Relevant background information:\n"
            if relevant_knowledge:
                for i, doc in enumerate(relevant_knowledge):
                    context += f"【Related documents {i + 1} - {doc['source']} (Similarity: {doc['similarity']:.3f})】\n"
                    context += f"{doc['content']}\n\n"
            else:
                context += "There is currently no relevant background information, and the analysis will be based on general rules.\n"

            # Constructing hierarchical analysis prompts
            prompt = self.hierarchical_analysis_prompt(traffic_record, context)

            # Calling LLM
            response = self.llm_client.chat.completions.create(
                model="your_model",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            llm_output = response.choices[0].message.content

            # Analytical results
            analysis_result = self.parse_llm_response(llm_output)

            return {
                "original_record": traffic_record,
                "hierarchical_analysis": analysis_result,
                "llm_output": llm_output,
                "rag_context_used": bool(relevant_knowledge),
                "relevant_docs_count": len(relevant_knowledge)
            }

        except Exception as e:
            print(f"Error during stratification analysis: {str(e)}")
            return {
                "original_record": traffic_record,
                "error": str(e),
                "llm_output": "Call failed"
            }

    def convert_to_natural_language(self, traffic_record: Dict[str, Any]) -> str:
        """Convert traffic records into natural language descriptions"""
        description = "Network traffic log analysis request:\n"
        description += f"- Source IP: {traffic_record.get('src_ip', 'unknown')}\n"
        description += f"- Target IP: {traffic_record.get('dst_ip', 'unknown')}\n"
        description += f"- Protocol: {traffic_record.get('protocol', 'unknown')}\n"

        if 'src_port' in traffic_record and 'dst_port' in traffic_record:
            description += f"- port: {traffic_record['src_port']} → {traffic_record['dst_port']}\n"

        if 'payload_length' in traffic_record:
            description += f"- Load length: {traffic_record['payload_length']} byte\n"

        if 'flow_duration' in traffic_record:
            description += f"- Flow duration: {traffic_record['flow_duration']} microseconds\n"

        description += f"- Current tag: {traffic_record.get('label', 'unknown')}\n"
        description += "\nPlease analyze whether the tags for this traffic record are reasonable and provide detailed reasons."

        return description

    def parse_llm_response(self, llm_output: str) -> Dict[str, Any]:
        """Analyzing LLM responses"""
        try:
            if llm_output.strip().startswith('{') and llm_output.strip().endswith('}'):
                return json.loads(llm_output.strip())

            json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

            return self.extract_info_from_text(llm_output)

        except json.JSONDecodeError:
            return self.extract_info_from_text(llm_output)
        except Exception as e:
            return {"verdict": "Parsing error", "reasoning": f"Parsing failed: {str(e)}", "suggested_label": "unknown"}

    def extract_info_from_text(self, text: str) -> Dict[str, Any]:
        """Extracting analytical information from text"""
        verdict = "unknown"
        reasoning = text
        suggested_label = "unknown"

        if "reasonable" in text and "unreasonable" not in text:
            verdict = "reasonable"
        elif "unreasonable" in text:
            verdict = "unreasonable"

        return {
            "verdict": verdict,
            "reasoning": reasoning,
            "suggested_label": suggested_label
        }


class ProtocolBasedDeduction:
    """Principle-based deductive reasoning - discovering data anomalies through protocol standards"""

    def __init__(self, llm_integration: LLMIntegration = None):
        self.protocol_knowledge = self._load_protocol_knowledge()
        self.llm = llm_integration

    def _load_protocol_knowledge(self) -> Dict[str, Any]:
        """Loading Protocol Principles Knowledge Base"""
        return {
            "TCP": {
                "connection_establishment": {
                    "standard": "RFC 793 Three-Way Handshake: SYN → SYN-ACK → ACK",
                    "expected_behavior": "The complete three-way handshake process",
                    "common_errors": ["Half-open connection", "SYN flood attack", "Incomplete handshake"]
                },
                "connection_termination": {
                    "standard": "RFC 793 Four-way handshake: FIN → ACK → FIN → ACK",
                    "expected_behavior": "Graceful closure achieved through bidirectional FIN packet exchange",
                    "common_errors": ["Unilateral FIN Termination", "RST violent termination", "Connecting suspension"]
                }
            }
        }

    def analyze_with_llm(self, traffic_record: Dict[str, Any], context: str = "") -> Dict[str, Any]:
        """Using LLM for protocol compliance analysis"""
        if not self.llm:
            return {"error": "LLM not initialized"}

        prompt = f"""As a protocol analysis expert, please analyze the protocol compliance of the following traffic records:

Background information:
{context}

Traffic records:
{self._describe_traffic_for_llm(traffic_record)}

Please focus on analyzing TCP protocol compliance (RFC 793) and output in JSON format: {{"protocol_analysis": {{"result": "合规/不合规", "issues": [], "recommendations": []}}}}"""

        response = self.llm.call_llm_analysis(prompt)
        return self.llm.parse_llm_response(response)

    def _describe_traffic_for_llm(self, record: Dict[str, Any]) -> str:
        """Describe traffic records for LLM"""
        desc = f"protocol: {record.get('protocol', 'unknown')}\n"
        desc += f"Source port: {record.get('src_port', 'unknown')} → Target port: {record.get('dst_port', 'unknown')}\n"
        desc += f"Flow duration: {record.get('flow_duration', 'unknown')}microseconds\n"

        # TCP flags
        flags_info = []
        for flag in ['fin', 'syn', 'rst', 'psh', 'ack', 'urg']:
            count = record.get(f'{flag}_flag_count', 0)
            if count > 0:
                flags_info.append(f"{flag.upper()}:{count}")

        if flags_info:
            desc += f"TCP flags: {', '.join(flags_info)}\n"

        return desc


class CausalChainAnalyzer:
    """Causal chain analysis of attack effectiveness"""

    def __init__(self, llm_integration: LLMIntegration = None):
        self.llm = llm_integration

    def analyze_with_llm(self, traffic_record: Dict[str, Any], context: str = "") -> Dict[str, Any]:
        """Attack effectiveness analysis using LLM"""
        if not self.llm:
            return {"error": "LLM not initialized"}

        attack_type = self._infer_attack_type(traffic_record.get('label', ''))

        prompt = f"""As an attack analysis expert, please evaluate the attack effectiveness of the following traffic records:

Background information:
{context}

Attack traffic logs (type: {attack_type}):
{self._describe_attack_traffic(traffic_record)}

Output JSON format: {{"attack_effectiveness": {{"score": 0-1, "verdict": "有效/无效", "issues": [], "improvements": []}}}}"""

        response = self.llm.call_llm_analysis(prompt)
        return self.llm.parse_llm_response(response)

    def _infer_attack_type(self, label: str) -> str:
        """Precisely mapping the 14 attack categories of CIC-IDS2017"""
        label_lower = label.lower().replace(' ', '').replace('-', '').replace('_', '')

        mapping = {
            # Tuesday
            'ftppatator': 'FTP-Patator',
            'sshpatator': 'SSH-Patator',

            # Wednesday
            'heartbleed': 'Heartbleed',
            'slowloris': 'DoS slowloris',
            'slowhttptest': 'DoS Slowhttptest',
            'hulk': 'DoS Hulk',
            'goldeneye': 'DoS GoldenEye',

            # Thursday
            'infiltration': 'Infiltration',
            'bruteforce': 'Web Attack–Brute Force',
            'xss': 'Web Attack–XSS',
            'sqlinjection': 'Web Attack–SQL Injection',

            # Friday
            'bot': 'Bot',
            'botnet': 'Bot',
            'ddos': 'DDoS',
            'portscan': 'PortScan',
            'port scanning': 'PortScan',
        }

        for key, standard_name in mapping.items():
            if key in label_lower:
                return standard_name

        if 'sql' in label_lower and 'injection' in label_lower:
            return 'Web Attack–SQL Injection'
        if 'brute' in label_lower and 'force' in label_lower:
            return 'Web Attack–Brute Force'

        return 'Benign' if 'benign' in label_lower or label_lower == '' else 'Unknown'

    def _describe_attack_traffic(self, record: Dict[str, Any]) -> str:
        """Generate high information density attack traffic summaries for LLM"""
        r = record

        desc = f"### Attack Traffic Summary (Label: {r.get('label', 'Unknown')} ###\n"

        # 1. Basic Information
        desc += f"• Source: {r.get('src_ip', '?')}:{r.get('src_port', '?')} → "
        desc += f"{r.get('dst_ip', '?')}:{r.get('dst_port', '?')}\n"

        # 2. Time and duration (crucial! Exposes timeout artifact)
        duration_sec = r.get('flow_duration', 0) / 1e6
        desc += f"• Duration: {duration_sec:.6f}s ({r.get('flow_duration', 0)} μs)\n"

        # 3. Traffic intensity (exposing zero-payload / handshake-only)
        fwd_pkt = r.get('total_fwd_packets', 0)
        bwd_pkt = r.get('total_bwd_packets', 0)
        fwd_bytes = r.get('total_fwd_bytes', 0)
        bwd_bytes = r.get('total_bwd_bytes', 0)
        desc += f"• Packets: {fwd_pkt} → / {bwd_pkt} ← | "
        desc += f"Bytes: {fwd_bytes} → / {bwd_bytes} ← "
        desc += f"(Payload bytes: {r.get('fwd_payload_bytes', 0)} → / {r.get('bwd_payload_bytes', 0)} ←)\n"

        # 4. TCP flags (exposing handshake-only / RST-terminated)
        flags = []
        for f in ['syn', 'fin', 'rst', 'psh', 'ack', 'urg', 'ece', 'cwr']:
            if r.get(f'fwd_{f}_flag_count', 0) > 0 or r.get(f'bwd_{f}_flag_count', 0) > 0:
                flags.append(f.upper())
        desc += f"• TCP Flags observed: {', '.join(flags) if flags else 'None'}\n"

        # 5. Protocols and ports (exposing incorrect ports during attacks)
        proto = r.get('protocol', '?')
        desc += f"• Protocol: {proto} | "
        desc += f"Source Port: {r.get('src_port', '?')} | "
        desc += f"Target Port: {r.get('dst_port', '?')}\n"

        # 6. Key statistics (exposure timeout, sub-stream, window, etc. artifacts)
        desc += f"• Avg packet size: Fwd/Bwd: {r.get('fwd_avg_packet_size', 0):.1f} / {r.get('bwd_avg_packet_size', 0):.1f}\n"
        desc += f"• Active/Idle time: {r.get('active_mean', 0) / 1e6:.3f}s / {r.get('idle_mean', 0) / 1e6:.3f}s\n"
        desc += f"• Subflows: {r.get('subflow_fwd_packets', 0)} → / {r.get('subflow_bwd_packets', 0)} ←\n"

        # 7. Window scaling (exposing GoldenEye, etc.)
        desc += f"• Init Win bytes: {r.get('init_win_bytes_forward', -1)} → / {r.get('init_win_bytes_backward', -1)} ←\n"

        return desc.strip()


class EnhancedExcelTrafficAnalyzerIntegrated:
    """Integrated Excel Traffic Analyzer - Combining three methodologies and LLM+RAG"""

    def __init__(self, rag_knowledge_paths: List[str] = None, index_path: str = "vector_index"):

        self.llm_integration = LLMIntegration(index_path=index_path)
        self.rag_knowledge_paths = rag_knowledge_paths or ["rag_data"]
        self.index_path = index_path

        self.protocol_analyzer = ProtocolBasedDeduction(self.llm_integration)
        self.causal_analyzer = CausalChainAnalyzer(self.llm_integration)

        self._initialize_rag_knowledge_base()

    def _initialize_rag_knowledge_base(self):
        """Initialize RAG knowledge base"""
        print("Initialize the RAG knowledge base...")

        knowledge_docs = self.llm_integration.build_rag_knowledge_base(self.rag_knowledge_paths)

        if knowledge_docs:
            print(f"The RAG knowledge base initialization is complete, containing {len(knowledge_docs)} documents.")
            if self.llm_integration.is_vector_store_loaded:
                print("Vector storage is ready, and intelligent retrieval can be performed.")
            else:
                print("Vector storage is not ready; a no-RAG mode will be used.")
        else:
            print("RAG knowledge base initialization failed; no RAG mode will be used.")

    def load_excel_data(self, file_path: str, sheet_name: Optional[str] = 0) -> pd.DataFrame:
        """Load Excel data - supports .xls and .xlsx formats"""
        try:
            if not os.path.exists(file_path):
                print(f"The file does not exist.: {file_path}")
                return pd.DataFrame()

            file_ext = os.path.splitext(file_path)[1].lower()

            if file_ext == '.xls':
                df = pd.read_excel(file_path, sheet_name=sheet_name, engine='xlrd')
            elif file_ext == '.xlsx':
                df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
            else:
                print(f"Unsupported Excel formats: {file_ext}")
                return pd.DataFrame()

            if isinstance(df, dict):
                if df:
                    first_sheet = list(df.keys())[0]
                    df = df[first_sheet]
                    print(f"Multiple worksheets were detected; the first worksheet is used. {first_sheet}")
                else:
                    print("The Excel file does not contain any worksheets.")
                    return pd.DataFrame()

            print(f"Successfully loaded Excel file: {file_path}")
            print(f"Data Shape: {df.shape}")
            return df

        except Exception as e:
            print(f"Failed to load Excel file: {str(e)}")
            return pd.DataFrame()

    def enhanced_column_mapping(self, df_columns: List[str]) -> Dict[str, str]:
        """Enhanced column name mapping"""
        standard_columns = {
            'flow_id': ['flow id', 'flowid', 'flow_id', 'Flow_ID', 'FLOW_ID', 'flowidnumber', 'FlowIdNumber'],
            'timestamp': ['timestamp', 'time', 'Timestamp', 'TIME', 'TIMESTAMP', 'timestamputc', 'TimestampUTC'],
            'src_ip': ['srcip', 'sourceip', 'source_ip', 'src ip', 'source ip', 'Source_IP', 'Source IP', 'SOURCE_IP',
                       'sourceipaddress', 'SourceIpAddress'],
            'dst_ip': ['dstip', 'destinationip', 'destination_ip', 'dst ip', 'dest ip', 'Dest_IP', 'Dest IP', 'DEST_IP',
                       'destinationipaddress', 'DestinationIpAddress'],
            'src_port': ['srcport', 'sourceport', 'source_port', 'src port', 'source port', 'Source_Port',
                         'Source Port', 'SOURCE_PORT', 'sourceportnumber', 'SourcePortNumber'],
            'dst_port': ['dstport', 'destinationport', 'destination_port', 'dst port', 'dest port', 'Dest_Port',
                         'Dest Port', 'DEST_PORT', 'destinationportnumber', 'DestinationPortNumber'],
            'protocol': ['protocol', 'proto', 'Protocol', 'PROTOCOL', 'protocoltype', 'ProtocolType'],
            'payload_length': ['payload', 'payloadlength', 'payload_length', 'length', 'Flow_Bytes', 'flow_bytes',
                               'Flow Bytes', 'FLOW_BYTES', 'bytes', 'Bytes', 'totalbytes', 'TotalBytes'],
            'flow_duration': ['duration', 'flowduration', 'flow_duration', 'time', 'Flow_Duration', 'Flow Duration',
                              'FLOW_DURATION', 'durationms', 'DurationMs'],
            'total_fwd_packets': ['total fwd packets', 'totalfwdpackets', 'total_fwd_packets', 'Total_Fwd_Packets',
                                  'TOTAL_FWD_PACKETS', 'totalforwardpackets', 'TotalForwardPackets'],
            'total_bwd_packets': ['total backward packets', 'totalbackwardpackets', 'total_backward_packets',
                                  'Total_Backward_Packets', 'TOTAL_BACKWARD_PACKETS', 'totalbackwardpackets',
                                  'TotalBackwardPackets'],
            'total_fwd_length': ['total length of fwd packets', 'totallengthoffwdpackets',
                                 'total_length_of_fwd_packets', 'Total_Length_of_Fwd_Packets',
                                 'TOTAL_LENGTH_OF_FWD_PACKETS', 'totalfwdlength', 'TotalFwdLength'],
            'total_bwd_length': ['total length of bwd packets', 'totallengthofbwdpackets',
                                 'total_length_of_bwd_packets', 'Total_Length_of_Bwd_Packets',
                                 'TOTAL_LENGTH_OF_BWD_PACKETS', 'totalbwdlength', 'TotalBwdLength'],
            'fwd_packet_length_max': ['fwd packet length max', 'fwdpacketlengthmax', 'fwd_packet_length_max',
                                      'Fwd_Packet_Length_Max', 'FWD_PACKET_LENGTH_MAX', 'fwdpacketmax', 'FwdPacketMax'],
            'fwd_packet_length_min': ['fwd packet length min', 'fwdpacketlengthmin', 'fwd_packet_length_min',
                                      'Fwd_Packet_Length_Min', 'FWD_PACKET_LENGTH_MIN', 'fwdpacketmin', 'FwdPacketMin'],
            'fwd_packet_length_mean': ['fwd packet length mean', 'fwdpacketlengthmean', 'fwd_packet_length_mean',
                                       'Fwd_Packet_Length_Mean', 'FWD_PACKET_LENGTH_MEAN', 'fwdpacketmean',
                                       'FwdPacketMean'],
            'fwd_packet_length_std': ['fwd packet length std', 'fwdpacketlengthstd', 'fwd_packet_length_std',
                                      'Fwd_Packet_Length_Std', 'FWD_PACKET_LENGTH_STD', 'fwdpacketstd', 'FwdPacketStd'],
            'bwd_packet_length_max': ['bwd packet length max', 'bwdpacketlengthmax', 'bwd_packet_length_max',
                                      'Bwd_Packet_Length_Max', 'BWD_PACKET_LENGTH_MAX', 'bwdpacketmax', 'BwdPacketMax'],
            'bwd_packet_length_min': ['bwd packet length min', 'bwdpacketlengthmin', 'bwd_packet_length_min',
                                      'Bwd_Packet_Length_Min', 'BWD_PACKET_LENGTH_MIN', 'bwdpacketmin', 'BwdPacketMin'],
            'bwd_packet_length_mean': ['bwd packet length mean', 'bwdpacketlengthmean', 'bwd_packet_length_mean',
                                       'Bwd_Packet_Length_Mean', 'BWD_PACKET_LENGTH_MEAN', 'bwdpacketmean',
                                       'BwdPacketMean'],
            'bwd_packet_length_std': ['bwd packet length std', 'bwdpacketlengthstd', 'bwd_packet_length_std',
                                      'Bwd_Packet_Length_Std', 'BWD_PACKET_LENGTH_STD', 'bwdpacketstd', 'BwdPacketStd'],
            'flow_bytes_per_sec': ['flow bytes/s', 'flowbytes/s', 'flow_bytes/s', 'Flow_Bytes/s', 'FLOW_BYTES/S',
                                   'flowbytespersec', 'FlowBytesPerSec'],
            'flow_packets_per_sec': ['flow packets/s', 'flowpackets/s', 'flow_packets/s', 'Flow_Packets/s',
                                     'FLOW_PACKETS/S', 'flowpacketspersec', 'FlowPacketsPerSec'],
            'flow_iat_mean': ['flow iat mean', 'flowiatmean', 'flow_iat_mean', 'Flow_IAT_Mean', 'FLOW_IAT_MEAN',
                              'flowiatmean', 'FlowIATMean'],
            'flow_iat_std': ['flow iat std', 'flowiatstd', 'flow_iat_std', 'Flow_IAT_Std', 'FLOW_IAT_STD', 'flowiatstd',
                             'FlowIATStd'],
            'flow_iat_max': ['flow iat max', 'flowiatmax', 'flow_iat_max', 'Flow_IAT_Max', 'FLOW_IAT_MAX', 'flowiatmax',
                             'FlowIATMax'],
            'flow_iat_min': ['flow iat min', 'flowiatmin', 'flow_iat_min', 'Flow_IAT_Min', 'FLOW_IAT_MIN', 'flowiatmin',
                             'FlowIATMin'],
            'fwd_iat_total': ['fwd iat total', 'fwdiattotal', 'fwd_iat_total', 'Fwd_IAT_Total', 'FWD_IAT_TOTAL',
                              'fwdiattotal', 'FwdIATTotal'],
            'fwd_iat_mean': ['fwd iat mean', 'fwdiattmean', 'fwd_iat_mean', 'Fwd_IAT_Mean', 'FWD_IAT_MEAN',
                             'fwdiattmean', 'FwdIATMean'],
            'fwd_iat_std': ['fwd iat std', 'fwdiattstd', 'fwd_iat_std', 'Fwd_IAT_Std', 'FWD_IAT_STD', 'fwdiattstd',
                            'FwdIATStd'],
            'fwd_iat_max': ['fwd iat max', 'fwdiattmax', 'fwd_iat_max', 'Fwd_IAT_Max', 'FWD_IAT_MAX', 'fwdiattmax',
                            'FwdIATMax'],
            'fwd_iat_min': ['fwd iat min', 'fwdiattmin', 'fwd_iat_min', 'Fwd_IAT_Min', 'FWD_IAT_MIN', 'fwdiattmin',
                            'FwdIATMin'],
            'bwd_iat_total': ['bwd iat total', 'bwdiattotal', 'bwd_iat_total', 'Bwd_IAT_Total', 'BWD_IAT_TOTAL',
                              'bwdiattotal', 'BwdIATTotal'],
            'bwd_iat_mean': ['bwd iat mean', 'bwdiattmean', 'bwd_iat_mean', 'Bwd_IAT_Mean', 'BWD_IAT_MEAN',
                             'bwdiattmean', 'BwdIATMean'],
            'bwd_iat_std': ['bwd iat std', 'bwdiattstd', 'bwd_iat_std', 'Bwd_IAT_Std', 'BWD_IAT_STD', 'bwdiattstd',
                            'BwdIATStd'],
            'bwd_iat_max': ['bwd iat max', 'bwdiattmax', 'bwd_iat_max', 'Bwd_IAT_Max', 'BWD_IAT_MAX', 'bwdiattmax',
                            'BwdIATMax'],
            'bwd_iat_min': ['bwd iat min', 'bwdiattmin', 'bwd_iat_min', 'Bwd_IAT_Min', 'BWD_IAT_MIN', 'bwdiattmin',
                            'BwdIATMin'],
            'fwd_psh_flags': ['fwd psh flags', 'fwdpshflags', 'fwd_psh_flags', 'Fwd_PSH_Flags', 'FWD_PSH_FLAGS',
                              'fwdpshflags', 'FwdPshFlags'],
            'bwd_psh_flags': ['bwd psh flags', 'bwdpshflags', 'bwd_psh_flags', 'Bwd_PSH_Flags', 'BWD_PSH_FLAGS',
                              'bwdpshflags', 'BwdPshFlags'],
            'fwd_urg_flags': ['fwd urg flags', 'fwdugflags', 'fwd_urg_flags', 'Fwd_URG_Flags', 'FWD_URG_FLAGS',
                              'fwdugflags', 'FwdUrgFlags'],
            'bwd_urg_flags': ['bwd urg flags', 'bwdugflags', 'bwd_urg_flags', 'Bwd_URG_Flags', 'BWD_URG_FLAGS',
                              'bwdugflags', 'BwdUrgFlags'],
            'fwd_header_length': ['fwd header length', 'fwdheaderlength', 'fwd_header_length', 'Fwd_Header_Length',
                                  'FWD_HEADER_LENGTH', 'fwdheaderlength', 'FwdHeaderLength'],
            'bwd_header_length': ['bwd header length', 'bwdheaderlength', 'bwd_header_length', 'Bwd_Header_Length',
                                  'BWD_HEADER_LENGTH', 'bwdheaderlength', 'BwdHeaderLength'],
            'fwd_packets_per_sec': ['fwd packets/s', 'fwdpackets/s', 'fwd_packets/s', 'Fwd_Packets/s', 'FWD_PACKETS/S',
                                    'fwdpacketspersec', 'FwdPacketsPerSec'],
            'bwd_packets_per_sec': ['bwd packets/s', 'bwdpackets/s', 'bwd_packets/s', 'Bwd_Packets/s', 'BWD_PACKETS/S',
                                    'bwdpacketspersec', 'BwdPacketsPerSec'],
            'min_packet_length': ['min packet length', 'minpacketlength', 'min_packet_length', 'Min_Packet_Length',
                                  'MIN_PACKET_LENGTH', 'minpacketlength', 'MinPacketLength'],
            'max_packet_length': ['max packet length', 'maxpacketlength', 'max_packet_length', 'Max_Packet_Length',
                                  'MAX_PACKET_LENGTH', 'maxpacketlength', 'MaxPacketLength'],
            'packet_length_mean': ['packet length mean', 'packetlengthmean', 'packet_length_mean', 'Packet_Length_Mean',
                                   'PACKET_LENGTH_MEAN', 'packetlengthmean', 'PacketLengthMean'],
            'packet_length_std': ['packet length std', 'packetlengthstd', 'packet_length_std', 'Packet_Length_Std',
                                  'PACKET_LENGTH_STD', 'packetlengthstd', 'PacketLengthStd'],
            'packet_length_variance': ['packet length variance', 'packetlengthvariance', 'packet_length_variance',
                                       'Packet_Length_Variance', 'PACKET_LENGTH_VARIANCE', 'packetlengthvariance',
                                       'PacketLengthVariance'],
            'fin_flag_count': ['fin flag count', 'finflagcount', 'fin_flag_count', 'FIN_Flag_Count', 'FIN_FLAG_COUNT',
                               'finflagcount', 'FinFlagCount'],
            'syn_flag_count': ['syn flag count', 'synflagcount', 'syn_flag_count', 'SYN_Flag_Count', 'SYN_FLAG_COUNT',
                               'synflagcount', 'SynFlagCount'],
            'rst_flag_count': ['rst flag count', 'rstflagcount', 'rst_flag_count', 'RST_Flag_Count', 'RST_FLAG_COUNT',
                               'rstflagcount', 'RstFlagCount'],
            'psh_flag_count': ['psh flag count', 'pshflagcount', 'psh_flag_count', 'PSH_Flag_Count', 'PSH_FLAG_COUNT',
                               'pshflagcount', 'PshFlagCount'],
            'ack_flag_count': ['ack flag count', 'ackflagcount', 'ack_flag_count', 'ACK_Flag_Count', 'ACK_FLAG_COUNT',
                               'ackflagcount', 'AckFlagCount'],
            'urg_flag_count': ['urg flag count', 'urgflagcount', 'urg_flag_count', 'URG_Flag_Count', 'URG_FLAG_COUNT',
                               'urgflagcount', 'UrgFlagCount'],
            'cwe_flag_count': ['cwe flag count', 'cweflagcount', 'cwe_flag_count', 'CWE_Flag_Count', 'CWE_FLAG_COUNT',
                               'cweflagcount', 'CweFlagCount'],
            'ece_flag_count': ['ece flag count', 'eceflagcount', 'ece_flag_count', 'ECE_Flag_Count', 'ECE_FLAG_COUNT',
                               'eceflagcount', 'EceFlagCount'],
            'down_up_ratio': ['down/up ratio', 'downupratio', 'down_up_ratio', 'Down/Up_Ratio', 'DOWN/UP_RATIO',
                              'downupratio', 'DownUpRatio'],
            'average_packet_size': ['average packet size', 'averagepacketsize', 'average_packet_size',
                                    'Average_Packet_Size', 'AVERAGE_PACKET_SIZE', 'averagepacketsize',
                                    'AveragePacketSize'],
            'avg_fwd_segment_size': ['avg fwd segment size', 'avgfwdsegmentsize', 'avg_fwd_segment_size',
                                     'Avg_Fwd_Segment_Size', 'AVG_FWD_SEGMENT_SIZE', 'avgfwdsegmentsize',
                                     'AvgFwdSegmentSize'],
            'avg_bwd_segment_size': ['avg bwd segment size', 'avgbwdsegmentsize', 'avg_bwd_segment_size',
                                     'Avg_Bwd_Segment_Size', 'AVG_BWD_SEGMENT_SIZE', 'avgbwdsegmentsize',
                                     'AvgBwdSegmentSize'],
            'fwd_header_length_1': ['fwd header length.1', 'fwdheaderlength.1', 'fwd_header_length.1',
                                    'Fwd_Header_Length.1', 'FWD_HEADER_LENGTH.1', 'fwdheaderlength1',
                                    'FwdHeaderLength1'],
            'fwd_avg_bytes_bulk': ['fwd avg bytes/bulk', 'fwdavgbytes/bulk', 'fwd_avg_bytes/bulk', 'Fwd_Avg_Bytes/Bulk',
                                   'FWD_AVG_BYTES/BULK', 'fwdavgbytesbulk', 'FwdAvgBytesBulk'],
            'fwd_avg_packets_bulk': ['fwd avg packets/bulk', 'fwdavgpackets/bulk', 'fwd_avg_packets/bulk',
                                     'Fwd_Avg_Packets/Bulk', 'FWD_AVG_PACKETS/BULK', 'fwdavgpacketsbulk',
                                     'FwdAvgPacketsBulk'],
            'fwd_avg_bulk_rate': ['fwd avg bulk rate', 'fwdavgbulkrate', 'fwd_avg_bulk_rate', 'Fwd_Avg_Bulk_Rate',
                                  'FWD_AVG_BULK_RATE', 'fwdavgbulkrate', 'FwdAvgBulkRate'],
            'bwd_avg_bytes_bulk': ['bwd avg bytes/bulk', 'bwdavgbytes/bulk', 'bwd_avg_bytes/bulk', 'Bwd_Avg_Bytes/Bulk',
                                   'BWD_AVG_BYTES/BULK', 'bwdavgbytesbulk', 'BwdAvgBytesBulk'],
            'bwd_avg_packets_bulk': ['bwd avg packets/bulk', 'bwdavgpackets/bulk', 'bwd_avg_packets/bulk',
                                     'Bwd_Avg_Packets/Bulk', 'BWD_AVG_PACKETS/BULK', 'bwdavgpacketsbulk',
                                     'BwdAvgPacketsBulk'],
            'bwd_avg_bulk_rate': ['bwd avg bulk rate', 'bwdavgbulkrate', 'bwd_avg_bulk_rate', 'Bwd_Avg_Bulk_Rate',
                                  'BWD_AVG_BULK_RATE', 'bwdavgbulkrate', 'BwdAvgBulkRate'],
            'subflow_fwd_packets': ['subflow fwd packets', 'subflowfwdpackets', 'subflow_fwd_packets',
                                    'Subflow_Fwd_Packets', 'SUBFLOW_FWD_PACKETS', 'subflowfwdpackets',
                                    'SubflowFwdPackets'],
            'subflow_fwd_bytes': ['subflow fwd bytes', 'subflowfwdbytes', 'subflow_fwd_bytes', 'Subflow_Fwd_Bytes',
                                  'SUBFLOW_FWD_BYTES', 'subflowfwdbytes', 'SubflowFwdBytes'],
            'subflow_bwd_packets': ['subflow bwd packets', 'subflowbwdpackets', 'subflow_bwd_packets',
                                    'Subflow_Bwd_Packets', 'SUBFLOW_BWD_PACKETS', 'subflowbwdpackets',
                                    'SubflowBwdPackets'],
            'subflow_bwd_bytes': ['subflow bwd bytes', 'subflowbwdbytes', 'subflow_bwd_bytes', 'Subflow_Bwd_Bytes',
                                  'SUBFLOW_BWD_BYTES', 'subflowbwdbytes', 'SubflowBwdBytes'],
            'init_win_bytes_forward': ['init win bytes forward', 'initwinbytesforward', 'init_win_bytes_forward',
                                       'Init_Win_Bytes_Forward', 'INIT_WIN_BYTES_FORWARD', 'initwinbytesforward',
                                       'InitWinBytesForward'],
            'init_win_bytes_backward': ['init win bytes backward', 'initwinbytesbackward', 'init_win_bytes_backward',
                                        'Init_Win_Bytes_Backward', 'INIT_WIN_BYTES_BACKWARD', 'initwinbytesbackward',
                                        'InitWinBytesBackward'],
            'act_data_pkt_fwd': ['act data pkt fwd', 'actdatapktfwd', 'act_data_pkt_fwd', 'Act_Data_Pkt_Fwd',
                                 'ACT_DATA_PKT_FWD', 'actdatapktfwd', 'ActDataPktFwd'],
            'min_seg_size_forward': ['min seg size forward', 'minsegsizeforward', 'min_seg_size_forward',
                                     'Min_Seg_Size_Forward', 'MIN_SEG_SIZE_FORWARD', 'minsegsizeforward',
                                     'MinSegSizeForward'],
            'active_mean': ['active mean', 'activemean', 'active_mean', 'Active_Mean', 'ACTIVE_MEAN', 'activemean',
                            'ActiveMean'],
            'active_std': ['active std', 'activestd', 'active_std', 'Active_Std', 'ACTIVE_STD', 'activestd',
                           'ActiveStd'],
            'active_max': ['active max', 'activemax', 'active_max', 'Active_Max', 'ACTIVE_MAX', 'activemax',
                           'ActiveMax'],
            'active_min': ['active min', 'activemin', 'active_min', 'Active_Min', 'ACTIVE_MIN', 'activemin',
                           'ActiveMin'],
            'idle_mean': ['idle mean', 'idlemean', 'idle_mean', 'Idle_Mean', 'IDLE_MEAN', 'idlemean', 'IdleMean'],
            'idle_std': ['idle std', 'idlestd', 'idle_std', 'Idle_Std', 'IDLE_STD', 'idlestd', 'IdleStd'],
            'idle_max': ['idle max', 'idlemax', 'idle_max', 'Idle_Max', 'IDLE_MAX', 'idlemax', 'IdleMax'],
            'idle_min': ['idle min', 'idlemin', 'idle_min', 'Idle_Min', 'IDLE_MIN', 'idlemin', 'IdleMin'],
            'label': ['label', 'class', 'category', 'tag', 'attack_type', 'Label', 'LABEL', 'attack', 'Attack',
                      'classification', 'Classification']
        }

        column_mapping = {}
        unmatched_columns = []

        for col in df_columns:
            matched = False
            col_normalized = col.lower().replace('_', '').replace(' ', '').replace('-', '').replace('.', '').replace(
                '/', '')

            for std_col, variants in standard_columns.items():
                variants_normalized = [
                    v.lower().replace('_', '').replace(' ', '').replace('-', '').replace('.', '').replace('/', '') for v
                    in variants]
                if col_normalized in variants_normalized:
                    column_mapping[col] = std_col
                    matched = True
                    break

            if not matched:
                unmatched_columns.append(col)
                column_mapping[col] = col

        if unmatched_columns:
            print(f"Column not matched: {unmatched_columns}")

        return column_mapping

    def preprocess_excel_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Preprocess Excel data - retain 80+ feature fields for in-depth LLM analysis"""
        if df.empty:
            return []

        column_mapping = self.enhanced_column_mapping(list(df.columns))
        df_std = df.rename(columns=column_mapping)

        records = []
        important_features = [
            # Basic Information
            'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol',
            'timestamp', 'flow_duration', 'label',

            # Packet count and bytes
            'total_fwd_packets', 'total_bwd_packets',
            'total_length_of_fwd_packets', 'total_length_of_bwd_packets',

            # Package length statistics (crucial!)
            'fwd_packet_length_max', 'fwd_packet_length_min', 'fwd_packet_length_mean', 'fwd_packet_length_std',
            'bwd_packet_length_max', 'bwd_packet_length_min', 'bwd_packet_length_mean', 'bwd_packet_length_std',

            # Time interval statistics (crucial!)
            'flow_iat_mean', 'flow_iat_std', 'flow_iat_max', 'flow_iat_min',
            'fwd_iat_total', 'fwd_iat_mean', 'fwd_iat_std', 'fwd_iat_max', 'fwd_iat_min',
            'bwd_iat_total', 'bwd_iat_mean', 'bwd_iat_std', 'bwd_iat_max', 'bwd_iat_min',

            # TCP Flags (Crucial! Used to determine if the handshake/waveback is working correctly)
            'fin_flag_count', 'syn_flag_count', 'rst_flag_count',
            'psh_flag_count', 'ack_flag_count', 'urg_flag_count',
            'cwe_flag_count', 'ece_flag_count',

            # Rate and proportion
            'flow_bytes_per_sec', 'flow_packets_per_sec',
            'down_up_ratio',

            # Window size (to determine if the attack tool has misconfigured the configuration).
            'init_win_bytes_forward', 'init_win_bytes_backward',

            # Active/Idle Time (to determine if it's a zombie outbreak)
            'active_mean', 'active_std', 'active_max', 'active_min',
            'idle_mean', 'idle_std', 'idle_max', 'idle_min',

            # Sub-stream characteristics
            'subflow_fwd_packets', 'subflow_fwd_bytes',
            'subflow_bwd_packets', 'subflow_bwd_bytes',
        ]

        for idx, row in df_std.iterrows():
            record = {'record_id': f"excel_row_{idx + 1}"}

            # Extract all possible features
            for feat in important_features:
                if feat in df_std.columns:
                    val = row[feat]
                    if pd.isna(val):
                        val = 0 if feat not in ['src_ip', 'dst_ip', 'protocol', 'label'] else '未知'
                    record[feat] = val
                else:
                    # Missing fields are padded with 0s or marked as unknown (this does not affect the analysis).
                    record[feat] = 0 if feat not in ['src_ip', 'dst_ip', 'protocol', 'label'] else '未知'

            # Special handling: Convert protocol to readable form
            proto_map = {'6': 'TCP', '17': 'UDP', '1': 'ICMP'}
            record['protocol'] = proto_map.get(str(record['protocol']), str(record['protocol']))

            records.append(record)

        print(f"Preprocessing complete, a total of {len(records)} records were extracted, each containing approximately {len(records[0]) if records else 0} feature fields.")
        return records

    def comprehensive_llm_analysis(self, traffic_record: Dict[str, Any]) -> Dict[str, Any]:
        """Integrated LLM Analysis - Integrating Stratified Analysis and RAG"""

        analysis_result = self.llm_integration.perform_hierarchical_analysis(traffic_record)

        return analysis_result

    def analyze_excel_file(self, excel_path: str, max_records: int = None, resume_from: str = None) -> List[
        Dict[str, Any]]:
        """
        Supports resume download + saves only the final result + progress bar display
        """
        print(f"Start analyzing Excel files: {excel_path}")

        # 1. Load data
        df = self.load_excel_data(excel_path)
        if df.empty:
            print("The Excel file is empty or failed to load.")
            return []

        records = self.preprocess_excel_data(df)
        if not records:
            print("No valid traffic records found")
            return []

        total = len(records)
        if max_records is not None and max_records > 0:
            total = min(max_records, total)

        records = records[:total]

        # 2. Preparation for resuming interrupted downloads
        output_path = "Final annotation review report.jsonl"
        analyzed_ids = set()

        if resume_from and os.path.exists(resume_from):
            print(f"Historical results file detected; loading analyzed records to enable resume download....")
            try:
                with open(resume_from, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            record_id = data["original_record"].get("record_id")
                            if record_id:
                                analyzed_ids.add(record_id)
                print(f"{len(analyzed_ids)} analyzed records have been skipped.")
            except Exception as e:
                print(f"Failed to load breakpoint file: {e}, will start from the beginning.")

        elif os.path.exists(output_path):
            print(f"An existing result file {output_path} was found; an attempt was made to load a breakpoint...")
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            record_id = data["original_record"].get("record_id")
                            if record_id:
                                analyzed_ids.add(record_id)
                print(f"{len(analyzed_ids)} analyzed records have been skipped.")
            except Exception as e:
                print(f"Failed to load result file: {e}")

        # 3. Filter out already analyzed records
        pending_records = []
        for rec in records:
            rid = rec.get("record_id")
            if rid not in analyzed_ids:
                pending_records.append(rec)
            else:
                pass  # Already analyzed, skip this step.

        print(
            f"Total records: {len(records)}, Completed: {len(records) - len(pending_records)}, To be analyzed this time: {len(pending_records)}")

        if len(pending_records) == 0:
            print("All records have been analyzed!")
            return []

        # 4. Start Analysis + Progress Bar + Real-time Append Saving (to prevent data loss in case of crash)
        results = []
        temp_results = []

        try:
            with open(output_path, 'a', encoding='utf-8') as f:
                for i, record in enumerate(tqdm(pending_records, desc="Analysis progress", unit="条")):
                    try:
                        analysis_result = self.comprehensive_llm_analysis(record)
                        result = {
                            "original_record": record,
                            "llm_rag_analysis": analysis_result,
                            "analysis_timestamp": datetime.now().isoformat()
                        }
                        results.append(result)
                        temp_results.append(result)

                        # A forced disk flush is performed every 50 records analyzed (balancing security and performance).
                        if len(temp_results) >= 50:
                            for r in temp_results:
                                f.write(json.dumps(r, ensure_ascii=False) + '\n')
                            f.flush()
                            temp_results.clear()

                        tqdm.write(f"Completed: {i + 1}/{len(pending_records)}")

                    except Exception as e:
                        error_result = {
                            "original_record": record,
                            "llm_rag_analysis": {"error": str(e)},
                            "analysis_timestamp": datetime.now().isoformat()
                        }
                        results.append(error_result)
                        temp_results.append(error_result)
                        tqdm.write(f"The {i + 1}th analysis failed.: {e}")

                for r in temp_results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
                f.flush()

        except KeyboardInterrupt:
            print("\nManual termination, saving analysis results...")
            with open(output_path, 'a', encoding='utf-8') as f:
                for r in temp_results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            print("It has been safely saved and will automatically skip the analyzed section on the next run.！")
            raise

        print(f"Analysis complete! A total of {len(pending_records)} records were processed.")
        print(f"The final result has been saved to: {output_path}")

        # Generate summary report
        all_results = []
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        all_results.append(json.loads(line))
            summary = self.generate_summary_report(all_results)
            with open("Analysis and Summary Report.json", 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print("Analysis and summary report has been generated: Final annotation review report.json")
        except:
            pass

        return results

    def save_analysis_results(self, results: List[Dict[str, Any]],
                              output_path: str = "optimized_llm_rag_analysis.jsonl"):
        """Save analysis results"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')

        print(f"The analysis results have been saved to: {output_path}")

    def generate_summary_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary report"""
        total_records = len(results)
        risk_distribution = {}
        scores = []
        rag_usage_stats = []

        for result in results:
            analysis = result.get('llm_rag_analysis', {})
            assessment = analysis.get('hierarchical_analysis', {}).get('comprehensive_assessment', {})
            risk_level = assessment.get('risk_level', 'unknown')
            overall_score = assessment.get('overall_score', 0.5)

            risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1
            scores.append(overall_score)

            # Statistics on RAG usage
            rag_used = analysis.get('rag_context_used', False)
            rag_usage_stats.append(rag_used)

        avg_score = sum(scores) / len(scores) if scores else 0
        rag_usage_rate = sum(rag_usage_stats) / len(rag_usage_stats) if rag_usage_stats else 0

        return {
            'total_records_analyzed': total_records,
            'average_risk_score': avg_score,
            'risk_distribution': risk_distribution,
            'rag_usage_rate': rag_usage_rate,
            'rag_effectiveness': 'High' if rag_usage_rate > 0.5 else 'Medium' if rag_usage_rate > 0.2 else 'Low',
            'analysis_date': datetime.now().isoformat()
        }


def main():
    print("Launch the optimized version of Excel Traffic Analyzer...")

    analyzer = EnhancedExcelTrafficAnalyzerIntegrated(
        rag_knowledge_paths=["rag_data"],
        index_path="vector_index"
    )

    analyzer.analyze_excel_file(
        excel_path="test1.xlsx", # Your file path
        max_records=None,        # Analyze all
        resume_from="Final annotation review report.jsonl"  # Automatically skip analyzed
    )

if __name__ == "__main__":
    main()
