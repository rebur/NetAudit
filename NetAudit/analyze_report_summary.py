import json
import os
import re
import pandas as pd
from openai import OpenAI  # 用于调用 qwen-turbo
from datetime import datetime
from tqdm import tqdm  # 进度条



KNOWN_ISSUES = {
    "DoS Hulk Misimplification", "Connection header set to Close instead of Keep-Alive", "Attack Invalid",
    "Lack of Attack Configuration Variety", "Fixed Parameters Unchanged", "Attack Tool Settings Unchanged",
    "Attack Launch/Demolition Artifact", "Irrelevant Traffic Such as Login Attempts or Login Page Access",
    "Attack Artifact", "Irrelevant Traffic Between Attacker and Victim",
    "TCP Appendices", "First FIN Packet Terminates Flow", "Violation of TCP Specification",
    "Remaining ACK/FIN Packets Create Meaningless Flow",
    "Ignoring RST Packets", "RST Does Not Terminate Flow", "Invalid Flow Continues",
    "Timeout and TCP Appendices Issues", "Timeout Causes Incorrect Flow Direction", "Source/Destination Switching",
    "TCP Segment Offset Issues", "TSO Packet IP Length is 0 Resolution Error", "Unknown Payload Flow",
    "Packet Timestamp Issues", "Local NIC Timestamp Causes Out-of-Order Packets", "SYN-ACK Before SYN",
    "Label Pollution/Corruption", "High Corruption Rate Such as Web Brute Force 95.16%", "Misassignment Rate > 5%",
    "Omitted Attacks", "Inaccurate Time Range Omitted Malicious Flows", "e.g., Port Scan 38.41%",
    "Temptative Attack Flows", "No-Payment Flows Marked as Complete Attacks", "Failed TCP Connections",
    "Time-Label-Based", "Background Traffic Marked as Malicious Due to Time/Host Filtering",
    "Fuzzy Class Labels", "Overlapping Flows Across Classes", "TCP Appendices Same as Benign",
    "Port/System Closed", "Unavailable Target Attack Marked as Successful",
    "No Malicious Payload", "Payload Exists but No Attack Content", "e.g., Retransmissions",
    "Target Unresponsive", "Victim Down", "SYN Packet Flow Marked as Attack",
    "Existing Attribute Calculation Error", "CICFlowMeter Implementation Failure",
    "Shortcut Learning", "Includes Identification Fields such as Flow ID, IP, Port, Timestamp", "Overfitting",
    "Missing ICMP Protocol Support", "Misunderstood Undelivered Packets",
    "Flow Duration Calculation Error", "Negative values or outliers",
    "IAT Statistics Errors", "Negative Mean/Standard Deviation", "Packet length statistics error",
    "Negative standard deviation/mean",
    "Header length calculation error",
    "Flag count error", "PSH, URG, ECE flag miscalculation",
    "Initial window byte error", "Negative value",
    "Sub-flow feature error",
    "Active/idle time calculation error",
    "Flow bytes/s and packet/s rate calculation error",
    "Inaccurate time range", "Inaccurate reporting window", "Missing flow",
    "Lack of details in benign traffic generation", "B-Profiles sampling, environment, statistics",
    "Insufficient attack tool parameters", "No configuration diversity information",
    "Closed-source tagging logic", "No transparency/reproducibility",
    "Missing reconciliation features", "Essentially missing Flow-ID, IP in the CSE-CIC-IDS2018 subset",
    "Aggregate metrics only", "Hidden imbalanced dataset class problem",
    "No performance per class", "Confusion matrix, missing F1 per class",
    "Overfitting to artifact", "Model learns incorrect patterns instead of true patterns"
}

# ==================== 超强容错加载器 ====================
def load_reports_ultra_robust(path="Final annotation review report.jsonl"):
    if not os.path.exists(path):
        print(f"The file does not exist.: {path}")
        return []
    print(f"Loading report file：{path}")
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    reports = []
    chunks = re.split(r'(?<=})\s*(?={)', content)
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk: continue
        chunk = re.sub(r'```json\s*|\s*```', '', chunk)
        if not (chunk.startswith('{') and chunk.endswith('}')): continue
        try:
            reports.append(json.loads(chunk))
        except:
            try:
                reports.append(json.loads(chunk.replace("'", '"')))
            except:
                continue
    print(f"Successfully loaded {len(reports):,} records.")
    return reports

# ==================== 提取 issues ====================
def extract_issues_with_novelty(reports, client):
    all_issues = []
    suspicious = []

    for r in tqdm(reports, desc="提取 issues"):
        rec = r.get("original_record", {})
        label = rec.get("label", "unknown")
        rid = rec.get("record_id", "unknown")

        analysis = r.get("llm_rag_analysis", {}) or r.get("hierarchical_analysis", {}) or {}
        if isinstance(analysis, str):
            try: analysis = json.loads(analysis)
            except: analysis = {}

        verdict = str(analysis.get("overall_verdict", "") or analysis.get("comprehensive_reasoning", "") or "")
        if any(k in verdict for k in ["Unreasonable", "Failed", "Suspicious", "Suggested Correction", "Problematic"]):
            suspicious.append({"record_id": rid, "label": label})

        def walk(node, all_issues):
            if not isinstance(node, dict): return
            for v in node.values():
                if isinstance(v, dict) and "issues" in v:
                    for issue in v.get("issues", []):
                        if not issue or not isinstance(issue, str): continue
                        text = issue.strip()
                        all_issues.append({
                            "record_id": rid,
                            "label": label,
                            "issue": text,
                            "severity": v.get("severity", "中")
                        })
                if isinstance(v, dict):
                    walk(v, all_issues)

        walk(analysis, all_issues)

    # 独特 issues
    df = pd.DataFrame(all_issues)
    unique_issues = list(df['issue'].unique())

    # 用 LLM 语义匹配 known_issues
    batch_size = 20  # 每批 20 个，防 token 超限
    novelty_map = {}
    for i in tqdm(range(0, len(unique_issues), batch_size), desc="LLM 语义匹配"):
        batch = unique_issues[i:i+batch_size]
        prompt = f"""
        You are a cybersecurity expert. Below is a list of known error types in CIC-IDS2017.：{json.dumps(list(KNOWN_ISSUES), ensure_ascii=False)}。
        For the following list of issues：{json.dumps(batch, ensure_ascii=False)}。
        Please determine whether each issue semantically matches any of the known types.
        If a match is found, return the most matching known type; otherwise, return "New Discovery".
        Output a JSON dictionary: keys are the original issue, are the matching results.
        """
        response = client.chat.completions.create(
            model="qwen-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        try:
            result = json.loads(response.choices[0].message.content.strip('```json').strip('```'))
            novelty_map.update(result)
        except Exception as e:
            print(f"LLM batch processing failed: {e}, skipping this batch.")

    # 应用 novelty
    df['novelty'] = df['issue'].map(novelty_map).fillna("New discovery")

    # 计算统计
    known = df[df['novelty'] != "New discovery"]
    novel = df[df['novelty'] == "New discovery"]

    print(f"The paper has reported an error ： {len(known):,} 条 ({len(known)/len(df)*100:.1f}%)")
    print(f"本New research findings ： {len(novel):,} 条 ({len(novel)/len(df)*100:.1f}%)")

    # 保存
    df.to_excel("[LLM Matching Version] All Errors.xlsx", index=False)
    print("Saved: [LLM Matching Version] All Errors.xlsx")

    return df, suspicious

# ==================== 主函数 ====================
def main():
    # Please note that you must use your own api_key!!!
    client = OpenAI(
        api_key="sk-your-key", # Replace your api_key！！！
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1" # Replace your url！！！
    )

    reports = load_reports_ultra_robust()
    if not reports:
        return

    df, suspicious = extract_issues_with_novelty(reports, client)

    print("\n" + "="*90)
    print("="*90)
    print(f"Total number of records analyzed     ： {len(reports):,}")
    print(f"Total number of problems found      ： {len(df):,}")

    # 保存报告时间戳
    timestamp = datetime.now().isoformat()
    print(f"Report generation time: {timestamp}")

if __name__ == "__main__":
    main()