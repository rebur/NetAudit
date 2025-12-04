Markdown# NetAudit: Automated Labeling-Quality Auditor for NIDS Datasets

**DSN 2025 Artifact** | Available • Functional • Reusable

本项目是论文 **NetAudit: A Reasoning-Driven Automated Auditing Tool for Labeling Quality in Network Intrusion Detection Datasets** 的完整可复现代码。

## 快速开始（< 5 分钟）

### 1. 克隆仓库
```bash
git clone https://github.com/rebur/NetAudit.git
cd NetAudit
2. 安装依赖
Bashpip install -r requirements.txt
3. 配置你的 LLM（必须！）
打开 enhanced_excel_traffic_analyzer_integrated.py，找到 setup_llm 函数，修改以下两行：
Pythonapi_key="sk-your-real-key-here",           # ← 填入你的通义千问 / OpenAI / Grok 等密钥
base_url="https://your-endpoint.com/v1"     # ← 填入对应平台的兼容地址
同时，在所有调用 LLM 的地方已写为：
Pythonmodel="your_model"   # ← 改为你实际使用的模型名，例如：
                     # "qwen-turbo" / "qwen-plus" / "gpt-4o" / "claude-3-5-sonnet-20241022" 等
我们实验使用的是 qwen-turbo，性价比最高，建议首次复现时使用相同模型。
4. 一键复现论文结果
Bash# Linux / macOS
bash run.sh

# Windows
run.bat
脚本将自动：

加载你的 CSV 文件
执行完整三层审计
输出 最终标注审查报告.jsonl
运行 analyze_report_summary.py 自动生成新错误统计
复现论文 Table 2（483 条新错误，9 个新缺陷类别）

预计总耗时：12–15 小时（与论文一致）
文件结构
textNetAudit/
├── enhanced_excel_traffic_analyzer_integrated.py   # 主程序
├── analyze_report_summary.py                        # 新颖性分析
├── rag_data/                                       # 6 篇权威文档（2篇论文 + 4个RFC）
├── vector_index/                                   # 预构建FAISS索引
├── requirements.txt
├── run.sh / run.bat
└── README.md
