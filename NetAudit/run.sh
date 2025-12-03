### 2. `run.sh`（Linux/macOS）

```bash
#!/bin/bash
echo "=================================="
echo "NetAudit 一键复现开始"
echo "=================================="

# 1. 运行主审计程序
echo "[1/2] 正在执行三层审计（预计 10-12 小时）..."
python enhanced_excel_traffic_analyzer_integrated.py

# 2. 运行新颖性分析
echo "[2/2] 正在执行新颖性检测与统计（约 3 小时）..."
python analyze_report_summary.py

echo "=================================="
echo "全部完成！"
echo "结果文件："
echo "   - 最终标注审查报告.jsonl"
echo "   - novelty_report.xlsx（含483条新错误统计）"
echo "=================================="