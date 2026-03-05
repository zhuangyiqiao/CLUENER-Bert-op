#!/usr/bin/env python
"""
Final comprehensive check before deployment
"""
import json
import sys
from pathlib import Path

def check_file_exists(path, description):
    if Path(path).exists():
        print(f"  ✅ {description}: {path}")
        return True
    else:
        print(f"  ❌ {description}: {path} NOT FOUND")
        return False

print("=" * 70)
print("🔐 部署前最终检查清单")
print("=" * 70)

all_pass = True

# 1. Core files
print("\n📁 核心文件检查:")
core_files = [
    ("scripts/train.py", "训练脚本"),
    ("scripts/predict.py", "预测脚本"),
    ("src/models/model.py", "模型定义"),
    ("src/models/crf.py", "CRF 层"),
    ("src/training/trainer.py", "训练器"),
    ("src/training/lr_scheduler.py", "学习率调度器"),
    ("src/data/processor.py", "数据处理器"),
    ("src/data/dataset.py", "数据集"),
    ("src/data/collate.py", "批处理器"),
    ("src/data/label_map.py", "标签映射"),
    ("src/config/config_parser.py", "配置解析器"),
    ("src/utils/logger.py", "日志记录"),
    ("src/utils/seed.py", "随机种子"),
    ("src/utils/tagging.py", "标签工具"),
]
for fpath, desc in core_files:
    if not check_file_exists(fpath, desc):
        all_pass = False

# 2. Config and data files
print("\n📊 配置和数据文件检查:")
data_files = [
    ("configs/bert_crf.json", "训练配置"),
    ("data/train.json", "训练数据"),
    ("data/dev.json", "验证数据"),
    ("data/test.json", "测试数据"),
]
for fpath, desc in data_files:
    if not check_file_exists(fpath, desc):
        all_pass = False

# 3. Config content validation
print("\n⚙️  配置内容检查:")
try:
    with open("configs/bert_crf.json", "r") as f:
        cfg = json.load(f)
    
    # Check critical keys
    required_keys = [
        ("model.pretrained_name", lambda c: c.get("model", {}).get("pretrained_name")),
        ("train.learning_rate", lambda c: c.get("train", {}).get("learning_rate")),
        ("train.learning_rate_head", lambda c: c.get("train", {}).get("learning_rate_head")),
        ("data.max_length", lambda c: c.get("data", {}).get("max_length")),
        ("train.num_epochs", lambda c: c.get("train", {}).get("num_epochs")),
        ("train.train_batch_size", lambda c: c.get("train", {}).get("train_batch_size")),
        ("train.eval_batch_size", lambda c: c.get("train", {}).get("eval_batch_size")),
    ]
    
    for key_name, getter in required_keys:
        val = getter(cfg)
        if val is not None:
            print(f"  ✅ {key_name} = {val}")
        else:
            print(f"  ❌ {key_name} NOT FOUND")
            all_pass = False
            
except Exception as e:
    print(f"  ❌ 配置文件读取失败: {e}")
    all_pass = False

# 4. Output directory
print("\n📂 输出目录检查:")
output_dir = Path("./outputs")
if output_dir.exists():
    print(f"  ✅ 输出目录存在: ./outputs/")
else:
    print(f"  ✅ 输出目录不存在，训练时会自动创建")

# 5. Python and dependencies
print("\n🐍 Python 和依赖检查:")
try:
    import torch
    print(f"  ✅ torch: {torch.__version__}")
except:
    print(f"  ❌ torch not found")
    all_pass = False

try:
    import transformers
    print(f"  ✅ transformers: {transformers.__version__}")
except:
    print(f"  ❌ transformers not found")
    all_pass = False

try:
    import seqeval
    print(f"  ✅ seqeval: 已安装")
except:
    print(f"  ❌ seqeval not found")
    all_pass = False

# 6. Module imports
print("\n📦 模块导入检查:")
sys.path.insert(0, str(Path.cwd()))

modules_to_check = [
    "src.config.config_parser",
    "src.data.processor",
    "src.data.dataset",
    "src.models.model",
    "src.training.trainer",
    "src.training.lr_scheduler",
    "src.utils.logger",
]

for module_name in modules_to_check:
    try:
        __import__(module_name)
        print(f"  ✅ {module_name}")
    except Exception as e:
        print(f"  ❌ {module_name}: {str(e)[:50]}")
        all_pass = False

# Final result
print("\n" + "=" * 70)
if all_pass:
    print("🎉 所有检查通过！项目可以部署到服务器")
    print("\n📌 下一步:")
    print("  1. 上传整个 ner-crf 目录到服务器")
    print("  2. 在服务器上运行: python scripts/train.py")
    print("  3. 监控 outputs/train.log 查看训练进度")
else:
    print("❌ 存在未通过的检查项，请修复后再部署")
    sys.exit(1)

print("=" * 70)
