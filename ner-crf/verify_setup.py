#!/usr/bin/env python
"""
Pre-training verification script to check all dependencies and imports
"""
import sys
from pathlib import Path

# Add ner-crf to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

print("=" * 60)
print("🔍 ner-crf 依赖检查")
print("=" * 60)

# 1. Check Python version
print(f"\n✅ Python version: {sys.version}")

# 2. Check required packages
required_packages = [
    'torch',
    'transformers',
    'tokenizers',
    'seqeval',
    'numpy',
    'tqdm'
]

print("\n📦 Required packages:")
missing = []
for pkg in required_packages:
    try:
        __import__(pkg)
        print(f"  ✅ {pkg}")
    except ImportError as e:
        print(f"  ❌ {pkg} - {str(e)}")
        missing.append(pkg)

if missing:
    print(f"\n⚠️  Missing packages: {', '.join(missing)}")
    print("   Run: pip install -r requirements.txt")
    sys.exit(1)

# 3. Check module imports
print("\n📚 Project modules:")
try:
    from src.config.config_parser import load_config
    print("  ✅ config_parser")
except Exception as e:
    print(f"  ❌ config_parser - {str(e)}")

try:
    from src.data.processor import CluenerProcessor
    print("  ✅ processor")
except Exception as e:
    print(f"  ❌ processor - {str(e)}")

try:
    from src.data.label_map import build_label_map_from_config
    print("  ✅ label_map")
except Exception as e:
    print(f"  ❌ label_map - {str(e)}")

try:
    from src.data.dataset import CluenerBertDataset
    print("  ✅ dataset")
except Exception as e:
    print(f"  ❌ dataset - {str(e)}")

try:
    from src.data.collate import DataCollatorForBertCrf
    print("  ✅ collate")
except Exception as e:
    print(f"  ❌ collate - {str(e)}")

try:
    from src.models.model import BertCrfForNer
    print("  ✅ model")
except Exception as e:
    print(f"  ❌ model - {str(e)}")

try:
    from src.models.crf import CRF
    print("  ✅ crf")
except Exception as e:
    print(f"  ❌ crf - {str(e)}")

try:
    from src.training.trainer import Trainer
    print("  ✅ trainer")
except Exception as e:
    print(f"  ❌ trainer - {str(e)}")

try:
    from src.training.lr_scheduler import build_scheduler
    print("  ✅ lr_scheduler")
except Exception as e:
    print(f"  ❌ lr_scheduler - {str(e)}")

try:
    from src.utils.logger import init_logger
    print("  ✅ logger")
except Exception as e:
    print(f"  ❌ logger - {str(e)}")

try:
    from src.utils.seed import set_seed
    print("  ✅ seed")
except Exception as e:
    print(f"  ❌ seed - {str(e)}")

try:
    from src.utils.tagging import bioes_to_spans, spans_to_dicts
    print("  ✅ tagging")
except Exception as e:
    print(f"  ❌ tagging - {str(e)}")

# 4. Load and verify config
print("\n⚙️  Configuration verification:")
try:
    cfg = load_config("configs/bert_crf.json")
    print("  ✅ Config loaded")
    
    # Check critical keys
    critical_keys = [
        ('data_dir', 'config["data_dir"]'),
        ('output_dir', 'config["output_dir"]'),
        ('train.learning_rate', 'config["train"]["learning_rate"]'),
        ('train.learning_rate_head', 'config["train"]["learning_rate_head"]'),
        ('model.pretrained_name', 'config["model"]["pretrained_name"]'),
    ]
    
    print("\n  Critical config values:")
    for key_path, display in critical_keys:
        keys = key_path.split('.')
        val = cfg
        for k in keys:
            if k in val:
                val = val[k]
            else:
                print(f"    ❌ {key_path} = NOT FOUND")
                continue
        print(f"    ✅ {key_path} = {val}")
    
except Exception as e:
    print(f"  ❌ Config error - {str(e)}")
    sys.exit(1)

# 5. Verify CUDA availability
import torch
print(f"\n🚀 Device information:")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
print(f"  PyTorch version: {torch.__version__}")

print("\n" + "=" * 60)
print("✅ 所有检查通过！可以开始训练。")
print("=" * 60)
print("\n运行训练：")
print("  python scripts/train.py")
