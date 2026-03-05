#!/usr/bin/env python
"""
Quick data loading test to verify processor and dataset
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config.config_parser import load_config
from src.data.processor import CluenerProcessor
from src.data.label_map import build_label_map_from_config
from src.data.dataset import CluenerBertDataset

print("=" * 60)
print("📊 数据加载测试")
print("=" * 60)

# Load config
cfg = load_config("configs/bert_crf.json")
print(f"\n✅ 配置已加载")
print(f"  Data dir: {cfg['data_dir']}")

# Load label map
label_map = build_label_map_from_config(cfg)
print(f"\n✅ Label map 已加载")
print(f"  总标签数: {label_map.num_labels}")
print(f"  Sample labels: O={label_map.label2id['O']}, B-name={label_map.label2id.get('B-name', 'N/A')}")

# Load data with processor
processor = CluenerProcessor(Path(cfg["data_dir"]))
print(f"\n✅ 数据处理器已初始化")

try:
    train_examples = processor.get_train_examples()
    print(f"✅ 训练数据加载成功: {len(train_examples)} 条样本")
    if train_examples:
        sample = train_examples[0]
        print(f"  Sample tokens: {sample['tokens'][:20]}...")
        print(f"  Sample labels: {sample['labels'][:20]}...")
except Exception as e:
    print(f"❌ 训练数据加载失败: {e}")
    sys.exit(1)

try:
    dev_examples = processor.get_dev_examples()
    print(f"✅ 开发数据加载成功: {len(dev_examples)} 条样本")
except Exception as e:
    print(f"❌ 开发数据加载失败: {e}")
    sys.exit(1)

# Test dataset
print(f"\n✅ 创建 dataset...")
pretrained_name = cfg["model"]["pretrained_name"]
max_length = cfg["data"]["max_length"]

try:
    train_ds = CluenerBertDataset(
        examples=train_examples[:5],  # Test with first 5 samples
        pretrained_name=pretrained_name,
        label_map=label_map,
        max_length=max_length
    )
    print(f"✅ Dataset 创建成功: {len(train_ds)} 条样本")
    
    # Try to get one sample
    sample = train_ds[0]
    print(f"✅ 样本读取成功")
    print(f"  input_ids shape: {len(sample['input_ids'])}")
    print(f"  label_ids shape: {len(sample['label_ids'])}")
    print(f"  valid_mask shape: {len(sample['valid_mask'])}")
    
except Exception as e:
    print(f"❌ Dataset 创建/读取失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ 所有数据加载测试通过！")
print("=" * 60)
