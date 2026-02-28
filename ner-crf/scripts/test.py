import sys
from pathlib import Path
from collections import Counter

print("[TEST] script start")

# 1. 路径设置
ROOT = Path(__file__).resolve().parent.parent   # ner-crf/
sys.path.insert(0, str(ROOT))
print("[TEST] ROOT =", ROOT)

try:
    from src.config.config_parser import load_config
    from src.data.processor import CluenerProcessor
    from src.data.label_map import build_label_map_from_config
    from src.data.dataset import CluenerBertDataset
    print("[TEST] imports ok")
except Exception as e:
    print("[TEST] import failed:", repr(e))
    raise

# 2. 加载配置与数据
config_path = ROOT / "configs" / "bert_crf.json"
cfg = load_config(str(config_path))
lm = build_label_map_from_config(cfg)

processor = CluenerProcessor(Path(cfg["data_dir"]))
train_examples = processor.get_train_examples()
print(f"[TEST] num_train_examples = {len(train_examples)}")

# 3. 构建 Dataset
ds = CluenerBertDataset(
    examples=train_examples,
    pretrained_name=cfg["model"]["pretrained_name"],
    label_map=lm,
    max_length=64
)

# 4. 初始化 Tokenizer (用于解码检查)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(cfg["model"]["pretrained_name"])

# 5. 全量遍历检查
print("\n[CHECK] Start validating all training data...")

error_count = 0
max_errors_to_show = 5  # 最多打印前 5 个错误的详细信息，避免刷屏
label_counter = Counter()
length_counter = Counter()

for idx in range(len(ds)):
    item = ds[idx]
    input_ids = item["input_ids"]
    label_ids = item["label_ids"]
    valid_mask = item["valid_mask"]
    
    # 基础长度检查
    if not (len(input_ids) == len(label_ids) == len(valid_mask)):
        if error_count < max_errors_to_show:
            print(f"\n[ERROR] Sample {idx}: Length mismatch!")
            print(f"  len(input_ids)={len(input_ids)}, len(label_ids)={len(label_ids)}, len(valid_mask)={len(valid_mask)}")
        error_count += 1
        continue

    # 维度检查 (确保没有多余维度)
    if isinstance(input_ids[0], list):
        if error_count < max_errors_to_show:
            print(f"\n[ERROR] Sample {idx}: input_ids has extra dimension!")
        error_count += 1
        continue

    # 统计信息收集
    length_counter[len(input_ids)] += 1
    for lid in label_ids:
        label_counter[lid] += 1

    # 深度对齐检查 (可选：检查特殊 token 的标签)
    # 通常 [CLS] (index 0) 的 label 应该是 0 (O) 或忽略
    # 如果 valid_mask[0] == 0 但 label_ids[0] != 0，可能是标注泄露
    if valid_mask[0] == 0 and label_ids[0] != 0:
         # 这取决于你的 CRF 实现，有些实现允许 CLS 有标签，有些强制为 0
         # 这里仅作为警告，不计入错误
         pass

# 6. 输出报告
print("\n" + "="*30)
print("[TEST] VALIDATION REPORT")
print("="*30)

if error_count > 0:
    print(f"❌ Found {error_count} errors in total.")
    print(f"   (Displayed first {min(error_count, max_errors_to_show)} errors above)")
else:
    print("✅ All samples passed alignment checks!")

print(f"\n📊 Sequence Length Distribution:")
for length, count in sorted(length_counter.items()):
    print(f"   Length {length:2d}: {count:5d} samples ({count/len(ds)*100:.1f}%)")

print(f"\n🏷️  Label ID Distribution (Top 10):")
# 将 ID 转换为标签名显示
top_labels = label_counter.most_common(10)
for lid, count in top_labels:
    label_name = lm.id_to_label.get(lid, "UNKNOWN")
    print(f"   ID {lid:2d} ({label_name:12s}): {count:6d}")

# 检查是否有未使用的标签
print(f"\nℹ️  Total unique labels in data: {len(label_counter)}")
print(f"ℹ️  Total labels in label_map: {len(lm.label_to_id)}")

# 简单抽样打印最后一条数据作为确认
print(f"\n🔍 Sanity Check (Last Sample):")
last_item = ds[-1]
tokens = tok.convert_ids_to_tokens(last_item["input_ids"])
labels = lm.decode(last_item["label_ids"])
# 过滤掉 padding
valid_tokens = [t for t, m in zip(tokens, last_item["valid_mask"]) if m == 1]
valid_labels = [l for l, m in zip(labels, last_item["valid_mask"]) if m == 1]
print(f"   Text: {''.join(valid_tokens).replace('##', '')}")
print(f"   Labels: {valid_labels[:10]}...")

print("\n[TEST] done")