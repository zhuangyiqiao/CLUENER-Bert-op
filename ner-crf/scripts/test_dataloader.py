import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from collections import Counter

# 添加 src 目录到 Python 路径
ROOT = Path(__file__).resolve().parent.parent  # ner-crf/
sys.path.insert(0, str(ROOT))  # 添加项目根目录

from src.config.config_parser import load_config
from src.data.processor import CluenerProcessor
from src.data.label_map import build_label_map_from_config
from src.data.dataset import CluenerBertDataset
from src.data.collate import DataCollatorForBertCrf

print("[TEST] script start")
print("[TEST] ROOT =", ROOT)

# 1. 加载配置与数据
config_path = ROOT / "configs" / "bert_crf.json"
cfg = load_config(str(config_path))
lm = build_label_map_from_config(cfg)

processor = CluenerProcessor(Path(cfg["data_dir"]))
train_examples = processor.get_train_examples()
print(f"[TEST] num_train_examples = {len(train_examples)}")

# 2. 构建 Dataset
ds = CluenerBertDataset(
    examples=train_examples,
    pretrained_name=cfg["model"]["pretrained_name"],
    label_map=lm,
    max_length=64
)

# 3. 构建 Collator 和 DataLoader
collator = DataCollatorForBertCrf(
    pretrained_name=cfg["model"]["pretrained_name"],
    label_map=lm
)

batch_size = 4
dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator)
print(f"[TEST] batch_size = {batch_size}, num_batches = {len(dl)}")

# 4. 全量遍历检查
print("\n[CHECK] Start validating all batches...")

error_count = 0
max_errors_to_show = 5
batch_count = 0

length_counter = Counter()      # 记录每个 batch 中实际序列长度的分布
label_pad_count = 0             # 统计 padding 区域的 label 值
valid_mask_errors = 0           # valid_mask 校验错误
label_pad_errors = 0            # label padding 校验错误

for batch_idx, batch in enumerate(dl):
    batch_count += 1
    input_ids = batch["input_ids"]      # [B, L]
    attention_mask = batch["attention_mask"]
    label_ids = batch["label_ids"]      # [B, L]
    valid_mask = batch["valid_mask"]    # [B, L]
    
    B, L = input_ids.shape
    
    # --- 检查 1: 张量形状一致性 ---
    if not (attention_mask.shape == label_ids.shape == valid_mask.shape == input_ids.shape):
        if error_count < max_errors_to_show:
            print(f"\n[ERROR] Batch {batch_idx}: Shape mismatch!")
            print(f"  input_ids: {input_ids.shape}")
            print(f"  attention_mask: {attention_mask.shape}")
            print(f"  label_ids: {label_ids.shape}")
            print(f"  valid_mask: {valid_mask.shape}")
        error_count += 1
        continue
    
    # --- 检查 2: 逐行检查 Padding 区域 ---
for i in range(B):
    # 找到最后一个 1 的位置（而不是 1 的总数）
    ones_indices = torch.where(valid_mask[i] == 1)[0]
    if len(ones_indices) == 0:
        continue  # 没有有效 token，跳过
    last_valid_pos = int(ones_indices[-1].item())  # 最后一个 1 的索引
    
    # 检查 padding 区域（最后一个 1 之后）
    if last_valid_pos < L - 1:
        vm_pad = valid_mask[i, last_valid_pos + 1:]
        if vm_pad.sum().item() != 0:
            if valid_mask_errors < max_errors_to_show:
                print(f"\n[ERROR] Batch {batch_idx}, Sample {i}: valid_mask padding not all 0!")
                print(f"  last_valid_pos={last_valid_pos}, max_len={L}")
                print(f"  valid_mask[{last_valid_pos + 1}:] = {vm_pad.tolist()}")
            valid_mask_errors += 1
            error_count += 1
        
        # 检查 label_ids 的 padding 部分
        label_pad = label_ids[i, last_valid_pos + 1:]
        label_pad_count += label_pad.numel()
        
        # 如果 valid_mask=0 但 label_ids 不是忽略值
        invalid_labels = label_pad[(label_pad != -100) & (label_pad != 0)]
        if invalid_labels.numel() > 0:
            if label_pad_errors < max_errors_to_show:
                print(f"\n[WARN] Batch {batch_idx}, Sample {i}: Label padding may not be ignored properly")
                print(f"  Padding labels: {label_pad.tolist()}")
            label_pad_errors += 1

# 5. 输出报告
print("\n" + "="*40)
print("[TEST] DATALOADER VALIDATION REPORT")
print("="*40)

if error_count > 0:
    print(f"❌ Found {error_count} errors in total.")
    print(f"   - valid_mask padding errors: {valid_mask_errors}")
    print(f"   - label padding warnings: {label_pad_errors}")
    print(f"   (Displayed first {min(error_count, max_errors_to_show)} errors above)")
else:
    print("✅ All batches passed validation!")

print(f"\n📊 Batch Statistics:")
print(f"   Total batches: {batch_count}")
print(f"   Total samples processed: {batch_count * batch_size}")

print(f"\n📏 Sequence Length Distribution (after padding):")
for length, count in sorted(length_counter.items()):
    print(f"   Length {length:2d}: {count:5d} samples ({count/(batch_count*batch_size)*100:.1f}%)")

# 6. 最后抽样检查最后一个 batch 的详细内容
print(f"\n🔍 Sanity Check (Last Batch):")
last_batch = batch  # 循环结束后 batch 是最后一个
print(f"   input_ids shape: {last_batch['input_ids'].shape}")
print(f"   label_ids shape: {last_batch['label_ids'].shape}")
print(f"   valid_mask[0]: {last_batch['valid_mask'][0].tolist()}")
print(f"   label_ids[0]: {last_batch['label_ids'][0].tolist()}")

# 检查第一行的 padding 区域
first_row_vm = last_batch["valid_mask"][0]
first_row_labels = last_batch["label_ids"][0]
seq_len = int(first_row_vm.sum().item())
if seq_len < len(first_row_vm):
    print(f"   Padding region (pos {seq_len} onwards):")
    print(f"     valid_mask: {first_row_vm[seq_len:].tolist()}")
    print(f"     label_ids: {first_row_labels[seq_len:].tolist()}")

print("\n[TEST] done")