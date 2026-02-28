import sys
from pathlib import Path

print("[TEST] script start")

# 让 Python 能找到 src 包：把 ner-crf 根目录加进 sys.path
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

# 配置文件路径
config_path = ROOT / "configs" / "bert_crf.json"
print("[TEST] config_path =", config_path)

cfg = load_config(str(config_path))
lm = build_label_map_from_config(cfg)

print("[TEST] data_dir =", cfg["data_dir"])
print("[TEST] pretrained =", cfg["model"]["pretrained_name"])

processor = CluenerProcessor(Path(cfg["data_dir"]))
train_examples = processor.get_train_examples()
print("[TEST] num_train_examples =", len(train_examples))

ds = CluenerBertDataset(
    examples=train_examples,
    pretrained_name=cfg["model"]["pretrained_name"],
    label_map=lm,
    max_length=64
)

item = ds[0]
print("[TEST] got item keys =", list(item.keys()))
print("[TEST] len(input_ids) =", len(item["input_ids"]))

# tokenizer tokens（这里如果卡住，说明 HF 下载/缓存/网络问题）
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(cfg["model"]["pretrained_name"])
bert_tokens = tok.convert_ids_to_tokens(item["input_ids"])

print("bert_tokens[:30]:", bert_tokens[:30])
print("valid_mask[:30]:", item["valid_mask"][:30])
print("label_ids[:30]:  ", item["label_ids"][:30])
print("decoded_labels[:30]:", lm.decode(item["label_ids"][:30]))

# 额外：检查 valid_mask=1 的位置是否和原始 labels 对齐
vm = item["valid_mask"]
decoded = lm.decode(item["label_ids"])
print("[CHECK] first 30 positions where valid_mask=1:")
count = 0
for i, m in enumerate(vm):
    if m == 1:
        print(f"  pos={i:02d} token={bert_tokens[i]} label={decoded[i]}")
        count += 1
        if count >= 12:
            break

print("[TEST] done")