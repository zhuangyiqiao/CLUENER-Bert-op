import sys
from pathlib import Path
from torch.utils.data import DataLoader

# 添加 src 目录到 Python 路径，确保模块可以导入
ROOT = Path(__file__).resolve().parent.parent  # ner-crf/
sys.path.insert(0, str(ROOT / "src"))

from config.config_parser import load_config
from data.processor import CluenerProcessor
from data.label_map import build_label_map_from_config
from data.dataset import CluenerBertDataset
from data.collate import DataCollatorForBertCrf

# 加载配置文件路径
config_path = ROOT / "configs" / "bert_crf.json"
cfg = load_config(str(config_path))
lm = build_label_map_from_config(cfg)

processor = CluenerProcessor(Path(cfg["data_dir"]))
train_examples = processor.get_train_examples()

ds = CluenerBertDataset(
    examples=train_examples,
    pretrained_name=cfg["model"]["pretrained_name"],
    label_map=lm,
    max_length=64
)

collator = DataCollatorForBertCrf(
    pretrained_name=cfg["model"]["pretrained_name"],
    label_map=lm
)

dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collator)

batch = next(iter(dl))
print("input_ids:", batch["input_ids"].shape, batch["input_ids"].dtype)
print("attention_mask:", batch["attention_mask"].shape)
print("label_ids:", batch["label_ids"].shape)
print("valid_mask:", batch["valid_mask"].shape)

# 检查 padding 的 valid_mask 应该是 0
print("valid_mask row0 tail:", batch["valid_mask"][0, -10:].tolist())