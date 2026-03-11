import json
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]   # 指向 ner-crf/
sys.path.insert(0, str(ROOT))

from src.data.processor import CluenerProcessor
from src.data.label_map import build_label_map_from_config
from src.data.dataset import CluenerBertDataset
from src.data.collate import DataCollatorForBertCrf
from src.models.model import BertCrfForNer


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def move_batch_to_device(batch, device):
    moved = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


def extract_gold_sequences(label_ids: torch.Tensor, valid_mask: torch.Tensor):
    label_ids = label_ids.detach().cpu()
    valid_mask = valid_mask.detach().cpu()

    B, T = label_ids.shape
    sequences = []
    for b in range(B):
        ids = []
        for t in range(T):
            if int(valid_mask[b, t].item()) == 1:
                ids.append(int(label_ids[b, t].item()))
        sequences.append(ids)
    return sequences


def main():
    exp_dir = Path("outputs/bert_crf_20260311_170658")   # 改成你的实验目录
    config_path = exp_dir / "config_used.json"
    ckpt_path = exp_dir / "best.pt"
    out_path = exp_dir / "dev_predictions.jsonl"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    device = resolve_device(cfg["train"].get("device", "auto"))
    data_dir = Path(cfg["data_dir"])
    pretrained_name = cfg["model"]["pretrained_name"]
    max_length = cfg["data"]["max_length"]

    label_map = build_label_map_from_config(cfg)

    processor = CluenerProcessor(data_dir)
    dev_examples = processor.get_dev_examples()

    dev_ds = CluenerBertDataset(
        examples=dev_examples,
        pretrained_name=pretrained_name,
        label_map=label_map,
        max_length=max_length,
    )

    collator = DataCollatorForBertCrf(
        pretrained_name=pretrained_name,
        label_map=label_map,
    )

    dev_loader = DataLoader(
        dev_ds,
        batch_size=cfg["train"]["eval_batch_size"],
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    model = BertCrfForNer(
        pretrained_name=pretrained_name,
        num_tags=label_map.num_labels,
        start_tag_id=label_map.START_ID,
        stop_tag_id=label_map.STOP_ID,
        dropout=cfg["model"].get("dropout", 0.1),
    )

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    with open(out_path, "w", encoding="utf-8") as fout:
        with torch.no_grad():
            for batch in dev_loader:
                raw_batch = batch
                batch = move_batch_to_device(batch, device)

                pred_paths = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    valid_mask=batch["valid_mask"],
                    label_ids=None,
                )

                true_paths = extract_gold_sequences(batch["label_ids"], batch["valid_mask"])

                # 这里假设 batch 里有 tokens 或 text 字段，按你的实际字段名改
                raw_texts = raw_batch.get("text", None)
                raw_tokens = raw_batch.get("tokens", None)

                for i, (pred_ids, true_ids) in enumerate(zip(pred_paths, true_paths)):
                    pred_labels = label_map.decode(pred_ids)
                    true_labels = label_map.decode(true_ids)

                    item = {
                        "gold": true_labels,
                        "pred": pred_labels,
                    }

                    if raw_texts is not None:
                        item["text"] = raw_texts[i]
                    if raw_tokens is not None:
                        item["tokens"] = raw_tokens[i]

                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()