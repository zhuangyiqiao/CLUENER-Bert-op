import json
import os
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from transformers import AutoTokenizer

from src.config.config_parser import load_config
from src.data.label_map import build_label_map_from_config
from src.models.model import BertCrfForNer
from src.utils.tagging import bioes_to_spans, spans_to_dicts


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_checkpoint_to_model(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    # our Trainer saves {"model_state_dict": ...}
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    return model


def build_submit_label(spans) -> Dict[str, Dict[str, List[List[int]]]]:
    """
    Convert spans -> CLUENER label dict:
      {
        "company": {"浙商银行": [[0,3]]},
        "name": {"叶老桂": [[9,11]]}
      }
    """
    label: Dict[str, Dict[str, List[List[int]]]] = {}
    for s in spans:
        ent_type = s.ent_type
        text = s.text
        start, end = s.start, s.end
        label.setdefault(ent_type, {})
        label[ent_type].setdefault(text, [])
        label[ent_type][text].append([start, end])
    return label


def predict_one_text(
    text: str,
    tokenizer,
    model: BertCrfForNer,
    label_map,
    max_length: int,
    device: torch.device
) -> Dict[str, Any]:
    """
    Returns:
      - tokens(chars)
      - tags(char-level BIOES)
      - spans(list)
      - also some debug: bert_tokens, valid_mask
    """
    chars = list(text)

    enc = tokenizer(
        chars,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_tensors=None
    )
    word_ids = enc.word_ids()  # len == seq_len
    input_ids = torch.tensor([enc["input_ids"]], dtype=torch.long, device=device)
    attention_mask = torch.tensor([enc["attention_mask"]], dtype=torch.long, device=device)

    # build valid_mask: 1 only on first subword of each original char; 0 on special/other subwords
    valid_mask_list = []
    prev_word_id = None
    for wid in word_ids:
        if wid is None:
            valid_mask_list.append(0)
        else:
            if wid != prev_word_id:
                valid_mask_list.append(1)
            else:
                valid_mask_list.append(0)
        prev_word_id = wid

    valid_mask = torch.tensor([valid_mask_list], dtype=torch.long, device=device)

    # decode with CRF
    model.eval()
    with torch.no_grad():
        pred_paths = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            valid_mask=valid_mask,
            label_ids=None
        )
    pred_ids_valid = pred_paths[0]  # length == number of valid positions

    pred_tags_valid = label_map.decode(pred_ids_valid)  # ['B-company', 'I-company', ...] for valid positions only

    # map back to char-level tags by word_id
    char_tags = ["O"] * len(chars)
    k = 0
    for t, m in enumerate(valid_mask_list):
        if m != 1:
            continue
        wid = word_ids[t]
        if wid is None:
            continue
        if wid >= len(chars):
            continue
        # pred_tags_valid is in the same order as valid positions
        if k < len(pred_tags_valid):
            char_tags[wid] = pred_tags_valid[k]
        k += 1

    spans = bioes_to_spans(chars, char_tags)

    bert_tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])

    return {
        "text": text,
        "tokens": chars,
        "tag_seq": " ".join(char_tags),
        "entities": spans_to_dicts(spans),  # readable
        "submit_label": build_submit_label(spans),  # CLUENER format
        "debug": {
            "bert_tokens": bert_tokens,
            "valid_mask": valid_mask_list
        }
    }


def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bert_crf.json", type=str)
    parser.add_argument("--ckpt", default=None, type=str, help="checkpoint path, default: <output_dir>/best.pt")
    parser.add_argument("--split", default="test", choices=["test", "dev", "train"], type=str)
    parser.add_argument("--max_samples", default=0, type=int, help="0 means all")
    parser.add_argument("--text", default=None, type=str, help="predict a single text instead of a split file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["train"].get("device", "auto"))

    label_map = build_label_map_from_config(cfg)

    pretrained_name = cfg["model"]["pretrained_name"]
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

    model = BertCrfForNer(
        pretrained_name=pretrained_name,
        num_tags=label_map.num_labels,
        start_tag_id=label_map.START_ID,
        stop_tag_id=label_map.STOP_ID,
        dropout=cfg["model"].get("dropout", 0.1),
    ).to(device)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = args.ckpt if args.ckpt else str(output_dir / "best.pt")
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = load_checkpoint_to_model(model, ckpt_path, device=device)

    max_length = int(cfg["data"]["max_length"])

    # Single text mode
    if args.text is not None:
        out = predict_one_text(args.text, tokenizer, model, label_map, max_length, device)
        print(json.dumps(out["entities"], ensure_ascii=False, indent=2))
        print(json.dumps(out["submit_label"], ensure_ascii=False, indent=2))
        return

    # File mode (train/dev/test jsonl)
    data_dir = Path(cfg["data_dir"])
    split_file = data_dir / f"{args.split}.json"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    # Outputs similar to official:
    # 1) test_prediction.jsonl (debug-readable)
    # 2) test_submit.jsonl (CLUENER submission format)
    pred_out_path = output_dir / f"{args.split}_prediction.jsonl"
    submit_out_path = output_dir / f"{args.split}_submit.jsonl"

    n = 0
    with open(pred_out_path, "w", encoding="utf-8") as f_pred, open(submit_out_path, "w", encoding="utf-8") as f_sub:
        for obj in iter_jsonl(split_file):
            text = obj["text"]
            sample_id = obj.get("id", n)

            res = predict_one_text(text, tokenizer, model, label_map, max_length, device)

            # debug-friendly record
            pred_record = {
                "id": sample_id,
                "text": text,
                "tag_seq": res["tag_seq"],
                "entities": res["entities"],
            }
            f_pred.write(json.dumps(pred_record, ensure_ascii=False) + "\n")

            # submission record
            sub_record = {
                "id": sample_id,
                "label": res["submit_label"]
            }
            f_sub.write(json.dumps(sub_record, ensure_ascii=False) + "\n")

            n += 1
            if args.max_samples > 0 and n >= args.max_samples:
                break

    print(f"[DONE] wrote: {pred_out_path}")
    print(f"[DONE] wrote: {submit_out_path}")


if __name__ == "__main__":
    main()