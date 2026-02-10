import os
import json
import argparse
from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer

ENTITY_TYPES = [
    "address", "book", "company", "game", "government",
    "movie", "name", "organization", "position", "scene"
]

def build_label_maps(entity_types: List[str]) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    # BIO scheme for token classification
    labels = ["O"]
    for t in entity_types:
        labels.append(f"B-{t}")
        labels.append(f"I-{t}")
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    return labels, label2id, id2label

def span_to_char_tags(text: str, label_dict: Dict) -> List[str]:
    """Convert CLUENER span labels to char-level BIO tags. end index is inclusive."""
    tags = ["O"] * len(text)
    if not label_dict:
        return tags

    for etype, entities in label_dict.items():
        for ent, spans in entities.items():
            for start, end in spans:
                # basic guard
                if not isinstance(start, int) or not isinstance(end, int):
                    continue
                if start < 0 or end < start or end >= len(text):
                    continue

                # Set B at start if free
                if tags[start] == "O":
                    tags[start] = f"B-{etype}"
                # Fill I for the rest
                for i in range(start + 1, end + 1):
                    if tags[i] == "O":
                        tags[i] = f"I-{etype}"
    return tags

def token_label_from_offset(offset: Tuple[int, int], char_tags: List[str]) -> str:
    """Map a token span (start,end_exclusive) to a BIO tag."""
    s, e = offset
    if s == e:
        return None  # special tokens
    chunk = char_tags[s:e]  # end exclusive
    if all(t == "O" for t in chunk):
        return "O"

    # choose first non-O inside token
    for idx, t in enumerate(chunk):
        if t != "O":
            ent_type = t.split("-", 1)[1]
            # If token starts exactly at a B- position, keep B-
            if idx == 0 and t.startswith("B-"):
                return f"B-{ent_type}"
            else:
                return f"I-{ent_type}"
    return "O"

def read_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def build_features(
    data_path: str,
    tokenizer,
    label2id: Dict[str, int],
    max_length: int
) -> Dict[str, torch.Tensor]:
    samples = read_jsonl(data_path)

    all_input_ids = []
    all_attention_mask = []
    all_labels = []

    for ex in samples:
        text = ex["text"]
        label_dict = ex.get("label", {})  # test.json may have no label
        char_tags = span_to_char_tags(text, label_dict)

        enc = tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

        offsets = enc["offset_mapping"]
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        labels = []
        for off, att in zip(offsets, attention_mask):
            # pad tokens: we also ignore in loss
            if att == 0:
                labels.append(-100)
                continue

            tag = token_label_from_offset(off, char_tags)
            if tag is None:  # [CLS]/[SEP] or special
                labels.append(-100)
            else:
                labels.append(label2id.get(tag, label2id["O"]))

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_labels.append(labels)

    return {
        "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(all_attention_mask, dtype=torch.long),
        "labels": torch.tensor(all_labels, dtype=torch.long),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hfl/chinese-bert-wwm-ext")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="cache")
    ap.add_argument("--max_length", type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    labels, label2id, id2label = build_label_maps(ENTITY_TYPES)
    print("Label size:", len(labels))
    print("Example labels:", labels[:6], "...")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    for split in ["train", "dev", "test"]:
        in_path = os.path.join(args.data_dir, f"{split}.json")
        if not os.path.exists(in_path):
            print(f"[SKIP] {in_path} not found")
            continue

        feats = build_features(in_path, tokenizer, label2id, args.max_length)
        out_path = os.path.join(args.out_dir, f"{split}.pt")
        torch.save(
            {
                "features": feats,
                "label2id": label2id,
                "id2label": id2label,
                "model": args.model,
                "max_length": args.max_length,
                "data_path": in_path,
            },
            out_path
        )
        print(f"[OK] saved {split}: {out_path}  ->  {feats['input_ids'].shape}")

if __name__ == "__main__":
    main()
