import os
import json
import argparse
from typing import Dict, List, Tuple, Any

import torch
from transformers import AutoTokenizer

IGNORE_INDEX = -100

# CLUENER 固定 10 类（建议固定，保证label顺序稳定）
TYPES = ["address", "book", "company", "game", "government",
         "movie", "name", "organization", "position", "scene"]


def make_label_map(scheme: str = "bios") -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    scheme: "bio" or "bios"
    BIO:  O + 10*(B,I) => 21
    BIOS: O + 10*(B,I,S) => 31   (注意：你要求的是 BIOS，不是 BIOES)
    """
    assert scheme in ("bio", "bios")
    labels = ["O"]
    for t in TYPES:
        labels.append(f"B-{t}")
        labels.append(f"I-{t}")
        if scheme == "bios":
            labels.append(f"S-{t}")

    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label


def build_token_tags_from_spans(
    text: str,
    label_dict: Dict[str, Any],
    offsets: List[Tuple[int, int]],
    scheme: str = "bios",
) -> List[str]:
    """
    text: 原句
    label_dict: CLUENER格式 {"name":{"叶老桂":[[9,11]]}, ...}  (span是char级，end inclusive)
    offsets: tokenizer返回的 offset_mapping，list[(start,end_exclusive)]
    scheme: "bio" or "bios"
    return: tags: list[str] 长度=token数, 取值 O / B-x / I-x / S-x
    """
    tags = ["O"] * len(offsets)

    def char_span_to_token_span(cs: int, ce_inclusive: int):
        ce = ce_inclusive + 1  # inclusive -> exclusive
        idxs = []
        for i, (ts, te) in enumerate(offsets):
            # special tokens may have (0,0) or invalid offsets
            if ts is None or te is None or te <= ts:
                continue
            # overlap between [ts,te) and [cs,ce)
            if te <= cs or ts >= ce:
                continue
            idxs.append(i)
        if not idxs:
            return None
        return idxs[0], idxs[-1]

    for etype, ents in (label_dict or {}).items():
        if etype not in TYPES:
            # 不认识的类型先跳过（保险）
            continue
        for ent_text, spans in (ents or {}).items():
            for start, end in spans:
                if not isinstance(start, int) or not isinstance(end, int):
                    continue
                if start < 0 or end < start or end >= len(text):
                    continue

                r = char_span_to_token_span(start, end)
                if r is None:
                    continue
                ts, te = r

                if scheme == "bios" and ts == te:
                    # single-token entity
                    if tags[ts] == "O":
                        tags[ts] = f"S-{etype}"
                    continue

                if tags[ts] == "O":
                    tags[ts] = f"B-{etype}"
                for j in range(ts + 1, te + 1):
                    if tags[j] == "O":
                        tags[j] = f"I-{etype}"

    return tags


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def convert_split(
    in_path: str,
    out_path: str,
    tokenizer,
    label2id: Dict[str, int],
    scheme: str,
    max_length: int,
    debug_n: int = 0,
):
    input_ids_list = []
    attn_list = []
    labels_list = []

    for idx, sample in enumerate(read_jsonl(in_path)):
        text = sample.get("text", "")
        label = sample.get("label", {}) or {}

        enc = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True,
            return_attention_mask=True,
        )
        offsets = enc["offset_mapping"]
        attn = enc["attention_mask"]
        ids = enc["input_ids"]

        tags = build_token_tags_from_spans(text, label, offsets, scheme=scheme)

        # 构造 labels：special token / padding 用 -100
        labels = []
        for i, (s, e) in enumerate(offsets):
            if attn[i] == 0:
                labels.append(IGNORE_INDEX)
            elif s == 0 and e == 0:
                # [CLS]/[SEP] 等特殊token通常offset是(0,0)
                labels.append(IGNORE_INDEX)
            else:
                labels.append(label2id[tags[i]])

        input_ids_list.append(ids)
        attn_list.append(attn)
        labels_list.append(labels)

        # debug：打印前 debug_n 条
        if debug_n > 0 and idx < debug_n:
            toks = tokenizer.convert_ids_to_tokens(ids)
            print("\n=== DEBUG SAMPLE ===")
            print("TEXT:", text)
            print("TOK/TAG (first 80 tokens):")
            show = []
            for t, lab, am, off in zip(toks, labels, attn, offsets):
                if am == 0:
                    break
                tag = "IGN" if lab == IGNORE_INDEX else list(label2id.keys())[list(label2id.values()).index(lab)]
                show.append(f"{t}/{tag}/{off}")
            print(" ".join(show[:80]))

    feats = {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "attention_mask": torch.tensor(attn_list, dtype=torch.long),
        "labels": torch.tensor(labels_list, dtype=torch.long),
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(
        {
            "features": feats,
            "label2id": label2id,
            "id2label": {int(k): v for k, v in {i: l for i, l in enumerate(sorted(label2id, key=label2id.get))}.items()},
            "model": tokenizer.name_or_path,
            "max_length": max_length,
            "data_path": in_path,
            "scheme": scheme,
        },
        out_path
    )
    print(f"[OK] saved: {out_path} -> {tuple(feats['input_ids'].shape)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data", help="Directory containing train.json/dev.json/test.json (jsonl)")
    ap.add_argument("--out_dir", default="cache_bios", help="Output cache directory")
    ap.add_argument("--model", default="hfl/chinese-bert-wwm-ext", help="Tokenizer/model name for tokenization")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--scheme", choices=["bio", "bios"], default="bios")
    ap.add_argument("--debug_n", type=int, default=0, help="Print first N samples token/tag/offset for sanity check")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    label2id, id2label = make_label_map(args.scheme)

    for split in ["train", "dev", "test"]:
        in_path = os.path.join(args.data_dir, f"{split}.json")
        if not os.path.exists(in_path):
            raise FileNotFoundError(f"Missing file: {in_path} (expected JSONL)")
        out_path = os.path.join(args.out_dir, f"{split}.pt")
        convert_split(
            in_path=in_path,
            out_path=out_path,
            tokenizer=tokenizer,
            label2id=label2id,
            scheme=args.scheme,
            max_length=args.max_length,
            debug_n=args.debug_n if split == "train" else 0,
        )

    print("[DONE] BIOS cache prepared in:", args.out_dir)


if __name__ == "__main__":
    main()
