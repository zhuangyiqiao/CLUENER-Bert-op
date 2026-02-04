# script/predict_to_cluener_json_v2.py
# coding: utf-8

import os
import json
import argparse
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification


IGNORE_INDEX = -100


def load_pt(path: str):
    obj = torch.load(path, map_location="cpu")
    feats = obj["features"]
    label2id = obj["label2id"]
    id2label = obj["id2label"]
    meta = {k: obj.get(k) for k in ["model", "max_length", "data_path"]}
    return feats, label2id, id2label, meta


def make_dataset(feats: Dict[str, torch.Tensor]) -> TensorDataset:
    # ✅ 兼容 token_type_ids
    if "token_type_ids" in feats:
        return TensorDataset(feats["input_ids"], feats["attention_mask"], feats["token_type_ids"])
    return TensorDataset(feats["input_ids"], feats["attention_mask"])


def read_jsonl_texts(path: str) -> List[str]:
    assert os.path.exists(path), f"jsonl not found: {path}"
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj.get("text", ""))
    return texts


# -------------------------
# ✅ token-level tags -> token spans (etype, token_start, token_end)
# 支持 BIO / BIOS (S-)
# -------------------------
def token_tags_to_token_spans(tags: List[str]) -> List[Tuple[str, int, int]]:
    spans = []
    cur_type = None
    cur_start = None

    def close(end_idx: int):
        nonlocal cur_type, cur_start
        if cur_type is not None and cur_start is not None and end_idx >= cur_start:
            spans.append((cur_type, cur_start, end_idx))
        cur_type, cur_start = None, None

    for i, t in enumerate(tags):
        t = t or "O"
        if t == "O":
            close(i - 1)
            continue

        if t.startswith("S-"):
            close(i - 1)
            spans.append((t[2:], i, i))
            continue

        if t.startswith("B-"):
            close(i - 1)
            cur_type = t[2:]
            cur_start = i
            continue

        if t.startswith("I-"):
            typ = t[2:]
            if cur_type is None:
                # 😵 I- 开头：当作 B-
                cur_type = typ
                cur_start = i
            elif typ != cur_type:
                # 😵 类型跳变：关旧开新
                close(i - 1)
                cur_type = typ
                cur_start = i
            continue

        # unknown -> O
        close(i - 1)

    close(len(tags) - 1)
    return spans


# -------------------------
# ✅ 用 offsets_mapping 把 token span -> char span
# offsets: List[Tuple[int,int]]  (start, end)  end 是“开区间”
# -------------------------
def token_span_to_char_span(
    token_span: Tuple[str, int, int],
    offsets: List[Tuple[int, int]],
) -> Optional[Tuple[str, int, int]]:
    etype, ts, te = token_span

    # 找到 span 覆盖 token 的有效 offset（过滤 special tokens 的 (0,0)）
    starts = []
    ends = []
    for i in range(ts, te + 1):
        if i < 0 or i >= len(offsets):
            continue
        s, e = offsets[i]
        if s == 0 and e == 0:
            continue
        starts.append(s)
        ends.append(e)

    if not starts or not ends:
        return None

    char_s = min(starts)
    char_e = max(ends) - 1  # end 是开区间，所以 -1 变成闭区间
    if char_e < char_s:
        return None
    return (etype, char_s, char_e)


def spans_to_cluener_label(text: str, spans: List[Tuple[str, int, int]]) -> Dict[str, Dict[str, List[List[int]]]]:
    """
    CLUENER label 格式：
    {etype: {entity_text: [[start,end], ...]}}
    """
    label_obj: Dict[str, Dict[str, List[List[int]]]] = {}
    for etype, s, e in spans:
        if s < 0 or e >= len(text) or e < s:
            continue
        ent = text[s : e + 1]
        if not ent:
            continue
        label_obj.setdefault(etype, {})
        label_obj[etype].setdefault(ent, [])
        label_obj[etype][ent].append([s, e])
    return label_obj


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="cache_bios", help="where {split}.pt is")
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--data_json", default=None, help="✅ 原始 jsonl（比如 data/dev.json 或 data/test.json）")
    ap.add_argument("--model_dir", default="outputs/bert_bios/best")
    ap.add_argument("--out_path", default="cluener_predict.json")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--max_samples", type=int, default=0)
    args = ap.parse_args()

    pt_path = os.path.join(args.cache_dir, f"{args.split}.pt")
    assert os.path.exists(pt_path), f"pt not found: {pt_path}"

    # ✅ 必须提供原始 text 文件（dev/test）
    assert args.data_json is not None, "Please pass --data_json (e.g. data/dev.json or data/test.json)"
    texts = read_jsonl_texts(args.data_json)

    feats, label2id, id2label, meta = load_pt(pt_path)
    n_pt = feats["input_ids"].shape[0]
    if len(texts) != n_pt:
        raise ValueError(f"text lines={len(texts)} != pt samples={n_pt}. Check that {args.data_json} matches {pt_path}")

    ds = make_dataset(feats)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # ✅ tokenizer：用 best 模型目录（与你训练一致）
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)

    # ✅ model
    config = AutoConfig.from_pretrained(
        args.model_dir,
        num_labels=len(label2id),
        id2label={int(k): v for k, v in id2label.items()},
        label2id=label2id,
    )
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir, config=config)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model.to(device)
    model.eval()

    # debug limit
    n = n_pt
    if args.max_samples and args.max_samples > 0:
        n = min(n, args.max_samples)

    all_rows = []
    idx_global = 0

    for batch in dl:
        if idx_global >= n:
            break

        if len(batch) == 3:
            input_ids, attn_mask, token_type_ids = [x.to(device) for x in batch]
            out = model(input_ids=input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids)
        else:
            input_ids, attn_mask = [x.to(device) for x in batch]
            out = model(input_ids=input_ids, attention_mask=attn_mask)

        preds = torch.argmax(out.logits, dim=-1).detach().cpu()  # (bs, seq)

        bs = preds.size(0)
        for bi in range(bs):
            i = idx_global
            if i >= n:
                break

            text = texts[i]

            # ✅ 用原文重新 tokenize，拿 offsets_mapping（关键）
            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                padding="max_length",
                truncation=True,
                max_length=feats["input_ids"].shape[1],
            )

            offsets = enc["offset_mapping"]  # len == max_length
            # 注意：fast tokenizer 会给 special tokens offsets=(0,0)

            pred_ids = preds[bi].tolist()
            token_tags = [id2label[int(x)] for x in pred_ids]

            # 1) token tags -> token spans
            token_spans = token_tags_to_token_spans(token_tags)

            # 2) token span -> char span
            char_spans: List[Tuple[str, int, int]] = []
            for ts in token_spans:
                cs = token_span_to_char_span(ts, offsets)
                if cs is not None:
                    char_spans.append(cs)

            label_obj = spans_to_cluener_label(text, char_spans)

            # ✅ 输出带 text，strict_text_match 才能对齐
            row = {"id": i, "text": text, "label": label_obj}
            all_rows.append(row)

            idx_global += 1

    with open(args.out_path, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[OK] wrote predictions -> {args.out_path}")
    # sanity
    if all_rows:
        r0 = all_rows[0]
        print("\n=== SANITY CHECK SAMPLE ===")
        print("TEXT:", r0["text"][:120])
        # 打印前几个实体
        spans_preview = []
        for etype, ent_map in r0["label"].items():
            for ent_text, pos_list in ent_map.items():
                for (s, e) in pos_list:
                    spans_preview.append((etype, s, e, ent_text))
        print("SPANS:", spans_preview[:10])


if __name__ == "__main__":
    main()

