# ✅ train_ner_torch.py  (BIO / BIOS 通用可训练版)
# 你的旧代码基础上，只做“必要修改”，并在修改处用表情标注：
# 🟡 = 新增 / 修改
# 🔵 = 原逻辑保留但加强
# 🔴 = 旧逻辑问题点说明（以注释形式保留）

import os
import json
import time
import argparse
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)

IGNORE_INDEX = -100


def load_pt(path: str):
    obj = torch.load(path, map_location="cpu")
    feats = obj["features"]
    label2id = obj["label2id"]
    id2label = obj["id2label"]
    meta = {k: obj.get(k) for k in ["model", "max_length", "data_path", "scheme"]}
    return feats, label2id, id2label, meta


def make_dataset(feats: Dict[str, torch.Tensor]) -> TensorDataset:
    return TensorDataset(feats["input_ids"], feats["attention_mask"], feats["labels"])


# 🟡 新增：BIO/BIOS 自动兼容的 span 抽取函数
def spans_from_tags(tags: List[str]) -> List[Tuple[str, int, int]]:
    """
    Convert BIO or BIOS tags to entity spans: (type, start, end_inclusive).

    - BIO:  B-xxx / I-xxx / O
    - BIOS: B-xxx / I-xxx / S-xxx / O

    🟡 自动判断：如果序列里出现过 S-，就按 BIOS 处理；否则按 BIO。
    """
    has_s = any((t is not None) and t.startswith("S-") for t in tags)

    spans: List[Tuple[str, int, int]] = []
    cur_type: Optional[str] = None
    cur_start: Optional[int] = None

    for i, t in enumerate(tags):
        if t is None or t == "O":
            if cur_type is not None:
                spans.append((cur_type, cur_start, i - 1))
                cur_type, cur_start = None, None
            continue

        # 🟡 BIOS 的单 token 实体：S-xxx
        if has_s and t.startswith("S-"):
            if cur_type is not None:
                spans.append((cur_type, cur_start, i - 1))
                cur_type, cur_start = None, None
            spans.append((t[2:], i, i))
            continue

        if t.startswith("B-"):
            if cur_type is not None:
                spans.append((cur_type, cur_start, i - 1))
            cur_type = t[2:]
            cur_start = i

        elif t.startswith("I-"):
            typ = t[2:]
            if cur_type is None:
                # 🔵 兼容“坏 BIO”：I- 开头也能开一个 span
                cur_type = typ
                cur_start = i
            elif typ != cur_type:
                spans.append((cur_type, cur_start, i - 1))
                cur_type = typ
                cur_start = i

        else:
            # 🟡 兜底：遇到未知标签，先结束当前 span
            if cur_type is not None:
                spans.append((cur_type, cur_start, i - 1))
                cur_type, cur_start = None, None

    if cur_type is not None:
        spans.append((cur_type, cur_start, len(tags) - 1))

    return spans


def f1_from_counts(tp, fp, fn):
    p = tp / (tp + fp + 1e-12)
    r = tp / (tp + fn + 1e-12)
    f1 = 2 * p * r / (p + r + 1e-12)
    return p, r, f1


@torch.no_grad()
def evaluate(model, dataloader, id2label: Dict[int, str], device):
    model.eval()

    tok_correct = 0
    tok_total = 0

    ent_tp = ent_fp = ent_fn = 0

    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        input_ids, attn_mask, labels = [x.to(device) for x in batch]
        out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        loss = out.loss
        logits = out.logits

        total_loss += float(loss.item())
        n_batches += 1

        preds = torch.argmax(logits, dim=-1)

        # 🔵 token-level accuracy（旧代码把 tok_tp/tok_fp 写成 P/R/F1 的概念会误导）
        mask = labels != IGNORE_INDEX
        gold = labels[mask].view(-1)
        pred = preds[mask].view(-1)
        tok_correct += int((pred == gold).sum().item())
        tok_total += int(gold.numel())

        # entity-level micro
        bs = input_ids.size(0)
        for i in range(bs):
            valid_pos = (labels[i] != IGNORE_INDEX).nonzero(as_tuple=False).squeeze(-1).tolist()
            if not valid_pos:
                continue
            last = valid_pos[-1]

            gold_ids = labels[i, : last + 1].tolist()
            pred_ids = preds[i, : last + 1].tolist()

            gold_tags = [id2label[x] if x != IGNORE_INDEX else "O" for x in gold_ids]
            pred_tags = [id2label[x] if x != IGNORE_INDEX else "O" for x in pred_ids]

            # 🟡 关键修改：用 spans_from_tags 兼容 BIO/BIOS
            gold_spans = set(spans_from_tags(gold_tags))
            pred_spans = set(spans_from_tags(pred_tags))

            ent_tp += len(gold_spans & pred_spans)
            ent_fp += len(pred_spans - gold_spans)
            ent_fn += len(gold_spans - pred_spans)

    avg_loss = total_loss / max(1, n_batches)
    token_acc = tok_correct / max(1, tok_total)

    ent_p, ent_r, ent_f1 = f1_from_counts(ent_tp, ent_fp, ent_fn)

    return {
        "loss": avg_loss,
        "token_acc": token_acc,
        "ent_p": ent_p,
        "ent_r": ent_r,
        "ent_f1": ent_f1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="cache")  # cache 或 cache_bios 都行
    ap.add_argument("--out_dir", default="outputs/bert_run")
    ap.add_argument("--model_name", default="hfl/chinese-bert-wwm-ext")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--save_best", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 🔵 seed（建议同时加上 cudnn 相关，但这里保持你原风格）
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_feats, label2id, id2label, meta = load_pt(os.path.join(args.cache_dir, "train.pt"))

    # ===== 🧪 DEBUG: 检查 CLS / SEP / PAD 的 label =====
    tok = AutoTokenizer.from_pretrained(args.model_name)

    sample_ids = train_feats["input_ids"][0]
    sample_labels = train_feats["labels"][0]
    sample_mask = train_feats["attention_mask"][0]

    tokens = tok.convert_ids_to_tokens(sample_ids.tolist())

    print("\n===== DEBUG SAMPLE =====")
    for i, (t, lid, m) in enumerate(zip(tokens, sample_labels.tolist(), sample_mask.tolist())):
      if t in [tok.cls_token, tok.sep_token] or m == 0:
        print(f"pos={i:3d} token={t:10s} mask={m} label={lid}")
    print("========================\n")
    # ===== 🧪 DEBUG END =====

    dev_feats, _, _, _ = load_pt(os.path.join(args.cache_dir, "dev.pt"))

    # 🟡 新增：打印 scheme，方便你确认读的是 BIO 还是 BIOS
    scheme = meta.get("scheme", "unknown")
    print(f"[INFO] cache_dir={args.cache_dir} scheme={scheme} num_labels={len(label2id)}")

    train_ds = make_dataset(train_feats)
    dev_ds = make_dataset(dev_feats)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False)

    num_labels = len(label2id)

    # config/model
    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label={int(k): v for k, v in id2label.items()},
        label2id=label2id,
    )
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, config=config)
    model.to(device)

    # optimizer/scheduler
    t_total = len(train_loader) * args.epochs
    warmup_steps = int(t_total * args.warmup_ratio)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, t_total)

    best_f1 = -1.0
    global_step = 0

    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    print("Logging to:", log_path)

    model.train()
    for epoch in range(1, args.epochs + 1):
        for batch in train_loader:
            t0 = time.time()
            input_ids, attn_mask, labels = [x.to(device) for x in batch]

            out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss = out.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            if global_step % 20 == 0:
                print(f"epoch {epoch} step {global_step}/{t_total} loss {loss.item():.4f} time {time.time()-t0:.2f}s")

            if global_step % args.eval_steps == 0:
                metrics = evaluate(model, dev_loader, id2label, device)
                record = {
                    "step": global_step,
                    "epoch": epoch,
                    "train_loss": float(loss.item()),
                    **metrics,
                }
                print(
                    f"[EVAL] step {global_step} dev_loss {metrics['loss']:.4f} "
                    f"token_acc {metrics['token_acc']:.4f} ent_f1 {metrics['ent_f1']:.4f}"
                )

                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                # save best
                if args.save_best and metrics["ent_f1"] > best_f1:
                    best_f1 = metrics["ent_f1"]
                    best_dir = os.path.join(args.out_dir, "best")
                    os.makedirs(best_dir, exist_ok=True)
                    model.save_pretrained(best_dir)

                    tok = AutoTokenizer.from_pretrained(args.model_name)
                    tok.save_pretrained(best_dir)

                    with open(os.path.join(best_dir, "label2id.json"), "w", encoding="utf-8") as f:
                        json.dump(label2id, f, ensure_ascii=False, indent=2)

                    # 🟡 新增：把 scheme 也写进去，方便回溯
                    with open(os.path.join(best_dir, "meta.json"), "w", encoding="utf-8") as f:
                        json.dump({"cache_dir": args.cache_dir, "scheme": scheme, "num_labels": num_labels}, f, ensure_ascii=False, indent=2)

                    print(f"[SAVE] new best ent_f1={best_f1:.4f} saved to {best_dir}")

        # epoch end eval
        metrics = evaluate(model, dev_loader, id2label, device)
        print(
            f"[EPOCH END] epoch {epoch} dev_loss {metrics['loss']:.4f} "
            f"token_acc {metrics['token_acc']:.4f} ent_f1 {metrics['ent_f1']:.4f}"
        )

    # final save
    final_dir = os.path.join(args.out_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)

    tok = AutoTokenizer.from_pretrained(args.model_name)
    tok.save_pretrained(final_dir)

    with open(os.path.join(final_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    # 🟡 新增：final 也写 meta
    with open(os.path.join(final_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"cache_dir": args.cache_dir, "scheme": scheme, "num_labels": num_labels}, f, ensure_ascii=False, indent=2)

    print("[DONE] final model saved to:", final_dir)


if __name__ == "__main__":
    main()
