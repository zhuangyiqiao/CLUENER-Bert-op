import os
import json
import math
import time
import argparse
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, get_linear_schedule_with_warmup

IGNORE_INDEX = -100


def load_pt(path: str):
    obj = torch.load(path, map_location="cpu")
    feats = obj["features"]
    label2id = obj["label2id"]
    id2label = obj["id2label"]
    meta = {k: obj.get(k) for k in ["model", "max_length", "data_path"]}
    return feats, label2id, id2label, meta


def make_dataset(feats: Dict[str, torch.Tensor]) -> TensorDataset:
    return TensorDataset(feats["input_ids"], feats["attention_mask"], feats["labels"])


def bio_entity_spans(tags: List[str]) -> List[Tuple[str, int, int]]:
    """
    Convert BIO tag sequence to entity spans: (type, start, end_inclusive) over token indices.
    """
    spans = []
    cur_type, cur_start = None, None
    for i, t in enumerate(tags):
        if t == "O" or t is None:
            if cur_type is not None:
                spans.append((cur_type, cur_start, i - 1))
                cur_type, cur_start = None, None
            continue
        if t.startswith("B-"):
            if cur_type is not None:
                spans.append((cur_type, cur_start, i - 1))
            cur_type = t[2:]
            cur_start = i
        elif t.startswith("I-"):
            typ = t[2:]
            if cur_type is None:
                # ill-formed, start a new span
                cur_type = typ
                cur_start = i
            elif typ != cur_type:
                # type changed, close and restart
                spans.append((cur_type, cur_start, i - 1))
                cur_type = typ
                cur_start = i
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

    # token-level counts (micro)
    tok_tp = tok_fp = tok_fn = 0

    # entity-level counts (micro)
    ent_tp = ent_fp = ent_fn = 0

    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        input_ids, attn_mask, labels = [x.to(device) for x in batch]
        out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        loss = out.loss
        logits = out.logits

        total_loss += loss.item()
        n_batches += 1

        preds = torch.argmax(logits, dim=-1)

        # token-level micro on non-ignored positions
        mask = labels != IGNORE_INDEX
        gold = labels[mask].view(-1)
        pred = preds[mask].view(-1)

        tok_tp += int((pred == gold).sum().item())
        tok_fp += int((pred != gold).sum().item())
        # tok_fn for token accuracy isn't meaningful; we report token-accuracy-like P/R/F1 as accuracy.
        # We'll treat:
        # P=R=Acc, F1=Acc
        # (Keep this simple to avoid confusion.)

        # entity-level micro
        # For each sample in batch, convert to BIO strings, then compare spans
        bs = input_ids.size(0)
        for i in range(bs):
            # consider only positions where labels != -100
            valid_pos = (labels[i] != IGNORE_INDEX).nonzero(as_tuple=False).squeeze(-1).tolist()
            if not valid_pos:
                continue
            last = valid_pos[-1]

            gold_ids = labels[i, : last + 1].tolist()
            pred_ids = preds[i, : last + 1].tolist()

            gold_tags = [id2label[x] if x != IGNORE_INDEX else "O" for x in gold_ids]
            pred_tags = [id2label[x] if x != IGNORE_INDEX else "O" for x in pred_ids]

            gold_spans = set(bio_entity_spans(gold_tags))
            pred_spans = set(bio_entity_spans(pred_tags))

            ent_tp += len(gold_spans & pred_spans)
            ent_fp += len(pred_spans - gold_spans)
            ent_fn += len(gold_spans - pred_spans)

    avg_loss = total_loss / max(1, n_batches)
    token_acc = tok_tp / max(1, (tok_tp + tok_fp))

    ent_p, ent_r, ent_f1 = f1_from_counts(ent_tp, ent_fp, ent_fn)

    return {
        "loss": avg_loss,
        "token_acc": token_acc,
        "ent_p": ent_p,
        "ent_r": ent_r,
        "ent_f1": ent_f1
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="cache")
    ap.add_argument("--out_dir", default="outputs/bert_bio")
    ap.add_argument("--model_name", default="hfl/chinese-bert-wwm-ext")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--save_best", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_feats, label2id, id2label, meta = load_pt(os.path.join(args.cache_dir, "train.pt"))
    dev_feats, _, _, _ = load_pt(os.path.join(args.cache_dir, "dev.pt"))

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
        label2id=label2id
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
                    **metrics
                }
                print(f"[EVAL] step {global_step} dev_loss {metrics['loss']:.4f} token_acc {metrics['token_acc']:.4f} ent_f1 {metrics['ent_f1']:.4f}")

                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                # save best
                if args.save_best and metrics["ent_f1"] > best_f1:
                    best_f1 = metrics["ent_f1"]
                    best_dir = os.path.join(args.out_dir, "best")
                    os.makedirs(best_dir, exist_ok=True)
                    model.save_pretrained(best_dir)
                    # save tokenizer for later decode
                    tok = AutoTokenizer.from_pretrained(args.model_name)
                    tok.save_pretrained(best_dir)
                    with open(os.path.join(best_dir, "label2id.json"), "w", encoding="utf-8") as f:
                        json.dump(label2id, f, ensure_ascii=False, indent=2)
                    print(f"[SAVE] new best ent_f1={best_f1:.4f} saved to {best_dir}")

        # epoch end eval
        metrics = evaluate(model, dev_loader, id2label, device)
        print(f"[EPOCH END] epoch {epoch} dev_loss {metrics['loss']:.4f} token_acc {metrics['token_acc']:.4f} ent_f1 {metrics['ent_f1']:.4f}")

    # final save
    final_dir = os.path.join(args.out_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tok = AutoTokenizer.from_pretrained(args.model_name)
    tok.save_pretrained(final_dir)
    with open(os.path.join(final_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    print("[DONE] final model saved to:", final_dir)


if __name__ == "__main__":
    main()
