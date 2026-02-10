# script/score_span_f1.py
# -*- coding: utf-8 -*-
import argparse
import json
from collections import defaultdict

TYPES = [
    "address", "book", "company", "game", "government",
    "movie", "name", "organization", "position", "scene"
]

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def label_to_spanset(sample):
    """
    gold format: {"label": {etype: {mention: [[start,end], ...]}, ...}}
    return: set of (etype, start, end)  (end inclusive)
    """
    spans = set()
    label = sample.get("label", {}) or {}
    for etype, ents in label.items():
        for mention, se_list in ents.items():
            for s, e in se_list:
                spans.add((etype, int(s), int(e)))
    return spans

def pred_to_spanset(sample):
    """
    pred format: {"label": {etype: {mention: [[start,end], ...]}, ...}}
    same as gold
    """
    return label_to_spanset(sample)

def safe_div(a, b):
    return a / b if b != 0 else 0.0

def prf(tp, fp, fn):
    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)
    f1 = safe_div(2 * p * r, p + r) if (p + r) != 0 else 0.0
    return p, r, f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/dev.json", help="gold dev jsonl path")
    ap.add_argument("--pred", default="cluener_dev_pred.json", help="pred jsonl path")
    ap.add_argument("--strict_text_match", action="store_true",
                    help="if set, also verify text field matches line-by-line")
    args = ap.parse_args()

    gold = load_jsonl(args.gold)
    pred = load_jsonl(args.pred)

    if len(gold) != len(pred):
        raise ValueError(f"len mismatch: gold={len(gold)} pred={len(pred)}")

    # per-type counts
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    # overall counts (micro)
    TP = FP = FN = 0

    bad_text = 0
    for i, (g, p) in enumerate(zip(gold, pred)):
        gt = g.get("text", "")
        pt = p.get("text", "")
        if args.strict_text_match and gt != pt:
            bad_text += 1

        gset = label_to_spanset(g)
        pset = pred_to_spanset(p)

        # overall micro
        inter = gset & pset
        TP += len(inter)
        FP += len(pset - gset)
        FN += len(gset - pset)

        # per type
        for et in TYPES:
            g_et = {x for x in gset if x[0] == et}
            p_et = {x for x in pset if x[0] == et}
            inter_et = g_et & p_et
            tp[et] += len(inter_et)
            fp[et] += len(p_et - g_et)
            fn[et] += len(g_et - p_et)

    # overall micro PRF
    P, R, F1 = prf(TP, FP, FN)

    # per-type PRF + macro F1
    per = {}
    f1_list = []
    for et in TYPES:
        p_, r_, f1_ = prf(tp[et], fp[et], fn[et])
        per[et] = (p_, r_, f1_, tp[et], fp[et], fn[et])
        f1_list.append(f1_)
    macro_f1 = sum(f1_list) / len(f1_list)

    print("===== OVERALL (micro) =====")
    print(f"TP={TP} FP={FP} FN={FN}")
    print(f"P={P:.4f} R={R:.4f} F1={F1:.4f}")
    print(f"Macro-F1(10 types avg)={macro_f1:.4f}")

    if args.strict_text_match:
        print(f"[TEXT CHECK] mismatch lines: {bad_text}/{len(gold)}")

    print("\n===== PER TYPE =====")
    print("type\tP\tR\tF1\tTP\tFP\tFN")
    for et in TYPES:
        p_, r_, f1_, tpi, fpi, fni = per[et]
        print(f"{et}\t{p_:.4f}\t{r_:.4f}\t{f1_:.4f}\t{tpi}\t{fpi}\t{fni}")

if __name__ == "__main__":
    main()
