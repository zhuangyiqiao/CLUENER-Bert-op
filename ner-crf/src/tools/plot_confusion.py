import json
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ===== 1. 改成你的预测文件路径 =====
pred_file = "outputs/bert_crf_20260311_170658/dev_predictions.jsonl"

# ===== 2. 实体类别列表（不含 O）=====
labels = [
    "address",
    "book",
    "company",
    "game",
    "government",
    "movie",
    "name",
    "organization",
    "position",
    "scene",
]

gold_labels = []
pred_labels = []

def normalize(label: str) -> str:
    """
    BIO -> entity type
    B-address / I-address -> address
    O -> O
    """
    if label == "O":
        return "O"
    if "-" in label:
        return label.split("-", 1)[1]
    return label

# ===== 3. 读取预测文件 =====
with open(pred_file, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        gold = data["gold"]
        pred = data["pred"]

        for g, p in zip(gold, pred):
            g_norm = normalize(g)
            p_norm = normalize(p)

            # 去掉 O-O，避免矩阵被无信息的非实体 token 淹没
            if g_norm == "O" and p_norm == "O":
                continue

            # 只保留至少一边是实体的 token
            gold_labels.append(g_norm)
            pred_labels.append(p_norm)

# ===== 4. 构造用于分析的完整标签集（包含 O）=====
labels_with_o = labels + ["O"]

cm = confusion_matrix(gold_labels, pred_labels, labels=labels_with_o)

# ===== 5. 行归一化（每个真实类别被预测成什么的比例）=====
cm = cm.astype(np.float64)
row_sums = cm.sum(axis=1, keepdims=True)

# 避免除零
cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

# ===== 6. 画图 =====
plt.figure(figsize=(11, 9))
sns.heatmap(
    cm_norm,
    xticklabels=labels_with_o,
    yticklabels=labels_with_o,
    cmap="Blues",
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    cbar=True
)

plt.xlabel("Predicted")
plt.ylabel("Gold")
plt.title("NER Entity-Level Normalized Confusion Matrix")

plt.tight_layout()
plt.savefig("confusion_matrix_entity_normalized.png", dpi=300)
print("Saved: confusion_matrix_entity_normalized.png")

# ===== 7. 打印 top confusion pairs（忽略对角线）=====
print("\nTop confusion pairs (count):")
pairs = []
for i, gold_name in enumerate(labels_with_o):
    for j, pred_name in enumerate(labels_with_o):
        if i == j:
            continue
        count = int(confusion_matrix(gold_labels, pred_labels, labels=labels_with_o)[i, j])
        if count > 0:
            pairs.append((gold_name, pred_name, count))

pairs.sort(key=lambda x: x[2], reverse=True)

for gold_name, pred_name, count in pairs[:20]:
    print(f"{gold_name:>12} -> {pred_name:<12} : {count}")