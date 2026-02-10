import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

log_path = Path(r"d:\cluener-ner\CLUENER-Bert-op\notes\train1-log")
out_png = log_path.with_name("train1_log_plots.png")

# 读取逐行 JSON
records = []
with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
df = pd.DataFrame.from_records(records)

# 强制数值类型，并添加行索引作为 x 轴（保证原始顺序）
num_cols = ["step", "epoch", "train_loss", "loss", "token_acc", "ent_p", "ent_r", "ent_f1"]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
df["entry"] = range(len(df))

# 计算累积 step 作为横坐标（当 step 在新一轮重置时累加上一次的最大值）
offset = 0
prev = -1
step_cum = []
for s in df["step"].fillna(0).astype(int):
    if s <= prev:
        offset += prev
    step_cum.append(s + offset)
    prev = s
df["step_cum"] = step_cum

# 添加：平滑窗口并计算移动平均，避免 NameError / KeyError
window = 5
for c in ["train_loss", "loss", "token_acc", "ent_f1"]:
    if c in df.columns:
        df[c + "_ma"] = df[c].rolling(window, min_periods=1).mean()

# 使用 step_cum 作为 x 轴（也可改为 "entry" 或 "epoch"）
xcol = "step_cum"

# 绘图
fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# Loss
if "train_loss" in df.columns and "loss" in df.columns:
    axs[0].plot(df[xcol], df["train_loss"], alpha=0.4, label="train_loss")
    axs[0].plot(df[xcol], df["loss"], alpha=0.4, label="loss")
    axs[0].plot(df[xcol], df["train_loss_ma"], linestyle="--", label=f"train_loss_{window}MA")
    axs[0].plot(df[xcol], df["loss_ma"], linestyle="--", label=f"loss_{window}MA")
    axs[0].set_ylabel("loss")
    axs[0].legend()

# Token accuracy
if "token_acc" in df.columns:
    axs[1].plot(df[xcol], df["token_acc"], marker="o", markersize=3, label="token_acc", alpha=0.7)
    axs[1].plot(df[xcol], df["token_acc_ma"], linestyle="--", label=f"token_acc_{window}MA")
    axs[1].set_ylabel("token_acc")
    axs[1].legend()

# Precision / Recall / F1
prf_cols = [c for c in ["ent_p", "ent_r", "ent_f1"] if c in df.columns]
if prf_cols:
    for c in prf_cols:
        axs[2].plot(df[xcol], df[c], marker=".", label=c)
    # 平滑 F1 如果存在
    if "ent_f1_ma" in df.columns:
        axs[2].plot(df[xcol], df["ent_f1_ma"], linestyle="--", label=f"ent_f1_{window}MA")
    axs[2].set_ylabel("entity metrics")
    axs[2].legend()

axs[2].set_xlabel(f"{xcol} (累计 step)")
plt.tight_layout()
plt.savefig(out_png, dpi=150)
print(f"Saved plot to: {out_png}")
plt.show()