```markdown
# CLUENER NER 实验进展报告：从训练指标到真实提交指标的工程闭环构建

**日期**：2026-02-04  
**数据集**：CLUENER2020  
**模型框架**：BERT-wwm-ext + TokenClassification（PyTorch, HuggingFace）  
**标注方案**：BIOS  
**最大序列长度**：128  

---

## 摘要

本次实验的核心目标，并非简单提升模型在训练日志中的 F1 指标，而是**打通从训练评估到最终提交格式评估的一致性闭环**。在初始阶段，模型在训练过程中表现出较高的实体级 F1（≈0.76），但在将预测结果转换为 CLUENER 官方提交格式后，在验证集上的 F1 急剧下降至≈0.54。通过系统排查，我们定位到问题根源在于**预测阶段文本解码方式与原始数据文本不一致，导致 span 对齐全面失效**。

通过重构预测脚本，使预测阶段严格基于原始 `json` 文本并利用 `offset_mapping` 完成 token-to-char 的精确映射，我们成功恢复了真实的模型性能。在严格的 span-level 评测下，模型在 dev 集上取得 **Micro-F1 = 0.7586，Macro-F1 = 0.7603**，与训练日志完全一致，证明整个数据流和评测链条已完全对齐。

这一工作的重要性在于：**从“模型训练成功”迈向“工程评测闭环正确”**，这是工业级 NER 系统最关键的能力。

---

## 1. 背景与目标

CLUENER 是一个典型的中文细粒度实体识别数据集，包含 10 类实体，标注格式为 span（字符级起止位置）。官方基线使用 TensorFlow + CRF，而本实验采用 PyTorch + HuggingFace 进行复现与优化。

今日实验的直接动机源于一个异常现象：

> 训练阶段 entity-F1 ≈ 0.76，但转换为提交格式 json 后，在 dev 上的 F1 ≈ 0.54。

这意味着模型本身并非性能不足，而是**预测结果与原始文本发生了系统性错位**。因此，本次实验的核心目标是：

- 找出训练评估与提交评估不一致的原因
- 修复预测与 span 解码逻辑
- 建立一个可复现、可提交、可验证的完整 NER 工程闭环

---

## 2. 实验方法与设置

### 模型结构

- Backbone：`hfl/chinese-bert-wwm-ext`
- Head：`AutoModelForTokenClassification`
- 解码方式：Argmax（无 CRF）

### 标注方案

- 从原始 span 标注转换为 **BIOS**
- 使用 BIOS 的原因是支持单字符实体，提高边界表达能力

### 训练配置

| 参数 | 设置 |
|---|---|
| Max Length | 128 |
| Batch Size | 16 |
| Epoch | 5 |
| Learning Rate | 3e-5 |
| Warmup Ratio | 0.1 |
| Optimizer | AdamW |
| Grad Clip | 1.0 |
| Scheduler | Linear Warmup |

### 数据流

```

json(span) → BIOS → tokenizer → pt缓存 → 训练 → 预测 → span还原 → json

```

---

## 3. 实验过程与结果

### 初始异常

使用旧版预测脚本生成 dev 预测：

```

[TEXT CHECK] mismatch lines: 1343/1343
Micro-F1 ≈ 0.54

```

说明预测 json 中的文本 **完全不等于** 原始 dev 文本。

### 问题根因

预测阶段使用：

```

tokenizer.decode(input_ids)

```

重建文本，导致：

- `[UNK]`
- 空格处理
- 标点变化

从而使所有 span 下标失效。

### 关键修复

重写预测逻辑：

```

读取 data/dev.json 原文 text
tokenizer(text, return_offsets_mapping=True)
token-level tag → offsets → char-level tag
char-level tag → span

```

### 修复后评测结果

```

===== OVERALL (micro) =====
TP=2379 FP=821 FN=693
P=0.7434 R=0.7744 F1=0.7586
Macro-F1=0.7603
[TEXT CHECK] mismatch lines: 0/1343

```

### 各类别表现

| 实体 | F1 |
|---|---|
| name | 0.864 |
| movie | 0.839 |
| game | 0.793 |
| government | 0.783 |
| position | 0.768 |
| organization | 0.759 |
| company | 0.751 |
| scene | 0.704 |
| address | 0.578 |

### 可视化建议

- 各实体 F1 柱状图
- 训练 loss 与 dev F1 曲线
- 修复前后 F1 对比表

---

## 4. 分析与讨论

### 是否达到预期？

是的。修复后，dev F1 与训练日志完全一致，证明模型本身没有问题。

### 为什么之前 F1 极低？

典型的 NER 工程陷阱：

> token-level 预测正确，但 char-level span 错位

这会导致 FP 爆炸（尤其在 game / address 类别）。

### 观察到的现象

- `name`、`movie`、`game` 表现极好，说明模型对连续实体学习充分
- `address` 较低，可能与实体长度不固定、样本稀疏有关
- `scene` 中等，存在边界模糊问题

### 对 MISSING / UNEXPECTED 的解释

加载预训练 BERT 时：

```

classifier.weight MISSING

```

这是正常现象，分类头随机初始化，不影响效果。

---

## 5. 结论与后续工作

### 今日最重要结论

> **完成了从训练 → 预测 → span → json → scorer 的完整工程闭环**

这一步远比简单提高 F1 更关键。

### 当前模型真实能力

> Dev Micro-F1 = 0.7586（可信，可提交）

### 下一步优化方向

1. 替换 backbone 为 `roberta-wwm-large-ext`
2. 引入 CRF + Viterbi 提升边界稳定性
3. 针对 `address` / `scene` 做数据与模型层面的改进

---

## 总结

错误**：训练评估与提交评估不一致。已修复。
```
