# 代码架构与设计思想讲授

> 作为导师给实习生的架构设计课程  
> 目标：从设计原理到实现细节的完整理解

---

## 第一讲: 项目架构的 5 层设计

### 🏗️ 金字塔架构图

```
┌─────────────────────────────────────────────┐
│     Layer 5: 应用层（Application）          │
│   train.py, predict.py - 入口脚本           │
├─────────────────────────────────────────────┤
│     Layer 4: 业务逻辑层（Logic）            │
│ Trainer, model pipeline - 训练和推理流程    │
├─────────────────────────────────────────────┤
│     Layer 3: 模型层（Model）                │
│ BertCrfForNer, CRF - 神经网络定义           │
├─────────────────────────────────────────────┤
│     Layer 2: 数据层（Data）                 │
│ Dataset, Collator, Processor - 数据流       │
├─────────────────────────────────────────────┤
│     Layer 1: 基础设施层（Infrastructure）   │
│ Config, Logger, Utils - 辅助工具            │
└─────────────────────────────────────────────┘
```

### 为什么分层？

**目标**：解耦（Decoupling）
- 每层只关注自己的职责
- 层之间通过清晰的接口通信
- 易于测试和维护

**例子**：
```python
# ❌ 坏的做法：逻辑混在一起
def train():
    # 1. 加载配置
    with open('config.json') as f:
        cfg = json.load(f)
    
    # 2. 加载数据（混杂在训练逻辑中）
    with open('train.json') as f:
        data = []
        for line in f:
            # 处理逻辑
            ...
    
    # 3. 创建模型（直接在训练函数中）
    model = torch.nn.Sequential(...)
    
    # 4. 训练（核心逻辑被数据加载埋没）
    for epoch in range(10):
        ...

# ✅ 好的做法：分层清晰
# Layer 1
config = load_config('config.json')

# Layer 2
processor = CluenerProcessor(data_dir)
dataset = CluenerBertDataset(processor.get_examples(), ...)
loader = DataLoader(dataset, collate_fn=collator)

# Layer 3
model = BertCrfForNer(...)

# Layer 4 (只关注训练逻辑，数据已隐藏在 loader 中)
trainer = Trainer(model, loader, ...)
trainer.train(num_epochs=10)
```

---

## 第二讲: 面向对象设计原则（OOP）

### 原则 1: 单一职责原则（SRP）

**定义**：一个类/函数只有一个改变的理由。

#### 好例子：CluenerBertDataset
```python
class CluenerBertDataset(Dataset):
    """
    职责：将原始样本 (tokens, labels) 转换为 PyTorch 张量
    
    只做：
    - 保存原始数据
    - tokenization
    - label 对齐
    - 返回张量
    """
    def __getitem__(self, idx):
        ex = self.examples[idx]
        # ... 处理逻辑
        return {"input_ids": ..., "label_ids": ..., ...}
```

#### 坏例子（违反 SRP）
```python
class CluenerDataset(Dataset):
    def __getitem__(self, idx):
        # ❌ 职责过多
        ex = self.examples[idx]
        
        # 职责 1: 数据加载
        text = ex["text"]
        
        # 职责 2: tokenization
        tokens = tokenize(text)
        
        # 职责 3: label 编码
        labels = encode_labels(...)
        
        # 职责 4: 数据增强
        text, labels = augment(text, labels)
        
        # 职责 5: 日志记录
        logger.info(f"Loaded {text}")
        
        # 职责 6: 质量检查
        assert len(tokens) == len(labels)
        
        return ...
```

**改进**：分离职责
```python
class CluenerDataset(Dataset):
    def __getitem__(self, idx):
        return self.processor.get_item(idx)

class DataProcessor:
    def get_item(self, idx):
        # 只负责加载和转换
        return self._tokenize_and_align(...)

class DataAugmenter:
    def augment(self, text, labels):
        # 只负责数据增强
        return augmented_text, augmented_labels
```

### 原则 2: 开闭原则（OCP）

**定义**：对扩展开放，对修改关闭。

#### 例子：CRF 的不同实现

```python
# ✅ 好的设计：定义接口
class BaseCRF(nn.Module):
    def forward(self, emissions, tags, mask):
        raise NotImplementedError
    
    def decode(self, emissions, mask):
        raise NotImplementedError

# 实现 1: 标准 CRF
class CRF(BaseCRF):
    def forward(self, ...):
        # 标准实现
        ...

# 实现 2: 添加新的 CRF 变体（对扩展开放）
class SemiMarkovCRF(BaseCRF):
    """支持长度约束的 CRF"""
    def forward(self, ...):
        # 新实现
        ...

# 在训练中：不需要修改 train.py，只需切换类
if config.crf_type == "standard":
    crf = CRF(...)
elif config.crf_type == "semi-markov":
    crf = SemiMarkovCRF(...)
```

### 原则 3: 里氏替换原则（LSP）

**定义**：派生类可以替换基类，而不破坏程序。

#### 例子：不同的优化器

```python
# 统一接口
optimizer = AdamW(params, lr=1e-5)  # 可以替换为
optimizer = RAdam(params, lr=1e-5)  # 或其他

# 在 trainer.py 中，不需要改代码，因为都有 step() 方法
for epoch in range(num_epochs):
    for batch in train_loader:
        loss = model(batch)
        loss.backward()
        optimizer.step()  # 任何优化器都支持
```

### 原则 4: 接口隔离原则（ISP）

**定义**：不要强迫客户端依赖它们不需要的接口。

#### 坏例子
```python
class DataLoader:
    def load(self, path): pass
    def augment(self, data): pass
    def validate(self, data): pass
    def visualize(self, data): pass  # ❌ 训练时不需要
    def export_to_csv(self, data): pass  # ❌ 推理时不需要
```

#### 好例子
```python
class DataLoader:
    def load(self, path): pass
    def __getitem__(self, idx): pass

class DataAugmenter:
    def augment(self, data): pass

class DataValidator:
    def validate(self, data): pass

class DataExporter:
    def export_to_csv(self, data): pass

# 训练时只引入需要的
train_loader = DataLoader(dataset)
augmenter = DataAugmenter()  # 可选

# 验证时
validator = DataValidator()
```

### 原则 5: 依赖反转原则（DIP）

**定义**：依赖抽象，而不是具体实现。

#### 实际应用

```python
# ❌ 坏：直接依赖具体类
class Trainer:
    def __init__(self):
        self.model = BertCrfForNer()  # 硬编码
        self.optimizer = AdamW(...)    # 硬编码

# ✅ 好：依赖注入（Dependency Injection）
class Trainer:
    def __init__(self, model, optimizer, scheduler, ...):
        self.model = model           # 接收注入的对象
        self.optimizer = optimizer   # 可以是任何优化器
        # 不关心具体实现，只关心接口

# 使用时
model = BertCrfForNer(...)
optimizer = AdamW(...)
trainer = Trainer(model, optimizer, ...)

# 如果要用不同的模型：
model = RoBertaCrfForNer(...)  # 只需换这一行
optimizer = RAdam(...)          # 只需换这一行
trainer = Trainer(model, optimizer, ...)  # 相同的 trainer！
```

---

## 第三讲: 数据流设计

### 数据操作的 3 个关键点

#### 1️⃣ 转换 vs 应用

```python
# ❌ 坏：过早转换
class Dataset:
    def __init__(self, examples):
        # 在初始化时就转换所有数据到 GPU
        self.examples = [
            torch.tensor(ex["input_ids"]).cuda()
            for ex in examples
        ]
    
    def __getitem__(self, idx):
        return self.examples[idx]

# ❌ 问题：
# - 占用过多 GPU 显存
# - 无法处理不同长度的样本

# ✅ 好：延迟转换
class Dataset:
    def __init__(self, examples):
        self.examples = examples  # 保留原始格式
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        # 返回 Python list/dict，不转换
        return {
            "input_ids": ex["input_ids"],
            "label_ids": ex["label_ids"],
            ...
        }

# 在 collate_fn 中转换
def collate_fn(batch):
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    # 这里才进行 padding 和转换
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True)
    return {"input_ids": input_ids, ...}
```

**为什么这样做？**
- Dataset 返回单个样本（CPU）
- DataLoader batch 处理（可在 CPU 或 GPU）
- Collator 负责 batch 整理（padding、转换）
- 清晰分离职责

#### 2️⃣ Lazy Loading vs Eager Loading

```python
# Eager Loading：全部加载到内存
class Dataset:
    def __init__(self, filepath):
        self.examples = []
        with open(filepath) as f:
            for line in f:
                self.examples.append(json.loads(line))
        # 问题：大数据集可能爆内存

# Lazy Loading：按需加载
class Dataset:
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = None
    
    def __getitem__(self, idx):
        if self.file is None:
            self.file = open(self.filepath)
        
        self.file.seek(0)
        for i, line in enumerate(self.file):
            if i == idx:
                return json.loads(line)
        
        self.file.close()

# 更好的做法：使用中间格式（如 TFRecord 或 PyArrow）
import pyarrow as pa

class Dataset:
    def __init__(self, filepath):
        self.table = pa.ipc.open_file(filepath)
    
    def __getitem__(self, idx):
        return self.table.slice(idx, 1).to_pandas()[0]
```

**选择标准**：
- 数据集 < 1GB：Eager Loading（简单快速）
- 数据集 1-10GB：Lazy Loading（节省内存）
- 数据集 > 10GB：分片处理 + 流式加载

#### 3️⃣ 对齐与验证

```python
# 关键问题：BERT tokenization 可能改变序列长度
original_text = "浙商银行"
original_labels = ["B-company", "I-company", "I-company", "I-company"]

# BERT tokenizer 可能产生
bert_tokens = ["[CLS]", "浙", "商", "##银", "##行", "[SEP]"]

# ❌ 问题：lengths 不匹配
assert len(original_labels) == len(bert_tokens)  # 4 != 6

# ✅ 解决：使用 valid_mask 指标

# 步骤 1: 获取 word_ids
word_ids = tokenizer(original_text).word_ids()
# word_ids = [None, 0, 1, 2, 3, None]
#            (对应 [CLS], 浙, 商, 银, 行, [SEP])

# 步骤 2: 只在 word_id 首次出现时标记为有效
valid_mask = []
prev_word_id = None
for word_id in word_ids:
    if word_id is None:
        valid_mask.append(0)
    elif word_id != prev_word_id:
        valid_mask.append(1)  # 首次出现
    else:
        valid_mask.append(0)
    prev_word_id = word_id
# valid_mask = [0, 1, 1, 1, 1, 0]

# 步骤 3: 标签对齐
aligned_labels = [
    original_labels[word_id] if word_id is not None and valid_mask[i] == 1
    else PAD_LABEL
    for i, word_id in enumerate(word_ids)
]
# aligned_labels = [PAD, B-co, I-co, I-co, I-co, PAD]

# 步骤 4: 验证
assert sum(valid_mask) == len(original_labels)
```

**这个对齐为什么重要？**
- BERT 和原文的 token 数不一致
- valid_mask 让我们知道哪些位置参与标签预测
- CRF 只在 valid 位置计算损失，避免对 subword 重复标注

---

## 第四讲: 模型设计的权衡

### 权衡 1: 模型复杂度 vs 性能提升

```python
# 选项 1: BERT only
class SimpleModel(nn.Module):
    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids, attention_mask)
        logits = self.classifier(bert_out.last_hidden_state)
        # 每个位置独立预测，忽略标签依赖
        return logits.argmax(-1)

# 性能：F1 = 75%
# 速度：快（无 CRF）
# 显存：3GB


# 选项 2: BERT + CRF（当前）
class BertCrfModel(nn.Module):
    def forward(self, input_ids, attention_mask, valid_mask, labels=None):
        bert_out = self.bert(input_ids, attention_mask)
        logits = self.classifier(bert_out.last_hidden_state)
        
        if labels is not None:
            loss = self.crf(logits, labels, valid_mask)
            return loss
        else:
            paths = self.crf.decode(logits, valid_mask)
            return paths

# 性能：F1 = 80%
# 速度：中等（CRF Viterbi）
# 显存：6GB


# 选项 3: Stacked CRF（更复杂）
class StackedCrfModel(nn.Module):
    def __init__(self, num_stacks=2):
        super().__init__()
        self.crfs = nn.ModuleList([CRF(...) for _ in range(num_stacks)])
    
    def forward(self, logits, labels, valid_mask):
        for i, crf in enumerate(self.crfs):
            if i == len(self.crfs) - 1:
                loss = crf(logits, labels, valid_mask)
            else:
                # 中间层的软 viterbi（可微）
                logits = crf.soft_viterbi(logits, valid_mask)
        return loss

# 性能：F1 = 82%（提升 2%，但相对复杂）
# 速度：慢（多层 CRF）
# 显存：8GB
```

**决策矩阵**：
| 方案 | F1 | 复杂度 | 推荐场景 |
|-----|----|--------|---------|
| BERT only | 75% | 低 | 对标签依赖预期不强 |
| BERT+CRF | 80% | 中 | ✅ 标准 NER（推荐） |
| Stacked CRF | 82% | 高 | 有充足显存且要求精度 |

**我的建议**：
- **实习生**：先用 BERT + single layer CRF（容易理解）
- **深化**：试试 stacked CRF（理解多层的收益）
- **优化**：如果精度要求不高，试试 BERT-only（速度优先）

### 权衡 2: 内存 vs 计算速度

```python
# 权衡 1: Batch Size
batch_size = 32  # 当前

# 效果：
# 小 batch (8): 更新频繁 → 可能更好的泛化，但训练慢
# 中 batch (32): 基准
# 大 batch (128): 训练快，但显存不够 → 用梯度累积

# 梯度累积实现
gradient_accumulation_steps = 4
effective_batch_size = batch_size * gradient_accumulation_steps  # 32 * 4 = 128

for step, batch in enumerate(train_loader):  # 每个 batch 32
    loss = model(batch)
    loss = loss / gradient_accumulation_steps
    loss.backward()
    
    if (step + 1) % gradient_accumulation_steps == 0:
        optimizer.step()  # 真正的梯度步


# 权衡 2: 混合精度（FP16）
from torch.cuda.amp import autocast

for step, batch in enumerate(train_loader):
    with autocast():  # 自动转换为 FP16
        loss = model(batch)
    
    loss.backward()
    optimizer.step()

# 效果：显存用量减半，速度提升 1.5-2.5 倍，精度损失 < 1%


# 权衡 3: 模型大小
# 使用更小的 BERT
from transformers import AutoModel

# bert-base-chinese：110M 参数，6GB 显存
model = AutoModel.from_pretrained("bert-base-chinese")

# distilbert-base-chinese：60M 参数，3GB 显存（速度快 2 倍）
model = AutoModel.from_pretrained("distilbert-base-chinese")

# 精度损失：通常 2-5%，可接受
```

---

## 第五讲: 训练动态详解

### Phase 1: Warmup 阶段（第 0-10% 步）

```
学习率：0 → 目标学习率
损失：快速下降（因为初始时权重随机）
梯度：初期可能不稳定

为什么 warmup？
1. 让优化器适应梯度尺度
2. 避免初期的大幅更新破坏预训练权重
3. 稳定初期的训练
```

代码实现：
```python
def build_scheduler(optimizer, num_training_steps, warmup_ratio=0.1):
    warmup_steps = int(num_training_steps * warmup_ratio)
    
    from transformers import get_linear_schedule_with_warmup
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )

# 学习率曲线
import matplotlib.pyplot as plt

lrs_bert = []
lrs_head = []
total_steps = 1000
warmup_ratio = 0.1

for step in range(total_steps):
    warmup_steps = int(total_steps * warmup_ratio)
    
    if step < warmup_steps:
        # 线性增长
        lr_bert = 5e-5 * (step / warmup_steps)
        lr_head = 1e-3 * (step / warmup_steps)
    else:
        # 线性衰减
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        lr_bert = 5e-5 * (1 - progress)
        lr_head = 1e-3 * (1 - progress)
    
    lrs_bert.append(lr_bert)
    lrs_head.append(lr_head)

plt.plot(range(total_steps), lrs_bert, label='BERT LR')
plt.plot(range(total_steps), lrs_head, label='Head LR')
plt.axvline(x=warmup_steps, linestyle='--')
plt.legend()
plt.show()
```

### Phase 2: 主训练阶段（第 10%-100% 步）

```
学习率：目标 → 0
损失：持续下降，但速度变慢（收敛中）

特征：
1. 权重收敛到局部最优
2. 梯度逐渐减小
3. 有时会出现振荡（正常现象）

陷阱：
- 太早停止：没收敛
- 太晚停止：过拟合

解决：使用 Early Stopping
```

代码实现：
```python
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, val_f1):
        if self.best_score is None:
            self.best_score = val_f1
            return False  # 继续训练
        
        if val_f1 > self.best_score + self.min_delta:
            # F1 有显著提升
            self.best_score = val_f1
            self.counter = 0
            return False  # 继续训练
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # 停止训练
            return False

# 在 trainer.py 中使用
early_stopping = EarlyStopping(patience=3)

for epoch in range(num_epochs):
    train_loss = train_one_epoch(...)
    dev_f1 = evaluate(...)
    
    if early_stopping(dev_f1):
        print("Early stopping at epoch", epoch)
        break
```

### Phase 3: CRF 的特殊动态

```python
# CRF loss 的计算过程

# 1. Forward: 计算 log Z（所有路径的配分函数）
#    log_Z 衡量模型对"整体"的理解

# 2. Gold Score: 计算正确路径的分数
#    gold_score 衡量模型对"正确答案"的理解

# 3. NLL = log_Z - gold_score
#    > 0：正确路径分数 < 平均路径分数（不好）
#    < 0：正确路径分数 > 平均路径分数（好）
#    越负越好

# 实例
path1_score = 10 + 5 + 8 = 23  (某个错误路径)
path2_score = 12 + 6 + 7 = 25  (某个错误路径)
path3_score = 11 + 7 + 9 = 27  (正确路径) ← gold_score
...

log_Z ≈ log_sum_exp([23, 25, 27, ...]) ≈ 27.5
gold_score = 27

NLL = 27.5 - 27 = 0.5  (目标是让这个数字变小)

# 训练进度
初期：log_Z ≈ 28, gold_score = 27, NLL = 1.0
中期：log_Z ≈ 27.2, gold_score = 27, NLL = 0.2
后期：log_Z ≈ 27.01, gold_score = 27, NLL = 0.01
```

**关键洞察**：
- CRF 同时优化两个目标：让正确路径分数高，错误路径分数低
- BERT-only 只优化正确路径的分数，忽略错误路径的竞争

---

## 第六讲: 调试技巧

### 技巧 1: 用小数据集验证

```python
# 问题：训练 1000 条数据需要 30 分钟，太慢
# 解决：用 10 条数据测试

def quick_test():
    # 只加载 10 条样本
    processor = CluenerProcessor(data_dir)
    all_examples = processor.get_train_examples()[:10]
    
    dataset = CluenerBertDataset(all_examples, ...)
    loader = DataLoader(dataset, batch_size=2)
    
    trainer = Trainer(...)
    
    # 训练 1 个 epoch，应该看到 loss 持续下降
    trainer.train(num_epochs=1)
    
    # 观察指标
    # - loss 应该从 ~4.5 下降到 < 1
    # - F1 应该从 ~20% 上升
    # - 没有崩溃即可

if __name__ == "__main__":
    quick_test()  # 应该 < 1 分钟完成
```

### 技巧 2: 日志记录的层次

```python
# ❌ 不好：日志太少
print(f"Loss: {loss}")

# ✅ 好：丰富的日志供调试

import logging

logger = logging.getLogger(__name__)

# 日志级别
logging.DEBUG     # 开发时最详细（变量值、中间步骤）
logging.INFO      # 信息性（epoch 进度、F1 值）
logging.WARNING   # 警告（发现异常但继续）
logging.ERROR     # 错误（无法继续）

# 实际使用
logger.debug(f"Batch shape: {batch['input_ids'].shape}")  # 太详细
logger.info(f"Epoch {epoch}/{num_epochs}, Loss: {loss:.4f}, F1: {f1:.4f}")  # 适量
logger.warning(f"F1 dropped from {prev_f1} to {curr_f1}, overfitting?")  # 有问题
logger.error(f"Data loading failed: {e}")  # 严重问题
```

### 技巧 3: 梯度检查

```python
def check_gradients(model, batch):
    """检查是否有 NaN/Inf 梯度"""
    
    optimizer = torch.optim.Adam(model.parameters())
    
    # 计算一个 batch 的损失
    loss = model(**batch)
    loss.backward()
    
    # 检查所有梯度
    nan_params = []
    large_grads = []
    zero_grads = []
    
    for name, param in model.named_parameters():
        if param.grad is None:
            logger.warning(f"No gradient for {name}")
            continue
        
        # 检查 NaN
        if torch.isnan(param.grad).any():
            nan_params.append(name)
        
        # 检查超大梯度（可能导致爆炸）
        grad_norm = param.grad.norm().item()
        if grad_norm > 10.0:
            large_grads.append((name, grad_norm))
        
        # 检查全 0 梯度（可能导致不学习）
        if (param.grad == 0).all():
            zero_grads.append(name)
    
    if nan_params:
        logger.error(f"NaN gradients in: {nan_params}")
        return False
    
    if large_grads:
        logger.warning(f"Large gradients: {large_grads}")
    
    if zero_grads:
        logger.warning(f"Zero gradients: {zero_grads}")
    
    return True

# 使用
if not check_gradients(model, batch):
    print("Stop training, gradient check failed")
else:
    print("Gradients look good")
```

### 技巧 4: 可视化训练曲线

```python
import matplotlib.pyplot as plt
import json

# 从日志文件读取训练曲线
def plot_training_curves(log_file):
    losses = []
    train_f1s = []
    dev_f1s = []
    epochs = []
    
    with open(log_file) as f:
        for line in f:
            if "Epoch" in line and "Loss" in line:
                # 解析 log 行
                # 例如: [Epoch 1] train_loss=0.5234 dev_f1=0.7856
                data = extract_data(line)
                epochs.append(data['epoch'])
                losses.append(data['loss'])
                train_f1s.append(data.get('train_f1', 0))
                dev_f1s.append(data.get('dev_f1', 0))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 图 1: Loss 曲线
    ax1.plot(epochs, losses, 'o-', label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    
    # 图 2: F1 曲线
    ax2.plot(epochs, train_f1s, 'o-', label='Train F1')
    ax2.plot(epochs, dev_f1s, 's-', label='Dev F1')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Training Progress')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png')

# 运行
plot_training_curves('outputs/train.log')
```

---

## 第七讲: 常见陷阱与解决

### 陷阱 1: 学习率设置不当

**症状**：loss = NaN

**根本原因**：学习率过大，导致权重更新发散

**诊断**：
```python
# 在 train.py 中添加
print(f"BERT LR: {optimizer.param_groups[0]['lr']:.2e}")
print(f"Head LR: {optimizer.param_groups[1]['lr']:.2e}")

# 应该看到：
# BERT LR: 5.00e-05
# Head LR: 1.00e-03
```

**解决**：
```json
{
  "learning_rate": 1e-5,         // 减小 BERT LR
  "learning_rate_head": 5e-4     // 减小 Head LR
}
```

### 陷阱 2: Subword 对齐错误

**症状**：loss 不下降，F1 随机波动

**根本原因**：标签对齐不正确，CRF 收到错乱的标签

**诊断**：
```python
# 在 dataset.py 中添加检查
def __getitem__(self, idx):
    ex = self.examples[idx]
    ...
    
    # 验证对齐
    valid_count = sum(self.valid_mask)
    expected_count = len(ex['labels'])
    
    if valid_count != expected_count:
        raise ValueError(
            f"Alignment error: {valid_count} valid positions "
            f"but {expected_count} original labels"
        )
    
    return {...}
```

**解决**：检查 `word_ids()` 的逻辑

### 陷阱 3: 内存溢出（OOM）

**症状**：CUDA out of memory 错误

**原因**：Batch Size 或 Max Length 太大

**诊断**：
```python
# 在加载数据前检查显存
import torch
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# 估算需要的显存
# BERT: 110M params * 4 bytes = 440MB（参数）
#       + activation = 2-3GB（前向传播）
# 梯度缓存 = 3GB
# 总计 ~6GB
```

**解决**：
```json
{
  "train_batch_size": 16,         // 从 32 减小
  "gradient_accumulation_steps": 2 // 从 1 增加，保持有效 batch size 一致
}
```

---

## 第八讲: 生产部署考虑

### 考虑 1: 模型版本管理

```python
# 每个 checkpoint 应该记录元数据
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch,
    "dev_f1": dev_f1,
    "metadata": {
        "model_name": "bert-base-chinese",
        "num_labels": 33,
        "config_file": "configs/bert_crf.json",
        "training_date": "2026-03-05",
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
}

torch.save(checkpoint, f"outputs/best.pt")

# 加载时验证兼容性
def load_checkpoint(path):
    ckpt = torch.load(path, map_location='cpu')
    
    # 验证版本
    if ckpt['metadata']['pytorch_version'] != torch.__version__:
        logger.warning(f"PyTorch version mismatch: "
                      f"{ckpt['metadata']} vs {torch.__version__}")
    
    return ckpt
```

### 考虑 2: 推理优化

```python
# 方式 1: 批处理推理（比单条快 10 倍）
def batch_predict(texts, batch_size=32):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # 一起处理 batch，利用 GPU 并行性
        pred = model.predict_batch(batch)
        results.extend(pred)
    return results

# 方式 2: 模型量化
quantized_model = torch.quantization.quantize_dynamic(
    model,
    qconfig_spec={torch.nn.Linear},
    dtype=torch.qint8
)
# 模型大小减半，速度提升 2-4 倍，精度损失 < 1%

# 方式 3: ONNX 导出（跨框架兼容）
import torch.onnx

torch.onnx.export(
    model,
    (input_ids, attention_mask, valid_mask),
    "model.onnx",
    opset_version=14,
)
```

### 考虑 3: 监控和告警

```python
class ModelMonitor:
    def __init__(self, alert_threshold=0.05):
        self.alert_threshold = alert_threshold
        self.baseline_metrics = None
    
    def evaluate(self, model, data_loader):
        f1 = evaluate_f1(model, data_loader)
        
        if self.baseline_metrics is None:
            self.baseline_metrics = f1
            return f1
        
        # 检查性能是否下降
        drop = self.baseline_metrics - f1
        if drop > self.alert_threshold:
            logger.error(f"Performance drop detected: {drop:.4f}")
            self.send_alert(f"F1 dropped from {self.baseline_metrics} to {f1}")
        
        return f1
    
    def send_alert(self, message):
        # 发送到监控系统（邮件、Slack 等）
        pass
```

---

## 最后的话

### 学完这个项目后，你应该可以：

1. **理解 NER 问题**
   - 什么是 NER
   - 为什么需要 CRF
   - 如何评估 NER 模型

2. **理解代码架构**
   - 5 层架构如何工作
   - 为什么这样分层
   - 如何扩展或修改

3. **调试问题**
   - 如何诊断训练问题
   - 如何优化超参数
   - 如何处理常见陷阱

4. **部署模型**
   - 如何保存和加载模型
   - 如何做推理优化
   - 如何监控模型性能

### 推荐学习路径

**第一周**：
- [ ] 运行 `python scripts/train.py`（完整流程）
- [ ] 修改 batch size（理解显存权衡）
- [ ] 观察日志（理解训练动态）

**第二周**：
- [ ] 调整学习率（理解优化器）
- [ ] 添加 early stopping（理解过拟合）
- [ ] 试试混合精度（理解优化技巧）

**第三周**：
- [ ] 实现新的 loss function（理解 CRF）
- [ ] 试试不同的 tokenizer（理解数据处理）
- [ ] 做推理优化（理解部署）

**第四周**：
- [ ] 实现知识蒸馏（理解模型压缩）
- [ ] 尝试多任务学习（理解深层设计）
- [ ] 写论文或博客总结（梳理理解）

### 保持好奇心

最后的话：这个项目涉及的每个细节都有深层的理由。当你遇到不理解的地方时，问自己：

- **为什么** 要这样做？
- **有没有** 更简单的做法？
- **权衡是什么**？（性能 vs 复杂度，速度 vs 精度）
- **如果改变** 这个参数会怎样？

通过这样的思考，你会逐渐从**知道怎么做**升级到**理解为什么这样做**，再升级到**能创新地做**。

祝你学习愉快！ 🚀

---

**文档作者**: AI 导师  
**最后更新**: 2026-03-05  
**适合对象**: 实习生、初级工程师
