# 快速参考卡片（Cheat Sheet）

> 在开发中快速查阅

---

## 🚀 常用命令

### 环境相关
```bash
# 验证环境
python verify_setup.py

# 测试数据加载
python test_data_loading.py

# 部署前检查
python final_check.py
```

### 训练相关
```bash
# 训练（默认配置）
python scripts/train.py

# 训练（自定义配置）
python scripts/train.py --config configs/bert_crf.json

# 查看实时日志
tail -f outputs/train.log

# 后台训练（不占用终端）
nohup python scripts/train.py > train.log 2>&1 &
```

### 预测相关
```bash
# 单条文本预测
python scripts/predict.py --text "浙商银行叶老桂" --model_path outputs/best.pt

# 批量预测
python scripts/predict.py --split dev --model_path outputs/best.pt

# 输出文件位置
# - outputs/dev_prediction.jsonl (调试用)
# - outputs/dev_submit.jsonl (提交用)
```

---

## 📝 配置参数说明

### 关键参数及推荐值

```json
{
  "model": {
    "pretrained_name": "hfl/chinese-bert-wwm-ext",  // 预训练模型
    "dropout": 0.1,                                  // Dropout 比例 (0.1-0.3)
    "seed": 42                                       // 随机种子
  },
  "data": {
    "max_length": 256,                              // 最大序列长度
    "pad_label": "O"                                // 填充标签
  },
  "train": {
    "num_epochs": 5,                                // 训练轮数 (5-20)
    "train_batch_size": 32,                         // 训练 batch (8-64)
    "eval_batch_size": 32,                          // 验证 batch (32-128)
    
    "learning_rate": 5e-5,                          // BERT LR (1e-5 到 5e-5)
    "learning_rate_head": 1e-3,                     // Head LR (1e-4 到 1e-2)
    "weight_decay": 0.01,                           // L2 正则 (0.01-0.1)
    
    "warmup_ratio": 0.1,                            // Warmup 比例 (0.05-0.2)
    "max_grad_norm": 1.0,                           // 梯度裁剪 (0.5-2.0)
    "gradient_accumulation_steps": 1,               // 梯度累积 (1-4)
    
    "log_every_steps": 50,                          // 日志频率
    "eval_every_steps": 0                           // 评估频率 (0=每个 epoch)
  }
}
```

### 参数调优指南

| 问题 | 症状 | 调整 |
|-----|------|------|
| 过拟合 | train F1 >> dev F1 | ↑dropout, ↑weight_decay, ↓num_epochs |
| 欠拟合 | 所有 F1 都低 | ↑num_epochs, ↓weight_decay, ↑learning_rate |
| 梯度爆炸 | loss=NaN | ↓learning_rate, ↑max_grad_norm |
| 训练慢 | 每 epoch > 5 分钟 | ↑batch_size, 使用 mixed precision |
| 显存不足 | CUDA OOM | ↓batch_size, ↑gradient_accumulation |

---

## 🔍 代码导航

### 核心类和函数

```python
# 📊 数据处理
from src.data.processor import CluenerProcessor          # JSON → examples
from src.data.dataset import CluenerBertDataset          # examples → tensors
from src.data.collate import DataCollatorForBertCrf      # batch padding

# 🏗️ 模型
from src.models.model import BertCrfForNer              # 主模型
from src.models.crf import CRF                          # CRF 层

# 🎓 训练
from src.training.trainer import Trainer                # 训练逻辑
from src.training.lr_scheduler import build_scheduler   # LR 调度

# 🛠️ 工具
from src.utils.logger import init_logger                # 日志
from src.utils.seed import set_seed                     # 随机种子
from src.utils.tagging import bioes_to_spans            # span 提取
```

### 常用模式

#### 加载配置
```python
from src.config.config_parser import load_config

cfg = load_config("configs/bert_crf.json")
# cfg is a dict with all configuration
```

#### 加载数据
```python
from src.data.processor import CluenerProcessor
from src.data.dataset import CluenerBertDataset
from torch.utils.data import DataLoader

processor = CluenerProcessor(data_dir)
examples = processor.get_train_examples()

dataset = CluenerBertDataset(examples, pretrained_name, label_map)
loader = DataLoader(dataset, batch_size=32, collate_fn=collator)
```

#### 创建模型
```python
from src.models.model import BertCrfForNer

model = BertCrfForNer(
    pretrained_name="hfl/chinese-bert-wwm-ext",
    num_tags=num_labels,
    start_tag_id=label_map.START_ID,
    stop_tag_id=label_map.STOP_ID,
    dropout=0.1
)
```

#### 创建优化器（分层学习率）
```python
from torch.optim import AdamW

bert_params = [p for n, p in model.named_parameters() if "bert" in n]
head_params = [p for n, p in model.named_parameters() if "bert" not in n]

optimizer = AdamW([
    {"params": bert_params, "lr": 5e-5, "weight_decay": 0.01},
    {"params": head_params, "lr": 1e-3, "weight_decay": 0.01},
])
```

#### 训练循环
```python
for epoch in range(num_epochs):
    # Train
    for batch in train_loader:
        loss = model(**batch)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    # Evaluate
    dev_f1, report = trainer.evaluate()
    print(f"Epoch {epoch}, Dev F1: {dev_f1:.4f}")
```

#### 预测
```python
model.eval()
with torch.no_grad():
    pred_paths = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        valid_mask=batch["valid_mask"],
        label_ids=None  # None 时进行预测而非计算 loss
    )
    # pred_paths 是预测的标签 ID 序列
```

---

## 🐛 调试快速指南

### 检查清单

```python
# 1. 数据加载是否正确？
batch = next(iter(train_loader))
assert batch["input_ids"].shape == (32, 256)
print(batch.keys())  # 应该有 input_ids, attention_mask 等

# 2. 模型前向传播是否正确？
model = BertCrfForNer(...)
output = model(**batch)
assert isinstance(output, float)  # 训练模式应返回 loss（标量）

# 3. 梯度是否存在？
loss = model(**batch)
loss.backward()
for name, param in model.named_parameters():
    assert param.grad is not None, f"No gradient: {name}"

# 4. 学习率是否正确设置？
print(f"LR 0: {optimizer.param_groups[0]['lr']}")  # BERT
print(f"LR 1: {optimizer.param_groups[1]['lr']}")  # Head

# 5. 设备是否正确？
batch = {k: v.to(device) for k, v in batch.items()}
model = model.to(device)
```

### 常见错误与速查

| 错误 | 原因 | 修复 |
|------|------|------|
| `RuntimeError: Expected all tensors to be on the same device` | 数据和模型 device 不匹配 | `batch = move_to_device(batch, device)` |
| `IndexError: index out of range` | label_ids 包含超出范围的标签 ID | 检查 label_map 是否正确 |
| `KeyError: 'bert'` | 优化器分组查找 "bert" 参数失败 | 检查模型中是否存在 `self.bert`属性 |
| `AssertionError: Dimensions don't match` | Subword 对齐失败 | 检查 `valid_mask` 的计算 |
| `cuda:0: out of memory` | 显存不足 | ↓batch_size 或使用 mixed precision |

---

## 📊 监控指标

### 训练过程中应该看到的

```
[Epoch 1] 
  Train Loss: 4.2103 → 2.5678 → 1.2345  (应该持续下降)
  Dev F1: 0.4521                         (初期较低)

[Epoch 2]
  Train Loss: 1.1234 → 0.8901 → 0.6543  (继续下降，但速度变慢)
  Dev F1: 0.6234                         (逐渐上升)

[Epoch 3]
  Train Loss: 0.6234 → 0.5891 → 0.5432  (逐渐收敛)
  Dev F1: 0.7456

...

[Epoch 10]
  Train Loss: 0.3123 → 0.3012 → 0.2998  (基本收敛)
  Dev F1: 0.7982                         (稳定在高值)
```

### 指标异常检查

```python
# ❌ 不好的迹象
- loss 不下降或上升 → 学习率太高或数据有问题
- loss 为 NaN → 梯度爆炸
- F1 随机波动 0.x 到 0.y → 标签对齐问题
- train F1 高但 dev F1 低 → 严重过拟合

# ✅ 好的迹象
- loss 平稳持续下降
- train F1 和 dev F1 都上升，差距 < 10%
- dev F1 在第 3-5 epoch 达到最高，之后稳定
- 没有梯度警告或错误
```

---

## 📁 文件变更快速参考

### 添加新实体类型

1. **更新数据集**（如果数据集更新）
   - 确保 JSON 包含新类型

2. **更新 label_map**
   ```json
   // configs/bert_crf.json 中的 labels.label2id
   {
     "O": 0,
     "B-new_type": 31,
     "I-new_type": 32
   }
   ```

3. **重新运行数据处理**
   - 新标签会自动处理

4. **训练**
   ```bash
   python scripts/train.py
   ```

### 更换预训练模型

1. **修改配置**
   ```json
   {
     "model": {
       "pretrained_name": "distilbert-base-chinese"
     }
   }
   ```

2. **重新训练**
   - 首次会下载新模型（~300MB）

### 调整超参数

编辑 `configs/bert_crf.json`，然后运行 `python scripts/train.py`

---

## 🎯 实验追踪

### 推荐的实验记录方式

```python
# experiment_log.csv
date,learning_rate,batch_size,dropout,f1_score,notes
2026-03-05,5e-5,32,0.1,0.7982,baseline
2026-03-05,1e-5,32,0.1,0.7945,lr_too_low
2026-03-05,5e-5,16,0.1,0.8012,smaller_batch
2026-03-05,5e-5,32,0.2,0.7956,higher_dropout
```

### 快速实验脚本

```python
import json
from pathlib import Path

def run_experiment(exp_name, config_overrides):
    # 加载默认配置
    with open('configs/bert_crf.json') as f:
        cfg = json.load(f)
    
    # 应用实验配置
    for key, value in config_overrides.items():
        # 支持嵌套 key，如 "train.learning_rate"
        keys = key.split('.')
        d = cfg
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value
    
    # 保存实验配置
    exp_dir = Path(f"experiments/{exp_name}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    with open(exp_dir / "config.json", 'w') as f:
        json.dump(cfg, f, indent=2)
    
    # 修改输出目录
    cfg['output_dir'] = str(exp_dir / 'outputs')
    
    # 运行训练
    # python scripts/train.py 会读新配置
    
    return exp_dir

# 运行实验
for lr in [1e-5, 5e-5, 1e-4]:
    for bs in [16, 32, 64]:
        run_experiment(
            f"lr_{lr}_bs_{bs}",
            {
                "train.learning_rate": lr,
                "train.train_batch_size": bs
            }
        )
```

---

## 🔗 相关资源

### 官方文档
- BERT: https://huggingface.co/docs/transformers/model_doc/bert
- PyTorch: https://pytorch.org/docs/stable/index.html
- seqeval: https://github.com/chakki-works/seqeval

### 论文
- BERT: https://arxiv.org/abs/1810.04805
- CRF: https://arxiv.org/abs/1106.0155
- Attention: https://arxiv.org/abs/1706.03762

### 数据集
- CLUENER: https://github.com/CLUEbenchmark/CLUENER2020

---

## 💡 Tips & Tricks

### 训练加速
```bash
# 使用混合精度（需要 GPU）
# 修改 train.py，添加 autocast()
```

### 模型压缩
```python
# 使用 DistilBERT（更快，更小）
# 在配置中改为 "distilbert-base-chinese"
```

### 参数共享
```python
# 某些情况下可以在不同任务间共享 BERT 层
```

### 在线学习
```python
# 持续迭代模型
# 1. 预测新数据
# 2. 人工修正
# 3. 加入训练集
# 4. 微调模型
```

---

**最后更新**: 2026-03-05  
**打印建议**: A4，黑白打印
