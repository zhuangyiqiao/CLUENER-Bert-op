#!/usr/bin/env python
import json
from pathlib import Path

cfg = json.load(open('configs/bert_crf.json'))
print('✅ Config loaded successfully')
print(f'Data dir: {cfg["data_dir"]}')
print(f'Output dir: {cfg["output_dir"]}')
print(f'Model: {cfg["model"]["pretrained_name"]}')
print(f'BERT LR: {cfg["train"]["learning_rate"]}')
print(f'Head LR: {cfg["train"]["learning_rate_head"]}')

# Check data files
data_dir = Path(cfg["data_dir"])
for fname in ['train.json', 'dev.json', 'test.json']:
    fpath = data_dir / fname
    if fpath.exists():
        print(f'✅ {fpath.relative_to(".")} exists')
    else:
        print(f'❌ {fpath.relative_to(".")} NOT FOUND')

print("\n📋 Config Summary:")
for k in ['num_epochs', 'train_batch_size', 'eval_batch_size', 'warmup_ratio', 'max_grad_norm']:
    print(f'  {k}: {cfg["train"].get(k, "N/A")}')
