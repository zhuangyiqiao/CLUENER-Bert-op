1.添加检查脚本，检查special token / padding 没有污染训练和评估
cd /root/autodl-tmp/cluener
python script/train_ner_torch2.py --model_name ./bert-wwm-ext --epochs 1 --eval_steps 999999
Device: cuda

===== DEBUG SAMPLE =====
pos=  0 token=[CLS]      mask=1 label=-100
pos= 51 token=[SEP]      mask=1 label=-100
pos= 52 token=[PAD]      mask=0 label=-100
pos= 53 token=[PAD]      mask=0 label=-100
pos= 54 token=[PAD]      mask=0 label=-100
pos= 55 token=[PAD]      mask=0 label=-100
pos= 56 token=[PAD]      mask=0 label=-100
pos= 57 token=[PAD]      mask=0 label=-100
pos= 58 token=[PAD]      mask=0 label=-100
pos= 59 token=[PAD]      mask=0 label=-100
pos= 60 token=[PAD]      mask=0 label=-100
pos= 61 token=[PAD]      mask=0 label=-100
pos= 62 token=[PAD]      mask=0 label=-100
pos= 63 token=[PAD]      mask=0 label=-100
pos= 64 token=[PAD]      mask=0 label=-100
pos= 65 token=[PAD]      mask=0 label=-100
pos= 66 token=[PAD]      mask=0 label=-100
pos= 67 token=[PAD]      mask=0 label=-100
pos= 68 token=[PAD]      mask=0 label=-100
pos= 69 token=[PAD]      mask=0 label=-100
pos= 70 token=[PAD]      mask=0 label=-100
pos= 71 token=[PAD]      mask=0 label=-100
pos= 72 token=[PAD]      mask=0 label=-100
pos= 73 token=[PAD]      mask=0 label=-100
pos= 74 token=[PAD]      mask=0 label=-100
pos= 75 token=[PAD]      mask=0 label=-100
pos= 76 token=[PAD]      mask=0 label=-100
pos= 77 token=[PAD]      mask=0 label=-100
pos= 78 token=[PAD]      mask=0 label=-100
pos= 79 token=[PAD]      mask=0 label=-100
pos= 80 token=[PAD]      mask=0 label=-100
pos= 81 token=[PAD]      mask=0 label=-100
pos= 82 token=[PAD]      mask=0 label=-100
pos= 83 token=[PAD]      mask=0 label=-100
pos= 84 token=[PAD]      mask=0 label=-100
pos= 85 token=[PAD]      mask=0 label=-100
pos= 86 token=[PAD]      mask=0 label=-100
pos= 87 token=[PAD]      mask=0 label=-100
pos= 88 token=[PAD]      mask=0 label=-100
pos= 89 token=[PAD]      mask=0 label=-100
pos= 90 token=[PAD]      mask=0 label=-100
pos= 91 token=[PAD]      mask=0 label=-100
pos= 92 token=[PAD]      mask=0 label=-100
pos= 93 token=[PAD]      mask=0 label=-100
pos= 94 token=[PAD]      mask=0 label=-100
pos= 95 token=[PAD]      mask=0 label=-100
pos= 96 token=[PAD]      mask=0 label=-100
pos= 97 token=[PAD]      mask=0 label=-100
pos= 98 token=[PAD]      mask=0 label=-100
pos= 99 token=[PAD]      mask=0 label=-100
pos=100 token=[PAD]      mask=0 label=-100
pos=101 token=[PAD]      mask=0 label=-100
pos=102 token=[PAD]      mask=0 label=-100
pos=103 token=[PAD]      mask=0 label=-100
pos=104 token=[PAD]      mask=0 label=-100
pos=105 token=[PAD]      mask=0 label=-100
pos=106 token=[PAD]      mask=0 label=-100
pos=107 token=[PAD]      mask=0 label=-100
pos=108 token=[PAD]      mask=0 label=-100
pos=109 token=[PAD]      mask=0 label=-100
pos=110 token=[PAD]      mask=0 label=-100
pos=111 token=[PAD]      mask=0 label=-100
pos=112 token=[PAD]      mask=0 label=-100
pos=113 token=[PAD]      mask=0 label=-100
pos=114 token=[PAD]      mask=0 label=-100
pos=115 token=[PAD]      mask=0 label=-100
pos=116 token=[PAD]      mask=0 label=-100
pos=117 token=[PAD]      mask=0 label=-100
pos=118 token=[PAD]      mask=0 label=-100
pos=119 token=[PAD]      mask=0 label=-100
pos=120 token=[PAD]      mask=0 label=-100
pos=121 token=[PAD]      mask=0 label=-100
pos=122 token=[PAD]      mask=0 label=-100
pos=123 token=[PAD]      mask=0 label=-100
pos=124 token=[PAD]      mask=0 label=-100
pos=125 token=[PAD]      mask=0 label=-100
pos=126 token=[PAD]      mask=0 label=-100
pos=127 token=[PAD]      mask=0 label=-100
========================

[INFO] cache_dir=cache scheme=None num_labels=21
Loading weights: 100%|█| 197/197 [00:00<00:00, 8597.75it/s, Materializing param=bert.encoder.layer.11.output.d
BertForTokenClassification LOAD REPORT from: ./bert-wwm-ext
Key                                        | Status     | 
-------------------------------------------+------------+-
cls.predictions.transform.LayerNorm.weight | UNEXPECTED | 
bert.pooler.dense.weight                   | UNEXPECTED | 
bert.pooler.dense.bias                     | UNEXPECTED | 
cls.predictions.bias                       | UNEXPECTED | 
cls.seq_relationship.bias                  | UNEXPECTED | 
cls.predictions.decoder.weight             | UNEXPECTED | 
cls.predictions.transform.dense.weight     | UNEXPECTED | 
cls.predictions.transform.LayerNorm.bias   | UNEXPECTED | 
cls.seq_relationship.weight                | UNEXPECTED | 
cls.predictions.transform.dense.bias       | UNEXPECTED | 
classifier.weight                          | MISSING    | 
classifier.bias                            | MISSING    | 

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical 
arch.                                                                                                         - MISSING       :those params were newly initialized because missing from the checkpoint. Consider training on
 your downstream task.                                                                                        Logging to: outputs/bert_run/train_log.jsonl
epoch 1 step 20/336 loss 1.1821 time 0.06s
epoch 1 step 40/336 loss 0.6340 time 0.06s
epoch 1 step 60/336 loss 0.4280 time 0.08s
epoch 1 step 80/336 loss 0.3282 time 0.08s
epoch 1 step 100/336 loss 0.3385 time 0.08s
epoch 1 step 120/336 loss 0.2881 time 0.08s
epoch 1 step 140/336 loss 0.2262 time 0.06s
epoch 1 step 160/336 loss 0.2592 time 0.06s
epoch 1 step 180/336 loss 0.2530 time 0.07s
epoch 1 step 200/336 loss 0.2049 time 0.07s
epoch 1 step 220/336 loss 0.2438 time 0.07s
epoch 1 step 240/336 loss 0.1814 time 0.07s
epoch 1 step 260/336 loss 0.2215 time 0.07s
epoch 1 step 280/336 loss 0.2102 time 0.06s
epoch 1 step 300/336 loss 0.2012 time 0.06s
epoch 1 step 320/336 loss 0.1847 time 0.06s
[EPOCH END] epoch 1 dev_loss 0.1918 token_acc 0.9436 ent_f1 0.7497
Writing model shards: 100%|█████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.64it/s]
[DONE] final model saved to: outputs/bert_run/final

2.把 scheme=None 变成“确定的 BIO 或 BIOS” ，我现在的 spans_from_tags() 会根据预测里有没有 S- 来“自动切换 BIO/BIOS”。训练早期这种切换会让 F1 不稳定。
检查label2id里面是有s的，也就是说用的确实是bios
scheme = meta.get("scheme", "unknown")改成scheme = meta.get("scheme") or "BIOS"   

3.train3改动：evaluate()与论文格式对齐，输出log对齐。
观察哪一类别的最难识别
训练结果跟之前的指数差不多，3伦之后难以收敛。
依旧是baseline
最好的点是：
step=1000 / epoch=3
dev_loss ≈ 0.197
micro_f1 ≈ 0.7607
macro_f1 ≈ 0.7582（你现在用 macro 做 best，很对）
然后到 epoch 4/5：
dev_loss 升到 0.22~0.24
macro_f1 回落到 0.75 左右
轻微过拟合：训练 loss 继续掉（0.13→0.04→0.02），但 dev 变差。


4.误差，区别
address & scene 明显拖后腿
从 per_type 看（以 step=1000 为例）：
address F1=0.604（最低）
scene F1=0.691（第二低）
其它大部分 0.75~0.79，name 高达 0.874
address：边界长、形式多变（“北京市海淀区…XX路XX号…”），容易被切碎、也容易预测出多余片段 → FP 高
scene：语义像地点/机构/活动场景的混合体，定义边界更主观 → FN/FP 都容易发生
你 step=200 那一轮 address 的 fp=292 很夸张，后面虽然好一些，但依旧是最差类别。
明天需要把专门把 address/scene 的错误拆开看
对 address / scene 做一个错误模式采样 看一下address 到底是“边界错”多，还是“类型错”多？
如果“边界错”多：说明 tokenizer/标注对齐/长实体切分是主要问题
如果“类型错”多：说明 scene/org/location 之间语义混淆，需要特征或后处理

明日：1.检查2.v4 （crf）