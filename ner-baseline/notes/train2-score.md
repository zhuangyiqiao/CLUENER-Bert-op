root@autodl-container-6dda4bb49e-dac8445a:~/autodl-tmp/cluener# python script/score_span_f1.py \
  --gold data/dev.json \
  --pred cluener_dev_pred.json \
  --strict_text_match
===== OVERALL (micro) =====
TP=2379 FP=821 FN=693
P=0.7434 R=0.7744 F1=0.7586
Macro-F1(10 types avg)=0.7603
[TEXT CHECK] mismatch lines: 0/1343

===== PER TYPE =====
type    P       R       F1      TP      FP      FN
address 0.5652  0.5925  0.5785  221     170     152
book    0.7321  0.7987  0.7640  123     45      31
company 0.7448  0.7566  0.7507  286     98      92
game    0.7530  0.8373  0.7929  247     81      48
government      0.7509  0.8178  0.7829  202     67      45
movie   0.8889  0.7947  0.8392  120     15      31
name    0.8518  0.8774  0.8644  408     71      57
organization    0.7390  0.7793  0.7586  286     101     81
position        0.7573  0.7783  0.7677  337     108     96
scene   0.6963  0.7129  0.7045  149     65      60
root@autodl-container-6dda4bb49e-dac8445a:~/autodl-tmp/cluener# 