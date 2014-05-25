#! /bin/bash
rm data/*.vw.cache
vw data/train.vw -c -k --passes 45 -l 0.85 -f data/model.vw --loss_function quantile --quantile_tau 0.6
vw data/test.vw -t -i data/model.vw -p data/shop.preds.txt
echo "id,repeatProbability" > data/predictions.csv
cat data/shop.preds.txt | awk '{print $2","$1}' >> data/predictions.csv
cat data/train.vw | awk '{print $2","$1}' | sed 's/.//' > data/labels.csv
