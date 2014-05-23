python gen_new_vw_features.py

csvjoin -c category data/items.csv data/categories.csv > data/items_categories.csv
export HEADERS=`head -n1 data/items_categories.csv`,item_popularity,item_avg_cost,category_avg_cost,item_avg_cost/category_avg_cost
cat data/items_categories.csv | awk -F ',' 'NR > 1 {print $0","$5/$9","$6/$5","$10/$9}' > data/items_categories.tmp; mv data/items_categories.tmp data/items_categories.csv
echo $HEADERS | cat - data/items_categories.csv > temp && mv temp data/items_categories.csv

python ~/Programming/phraug2/csv2vw.py data/train_april.csv data/train_april.vw
python ~/Programming/phraug2/csv2vw.py data/train_march.csv data/train_march.vw
python ~/Programming/phraug2/csv2vw.py data/train.csv data/train.vw
python ~/Programming/phraug2/csv2vw.py data/test.csv data/test.vw
