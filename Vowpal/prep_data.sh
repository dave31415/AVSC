echo "Combining item and category data"
csvjoin -c item_category,category_id data/items.csv data/categories.csv > data/items_categories.csv
export HEADERS=`head -n1 data/items_categories.csv`,item_popularity,item_avg_cost,category_avg_cost,item_avg_cost_div_by_category_avg_cost
cat data/items_categories.csv | awk -F ',' 'NR > 1 {print $0","$5/$9","$6/$5","$10/$9","($6/$5)/($10/$9)}' > data/items_categories.tmp; mv data/items_categories.tmp data/items_categories.csv
echo $HEADERS | cat - data/items_categories.csv > temp && mv temp data/items_categories.csv

echo "Joining item/category data into the aggregated training data"
csvjoin -d "," -c offer_item_id,item_id data/train_april.csv data/items_categories.csv > temp; mv temp data/train_april.csv
csvjoin -d "," -c offer_item_id,item_id data/train_march.csv data/items_categories.csv > temp; mv temp data/train_march.csv
csvjoin -d "," -c offer_item_id,item_id data/train.csv data/items_categories.csv > temp; mv temp data/train.csv
csvjoin -d "," -c offer_item_id,item_id data/test.csv data/items_categories.csv > temp; mv temp data/test.csv

echo "Converting files to VW format"
python csv_to_vw.py -l 1 -i item_id,offer_item_id,offer_date data/train_april.csv data/train_april.vw
python csv_to_vw.py -l 1 -i item_id,offer_item_id,offer_date data/train_march.csv data/train_march.vw
python csv_to_vw.py -l 1 -i item_id,offer_item_id,offer_date data/train.csv data/train.vw
python csv_to_vw.py -l 1 -i item_id,offer_item_id,offer_date data/test.csv data/test.vw
