This code was forked from Triskellion

to use:
first run the the data reducer on the transaction file
(the reducer is commented out in the file to run faster) 
assuming the data is in a directory called data

-----------------------------------------------------------------------------------------------
python gen_vw_features.py  
#2m12.263s

vw data/train.vw -c -k --passes 40 -l 0.85 -f data/model.vw --loss_function quantile --quantile_tau 0.6 
# 1m18.721s
vw data/test.vw -t -i data/model.vw -p data/shop.preds.txt
# 0m0.911s

python gen_submission.py
-----------------------------------------------------------------------------------------------
#Kaggle aquire valued shoppers challenge#
Code (Feature generation) for the Kaggle challenge: acquire-valued-shoppers

See https://kaggle.com for rules and data

See http://www.kaggle.com/c/acquire-valued-shoppers-challenge/ for the competition

See http://mlwave.com for tutorial and discussion.
