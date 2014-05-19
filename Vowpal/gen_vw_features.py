# -*- coding: UTF-8 -*-

"""
Kaggle Challenge: 
"http://www.kaggle.com/c/acquire-valued-shoppers-challenge/" 
'Reduce the data and generate features' by Triskelion 
After a forum post by BreakfastPirate
Very mediocre and hacky code, single-purpose, but pretty fast
Some refactoring by Zygmunt Zając <zygmunt@fastml.com>
More refactoring done
"""

from datetime import datetime, date
from collections import defaultdict

loc_offers = "../data/offers.csv"
loc_transactions = "../data/transactions.csv"
loc_train = "../data/trainHistory.csv"
loc_test = "../data/testHistory.csv"
loc_reduced = "../data/reduced.csv" 
loc_out_train = "../data/train.vw"
loc_out_march = "../data/train_march.vw"
loc_out_april = "../data/train_april.vw"
loc_out_test = "../data/test.vw"

#training row descriptors
id_index = 0
training_offer_id_index = 2
train_repeater_index = 5
training_date_index = -1

#transaction row descriptors
id_index = 0
chain_index = 1
dept_index = 2
category_index = 3
company_index = 4
brand_index = 5
date_index = 6
productsize_index = 7
productmeasure_index = 8
purchasequantity_index = 9
purchaseamount_index = 10



#offer row descriptors
offer_category_index = 1
offer_quantity_index = 2
offer_company_index = 3
offer_value_index = 4
offer_brand_index = 5

def update_buying_history_feature(features, feature_name, row, training_row):
	features['has_bought_' + feature_name] += 1.0
	features['has_bought_' + feature_name + '_q'] += float( row[purchasequantity_index] )
	features['has_bought_' + feature_name + '_a'] += float( row[purchaseamount_index] )

	date_diff_days = diff_days(row[6],training_row[-1])
	if date_diff_days < 30:
		features['has_bought_' + feature_name + '_30'] += 1.0
		features['has_bought_' + feature_name + '_q_30'] += float( row[purchasequantity_index] )
		features['has_bought_' + feature_name + '_a_30'] += float( row[purchaseamount_index] )
	if date_diff_days < 60:
		features['has_bought_' + feature_name + '_60'] += 1.0
		features['has_bought_' + feature_name + '_q_60'] += float( row[purchasequantity_index] )
		features['has_bought_' + feature_name + '_a_60'] += float( row[purchaseamount_index] )
	if date_diff_days < 90:
		features['has_bought_' + feature_name + '_90'] += 1.0
		features['has_bought_' + feature_name + '_q_90'] += float( row[purchasequantity_index] )
		features['has_bought_' + feature_name + '_a_90'] += float( row[purchaseamount_index] )
	if date_diff_days < 180:
		features['has_bought_' + feature_name + '_180'] += 1.0
		features['has_bought_' + feature_name + '_q_180'] += float( row[purchasequantity_index] )
		features['has_bought_' + feature_name + '_a_180'] += float( row[purchaseamount_index] )

def reduce_data(loc_offers, loc_transactions, loc_reduced):
  start = datetime.now()
  #get all categories and comps on offer in a dict
  offers_cat = {}
  offers_co = {}
  for e, line in enumerate( open(loc_offers) ):
	offers_cat[ line.split(",")[1] ] = 1
	offers_co[ line.split(",")[3] ] = 1
  #open output file
  with open(loc_reduced, "wb") as outfile:
	#go through transactions file and reduce
	reduced = 0
	for e, line in enumerate( open(loc_transactions) ):
	  if e == 0:
		outfile.write( line ) #print header
	  else:
		#only write when if category in offers dict
		if line.split(",")[3] in offers_cat or line.split(",")[4] in offers_co:
		  outfile.write( line )
		  reduced += 1
	  #progress
	  if e % 5000000 == 0:
		print e, reduced, datetime.now() - start
  print e, reduced, datetime.now() - start

def diff_days(s1,s2):
	date_format = "%Y-%m-%d"
	a = datetime.strptime(s1, date_format)
	b = datetime.strptime(s2, date_format)
	delta = b - a
	return delta.days
	
def load_train_ids():
	train_ids = {}
	for e, line in enumerate( open(loc_train) ):
		if e > 0:
			row = line.strip().split(",")
			train_ids[row[id_index]] = row
	return train_ids

def load_test_ids():
	test_ids = {}
	for e, line in enumerate( open(loc_test) ):
		if e > 0:
			row = line.strip().split(",")
			test_ids[row[id_index]] = row
	return test_ids
	
def load_offers():
	offers = {}
	for e, line in enumerate( open(loc_offers) ):
		row = line.strip().split(",")
		offers[ row[id_index] ] = row
	return offers

def output_features(features, last_id, out_test, out_train, out_march, out_april):
	#negative features
	if "has_bought_company" not in features:
		features['never_bought_company'] = 1
	if "has_bought_category" not in features:
		features['never_bought_category'] = 1
	if "has_bought_brand" not in features:
		features['never_bought_brand'] = 1
	if "has_bought_brand" in features and "has_bought_category" in features and "has_bought_company" in features:
		features['has_bought_brand_company_category'] = 1
	if "has_bought_brand" in features and "has_bought_category" in features:
		features['has_bought_brand_category'] = 1
	if "has_bought_brand" in features and "has_bought_company" in features:
		features['has_bought_brand_company'] = 1
	outline = ""
	in_march = diff_days('2013-04-1', features['offer_date']) < 0
	del features['offer_date']
	test = False
	for k, v in features.items():
		if k == "label" and v == 0.5:
			#test
			outline = "1 '" + last_id + " |f" + outline
			test = True
		elif k == "label":
			outline = str(v) + " '" + last_id + " |f" + outline
		else:
			outline += " " + k+":"+str(v) 
	outline += "\n"
	if test:
		out_test.write( outline )
	else:
		out_train.write( outline )
		if in_march:
			out_march.write( outline )
		else:
			out_april.write( outline )

def generate_features(loc_train, loc_test, loc_transactions, loc_out_train, loc_out_test):
	offers = load_offers()
	train_ids = load_train_ids()
	test_ids = load_test_ids()

	with open(loc_out_train, "wb") as out_train, open(loc_out_test, "wb") as out_test, open(loc_out_march, "wb") as out_march, open(loc_out_april, "wb") as out_april:
		last_id = 0
		features = defaultdict(float)
		for e, line in enumerate( open(loc_transactions) ):
			if e > 0: #skip header
				row = line.strip().split(",")
				reading_rows_for_new_shopper = (last_id != row[id_index] and e != 1)
				if reading_rows_for_new_shopper: 
					output_features(features, last_id, out_test, out_train, out_march, out_april)
					features = defaultdict(float)
				if row[id_index] in train_ids or row[id_index] in test_ids:
					#generate label and history
					if row[id_index] in train_ids:
						training_row = train_ids[row[id_index]]
						if train_ids[row[id_index]][train_repeater_index] == "t":
							features['label'] = 1
						else:
							features['label'] = 0
					else:
						training_row = test_ids[row[id_index]]
						features['label'] = 0.5
					offer_row = offers[ training_row[training_offer_id_index] ]
					features['offer_value'] = offer_row[offer_value_index]
					features['offer_quantity'] = offer_row[offer_quantity_index]
					features['offer_date'] = training_row[training_date_index]	
					
					features['total_spend'] += float( row[purchaseamount_index] )	
					transaction_company_matches_offer = offer_row[offer_company_index] == row[company_index]
					transaction_category_matches_offer = offer_row[offer_category_index] == row[category_index]
					transaction_brand_matches_offer = offer_row[offer_brand_index] == row[brand_index]	
					if transaction_company_matches_offer:
						update_buying_history_feature(features, 'company', row, training_row)
					if transaction_category_matches_offer:
						update_buying_history_feature(features, 'category', row, training_row)
					if transaction_brand_matches_offer:
						update_buying_history_feature(features, 'brand', row, training_row)
				last_id = row[id_index]
				if e % 100000 == 0:
					print e

if __name__ == '__main__':
	#reduce_data(loc_offers, loc_transactions, loc_reduced)
	generate_features(loc_train, loc_test, loc_reduced, loc_out_train, loc_out_test)

	
	
