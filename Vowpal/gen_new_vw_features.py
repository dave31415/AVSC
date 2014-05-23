# -*- coding: UTF-8 -*-

"""
Kaggle Challenge: 
"http://www.kaggle.com/c/acquire-valued-shoppers-challenge/" 
'Reduce the data and generate features' by Triskelion 
After a forum post by BreakfastPirate
Very mediocre and hacky code, single-purpose, but pretty fast
Some refactoring by Zygmunt Zaj��c <zygmunt@fastml.com>
More refactoring done
"""

from datetime import datetime, date
from data_reducer import *
from offer_row import *
from test_row import *
from train_row import *
from transaction_row import *

loc_offers = "../data/offers.csv"
loc_transactions = "../data/transactions.csv"
loc_train = "../data/trainHistory.csv"
loc_test = "../data/testHistory.csv"
loc_reduced = "../data/reduced.csv" 
loc_out_train = "../data/train.vw"
loc_out_march = "../data/train_march.vw"
loc_out_april = "../data/train_april.vw"
loc_out_test = "../data/test.vw"

id_index = 0

#training row descriptors
training_chain_index = 1
training_offer_id_index = 2
training_market_index = 3
training_repeater_index = 5
training_date_index = -1

#transaction row descriptors
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
	add_to_dict(features, 'has_bought_' + feature_name, 1.0)
	add_to_dict(features, 'has_bought_' + feature_name + '_q', float( row.purchasequantity ))
	add_to_dict(features, 'has_bought_' + feature_name + '_a', float( row.purchaseamount ))

	date_diff_days = diff_days(row.date,training_row.date)
	add_time_limited_history_features(30, row, features, date_diff_days, feature_name)
	add_time_limited_history_features(60, row, features, date_diff_days, feature_name)
	add_time_limited_history_features(90, row, features, date_diff_days, feature_name)
	add_time_limited_history_features(180, row, features, date_diff_days, feature_name)
	
def add_time_limited_history_features(num_days, row, features, date_diff_days, feature_name):
	if date_diff_days < num_days:
		add_to_dict(features, 'has_bought_' + feature_name + '_' + str(num_days), 1.0)
		add_to_dict(features, 'has_bought_' + feature_name + '_q_' + str(num_days), float( row.purchasequantity ))
		add_to_dict(features, 'has_bought_' + feature_name + '_a_' + str(num_days), float( row.purchaseamount ))

def add_to_dict(dict, key, amount):
	if key in dict:
		dict[key] += amount
	else:
		dict[key] = 0.0
		dict[key] += amount

def diff_days(s1,s2):
	date_format = "%Y-%m-%d"
	a = datetime.strptime(s1, date_format)
	b = datetime.strptime(s2, date_format)
	delta = b - a
	return delta.days
	
def load_training_rows():
	training = {}
	for e, line in enumerate( open(loc_train) ):
		if e > 0:
			row = TrainRow(line.strip())
			training[row.id] = row
	return training

def load_test_rows():
	tests = {}
	for e, line in enumerate( open(loc_test) ):
		if e > 0:
			row = TestRow(line.strip())
			tests[row.id] = row
	return tests
	
def load_offer_rows():
	offers = {}
	for e, line in enumerate( open(loc_offers) ):
		if e > 0:
			row = OfferRow(line.strip())
			offers[row.id] = row
	return offers

def output_features(features, last_id, out_test, out_train, out_march, out_april):
	#negative features
	if "has_bought_company" not in features:
		features['never_bought_company'] = 1
	if "has_bought_category" not in features:
		features['never_bought_category'] = 1
	if "has_bought_brand" not in features:
		features['never_bought_brand'] = 1
	if "has_bought_dept" not in features:
		features['never_bought_dept'] = 1
	if "has_bought_brand" in features and "has_bought_category" in features and "has_bought_company" in features:
		features['has_bought_brand_company_category'] = 1
	if "has_bought_brand" in features and "has_bought_category" in features:
		features['has_bought_brand_category'] = 1
	if "has_bought_brand" in features and "has_bought_company" in features:
		features['has_bought_brand_company'] = 1
	if "has_bought_brand" in features and "has_bought_dept" in features:
		features['has_bought_brand_dept'] = 1
	if "has_bought_company" in features and "has_bought_dept" in features:
		features['has_bought_company_dept'] = 1
	outline = ""
	in_march = diff_days('2013-04-1', features['offer_date']) < 0
	del features['offer_date']
	test = False
	for k, v in features.items():
		if k == "label" and v == 0.5:
			#test
			outline = "1 '" + str( last_id ) + " |f" + outline
			test = True
		elif k == "label":
			outline = str(v) + " '" + str( last_id ) + " |f" + outline
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

def reset_features(row, training_row, train_ids, offers):
	features = dict()
	if row.id in train_ids:
		if training_row.repeater == "t":
			features['label'] = 1
		else:
			features['label'] = 0
	else:
		features['label'] = 0.5
	this_offer = offers[ training_row.offer]
	features['offer_value'] = this_offer.value
	features['offer_quantity'] = this_offer.quantity
	features['offer_date'] = training_row.date	
	features['offer_chain'] = training_row.chain
	features['offer_market'] = training_row.market
	features['total_spend'] = 0.0
	return features

def generate_features(loc_train, loc_test, loc_transactions, loc_out_train, loc_out_test):
	offers = load_offer_rows()
	train_ids = load_training_rows()
	test_ids = load_test_rows()

	with open(loc_out_train, "wb") as out_train, open(loc_out_test, "wb") as out_test, open(loc_out_march, "wb") as out_march, open(loc_out_april, "wb") as out_april:
		last_id = 0
		features = dict()
		features_dept = list()
		for e, line in enumerate( open(loc_transactions) ):
			if e > 0: #skip header
				row = TransactionRow(line.strip())
				new_shopper = last_id != row.id
				if new_shopper and e != 1: 
					output_features(features, last_id, out_test, out_train, out_march, out_april)
				if new_shopper:
					if row.id in train_ids:
						training_row = train_ids[row.id]
					else:
						training_row = test_ids[row.id]
					features = reset_features(row, training_row, train_ids, offers)
					features_dept = list()
				if row.id in train_ids or row.id in test_ids:
					if row.id in train_ids:
						training_row = train_ids[row.id]
					else:
						training_row = test_ids[row.id]
					#generate label and history
					offer = offers[ training_row.offer]
					features['total_spend'] += float( row.purchaseamount )	
					if features['offer_chain'] == row.chain:
						features['has_bought_at_chain_before'] = 1
					if offer.company == row.company:
						update_buying_history_feature(features, 'company', row, training_row)
					if offer.category == row.category:
						update_buying_history_feature(features, 'category', row, training_row)
					if offer.brand == row.brand:
						update_buying_history_feature(features, 'brand', row, training_row)
					if offer.company == row.company and offer.category == row.category and offer.brand == row.brand and not (row.dept in features_dept):
						features_dept.append(row.dept)
					if row.dept in features_dept:	
						update_buying_history_feature(features, 'dept', row, training_row)

				last_id = row.id
				if e % 100000 == 0:
					print e

if __name__ == '__main__':
	#DataReducer(loc_offers, loc_transactions, loc_reduced).reduce_data()
	generate_features(loc_train, loc_test, loc_reduced, loc_out_train, loc_out_test)