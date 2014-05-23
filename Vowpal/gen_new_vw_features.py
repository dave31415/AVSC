# -*- coding: UTF-8 -*-

"""
Kaggle Challenge: 
"http://www.kaggle.com/c/acquire-valued-shoppers-challenge/" 
'Reduce the data and generate features' by Triskelion 
After a forum post by BreakfastPirate
Very mediocre and hacky code, single-purpose, but pretty fast
Some refactoring by Zygmunt Zaj������c <zygmunt@fastml.com>
More refactoring done
"""

from datetime import datetime, date
from data_reducer import *
from offer_row import *
from test_row import *
from train_row import *
from transaction_row import *
from csv import DictWriter

loc_offers = "../data/offers.csv"
loc_transactions = "../data/transactions.csv"
loc_train = "../data/trainHistory.csv"
loc_test = "../data/testHistory.csv"
loc_reduced = "../data/reduced.csv" 
loc_out_train = "../data/train.csv"
loc_out_march = "../data/train_march.csv"
loc_out_april = "../data/train_april.csv"
loc_out_test = "../data/test.csv"
loc_out_categories = "../data/categories.csv"
loc_out_items = "../data/items.csv"

id_index = 0

feature_fields = [
"has_bought_at_chain_before",
"has_bought_brand",
"has_bought_brand_180",
"has_bought_brand_30",
"has_bought_brand_60",
"has_bought_brand_90",
"has_bought_brand_a",
"has_bought_brand_a_180",
"has_bought_brand_a_30",
"has_bought_brand_a_60",
"has_bought_brand_a_90",
"has_bought_brand_category",
"has_bought_brand_company",
"has_bought_brand_company_category",
"has_bought_brand_dept",
"has_bought_brand_q",
"has_bought_brand_q_180",
"has_bought_brand_q_30",
"has_bought_brand_q_60",
"has_bought_brand_q_90",
"has_bought_category",
"has_bought_category_180",
"has_bought_category_30",
"has_bought_category_60",
"has_bought_category_90",
"has_bought_category_a",
"has_bought_category_a_180",
"has_bought_category_a_30",
"has_bought_category_a_60",
"has_bought_category_a_90",
"has_bought_category_q",
"has_bought_category_q_180",
"has_bought_category_q_30",
"has_bought_category_q_60",
"has_bought_category_q_90",
"has_bought_company",
"has_bought_company_180",
"has_bought_company_30",
"has_bought_company_60",
"has_bought_company_90",
"has_bought_company_a",
"has_bought_company_a_180",
"has_bought_company_a_30",
"has_bought_company_a_60",
"has_bought_company_a_90",
"has_bought_company_dept",
"has_bought_company_q",
"has_bought_company_q_180",
"has_bought_company_q_30",
"has_bought_company_q_60",
"has_bought_company_q_90",
"has_bought_dept",
"has_bought_dept_180",
"has_bought_dept_30",
"has_bought_dept_60",
"has_bought_dept_90",
"has_bought_dept_a",
"has_bought_dept_a_180",
"has_bought_dept_a_30",
"has_bought_dept_a_60",
"has_bought_dept_a_90",
"has_bought_dept_q",
"has_bought_dept_q_180",
"has_bought_dept_q_30",
"has_bought_dept_q_60",
"has_bought_dept_q_90",
"never_bought_brand",
"never_bought_category",
"never_bought_company",
"never_bought_dept",
"offer_brand",
"offer_category",
"offer_chain",
"offer_company",
"offer_date",
"offer_market",
"offer_quantity",
"offer_value",
"total_spend"]

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
	add_to_dict(features, ''.join(['has_bought_', feature_name]), 1.0)
	add_to_dict(features, ''.join(['has_bought_', feature_name, '_q']), row.purchasequantity)
	add_to_dict(features, ''.join(['has_bought_', feature_name, '_a']), row.purchaseamount)

	date_diff_days = diff_days(row.date,training_row.date)
	add_time_limited_history_features(30, row, features, date_diff_days, feature_name)
	add_time_limited_history_features(60, row, features, date_diff_days, feature_name)
	add_time_limited_history_features(90, row, features, date_diff_days, feature_name)
	add_time_limited_history_features(180, row, features, date_diff_days, feature_name)
	
def add_time_limited_history_features(num_days, row, features, date_diff_days, feature_name):
	if date_diff_days < num_days:
		add_to_dict(features, ''.join(['has_bought_', feature_name, '_', str(num_days)]), 1.0)
		add_to_dict(features, ''.join(['has_bought_', feature_name, '_q_', str(num_days)]), row.purchasequantity)
		add_to_dict(features, ''.join(['has_bought_', feature_name, '_a_', str(num_days)]), row.purchaseamount)

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

def output_features(features, last_id, test, out_train, out_test, out_march, out_april):
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

	in_march = diff_days('2013-04-1', features['offer_date']) < 0
	if test:
		out_test.writerow( features )
	else:
		out_train.writerow( features )
		if in_march:
			out_march.writerow( features )
		else:
			out_april.writerow( features )

def reset_features(row, training_row, train_ids, offers):
	features = dict()
	offer = offers[ training_row.offer]
	features['offer_value'] = offer.value
	features['offer_quantity'] = offer.quantity
	features['offer_date'] = training_row.date	
	features['offer_brand'] = offer.brand
	features['offer_category'] = offer.category
	features['offer_chain'] = training_row.chain
	features['offer_company'] = offer.company
	features['offer_market'] = training_row.market
	features['total_spend'] = 0.0
	return features

def output_items_and_categories(items, categories):
	items_cols = ["item_id", "category", "company", "brand", "total", "amount", "quantity"]
	items_headers = dict()
	items_dw = DictWriter(open(loc_out_items, "wb"), delimiter=',', fieldnames=items_cols)
	for n in items_cols:
		items_headers[n] = n
	items_dw.writerow(items_headers)
	for key in items.iterkeys():
		keys = key.split("_")
		items[key]['item_id'] = key
		items[key]['category'] = keys[0]
		items[key]['company'] = keys[1]
		items[key]['brand'] = keys[2]
		items_dw.writerow(items[key])
			
	categories_cols = ["category", "total", "amount", "quantity"]
	categories_header = dict()
	categories_dw = DictWriter(open(loc_out_categories, "wb"), delimiter=',', fieldnames=categories_cols)
	for n in categories_cols:
		categories_header[n] = n
	categories_dw.writerow(categories_header)
	for key in categories.iterkeys():
		categories[key]['category'] = key
		categories_dw.writerow(categories[key])
			
def update_items_history(items, row):
	item_id = str(row.category) + "_" + str(row.company) + "_" + str(row.brand)
	if item_id not in items.keys():
		items[item_id] = dict()
	add_to_dict(items[item_id], "total", 1.0)
	add_to_dict(items[item_id], "amount", row.purchaseamount)
	add_to_dict(items[item_id], "quantity", row.purchasequantity)

def update_categories_history(categories, row):
	if row.category not in categories.keys():
		categories[row.category] = dict()
	add_to_dict(categories[row.category], "total", 1.0)
	add_to_dict(categories[row.category], "amount", row.purchaseamount)
	add_to_dict(categories[row.category], "quantity", row.purchasequantity)

def generate_features(loc_train, loc_test, loc_transactions, loc_out_train, loc_out_test):
	offers = load_offer_rows()
	train_ids = load_training_rows()
	test_ids = load_test_rows()
	items = dict()
	categories = dict()
	out_train = DictWriter(open(loc_out_train, "wb"), delimiter=',', fieldnames=feature_fields)
	out_test = DictWriter(open(loc_out_test, "wb"), delimiter=',', fieldnames=feature_fields)
	out_march = DictWriter(open(loc_out_march, "wb"), delimiter=',', fieldnames=feature_fields)
	out_april = DictWriter(open(loc_out_april, "wb"), delimiter=',', fieldnames=feature_fields)

	feature_headers = dict()
	for n in feature_fields:
		feature_headers[n] = n
	out_train.writerow(feature_headers)
	out_test.writerow(feature_headers)
	out_march.writerow(feature_headers)
	out_april.writerow(feature_headers)

	last_id = 0
	features = dict()
	features_dept = list()
	start = datetime.now()
	for e, line in enumerate( open(loc_transactions) ):
		if e > 0: #skip header
			row = TransactionRow(line.strip())
			new_shopper = last_id != row.id
			if new_shopper and e != 1:
				output_features(features, last_id, row.id in test_ids, out_train, out_test, out_march, out_april)
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
				if row.dept in features_dept:	
					update_buying_history_feature(features, 'dept', row, training_row)
				if offer.company == row.company and offer.category == row.category and offer.brand == row.brand and not (row.dept in features_dept):
					features_dept.append(row.dept)
			diff = diff_days(row.date, '2013-04-01')
			transaction_within_180_days = 0 < diff and diff < 180
			if transaction_within_180_days:
				update_items_history(items, row)
				update_categories_history(categories, row)

			last_id = row.id
			if e % 100000 == 0:
				print e, datetime.now() - start
	print datetime.now() - start
	output_items_and_categories(items, categories)
	
if __name__ == '__main__':
	generate_features(loc_train, loc_test, loc_reduced, loc_out_train, loc_out_test)

