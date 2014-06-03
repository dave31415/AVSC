import pandas as pd
import numpy as np


# cat, comp, brand, chain= categorical
# compute av entropy=-Sum[P*log P] for each feature, then normalize it entropy/num(unique_feature)
#algorithm:
# find unique entry in each feature
# compute number of true for each unique entry , divide by total number of that entry-this is P
# we are interested in entropy of a feature, so sum over all such entries of a feature-this is av entropy
# different feature have different number of unique entries, we somehow have to normalize it (or not? may be?
# should discuss this!), so divide av entropy by num of unique entries

data_dir='data/'
# read train data and offer data, then combine them on offer
file_name='trainHistory.csv'
file_to_open=data_dir+file_name
try:
  train_history=pd.read_csv(file_to_open)
  #print train_history[:1]
except IOError:
    print 'expect filename in data/'
    exit()

file_name='offers.csv'
file_to_open=data_dir+file_name
try:
  offers=pd.read_csv(file_to_open)
except IOError:
    print 'expect filename in data/'
    exit()

joined_data= pd.merge(train_history, offers, left_on='offer', right_on='offer', how='inner')

print joined_data.columns.values # column names
print joined_data[0:1] # 1st line
del joined_data['quantity']  # dont need quantity
#  number of unique values
print len(set(joined_data['chain'])), len(set(joined_data['offer'])),len(set(joined_data['market'])),\
len(set(joined_data['category'])),len(set(joined_data['company'])),len(set(joined_data['offervalue'])),\
len(set(joined_data['brand']))



list_feature=['chain','company', 'brand','category']
print 'feature','avg. entropy', 'normalized entropy'
for cat in list_feature:
  unique= list(set(joined_data[cat])) # unique entries
  U=0.0 # initialize entropy
  for u in unique:
     P= joined_data[(joined_data[cat]==u)&(joined_data.repeater=='t')].shape[0]/float(joined_data.shape[0])
     if P!=0: tmp= P*np.log(P) #exclude 0
     U+=tmp
  print cat,-1*U, -1*U/len(unique) # negative to match the definition
# chain category brand company
# category company brand chain