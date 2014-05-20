import pandas as pd
import time
import os
import numpy as np
import sys
from collections import defaultdict
import csv

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,\
    AdaBoostClassifier, ExtraTreesClassifier
from sklearn import metrics
from sklearn import preprocessing
from scipy.sparse import csr_matrix
from sklearn import linear_model, svm
from datetime import datetime, timedelta

class DataWrangling:

   def __init__(self, data_dir, train_file, test_file, offer_file, transaction_file):
        self._data_dir= data_dir
        self._train_file= train_file
        self._test_file= test_file
        self._offer_file= offer_file
        self._transaction_file=transaction_file
        self._joined_data= None

   def get_filepath(self,file):
       return self._data_dir + file

   def combine_trainhistory_offers(self, outfile=''):
       # this code combines trainHistory+offers on offer
       outfile=self.get_filepath(outfile)
       file_to_open=self.get_filepath(self._train_file)
       train_history=pd.read_csv(file_to_open)
       #print train_history[:1]

       file_to_open=self.get_filepath(self._offer_file)
       offers=pd.read_csv(file_to_open)

       self._joined_data= pd.merge(train_history, offers, left_on='offer', right_on='offer', how='inner')
       #print joined_data.shape
       #print joined_data[:1]
       print self._joined_data[:1]
       self._joined_data.to_csv(outfile)

   def reduce_transaction(self, outfile= ''):
       #this code reads in transactions data, print out the rows that has matching category with offers.csv
       outfile=self.get_filepath(outfile)
       offers_cat = {}
       t_i=time.time()
       for e, line in enumerate( open(self.get_filepath(self._offer_file))):
           offers_cat[ line.split(",")[1] ] = 1
       print offers_cat
       t_i=time.time()
       with open(outfile, "wb") as outfile:
             [outfile.write(line) for e,line in enumerate(open(self.get_filepath(self._transaction_file)))
              if line.split(",")[3] in offers_cat]
       print 'time takes ',time.time()-t_i

   def aggregate_transaction(self, infile='', outfile=''):
       #this code read reduced.csv (obtained from transactions file by keeping the
       # rows whose category exists in offers.csv), create dictionaries which sum over
       # productsize, purchasequantity and purchaseamount for each [customer,category] pair
       # then writes the above dictionaries key, value pair to csv file
       infile=self.get_filepath(infile)
       outfile=self.get_filepath(outfile)

       if os.path.isfile(infile) is False:
           print 'need to run reduce_transaction first'
           return 0
       productsize, purchasequantity, purchaseamount=defaultdict(float), defaultdict(int), defaultdict(float)
       t_i= time.time()
       with open(infile) as infile:
          next(infile)
          for line in infile:
             entries=line.split(',')
             productsize[(entries[0],entries[3])]+=float(entries[7])
             purchasequantity[(entries[0],entries[3])]+=int(entries[9])
             purchaseamount[(entries[0],entries[3])]+= float(entries[10])

       ct=0
       with open(outfile,'w') as f:
            write=csv.writer(f,delimiter=',')
            write.writerow(['id','category','productsize','purchasequantity','purchaseamount'])
            for key, value in productsize.items():
                 #print key[0],key[1],value,purchasequantity[key],purchaseamount[key]
                 write.writerow([key[0],key[1],value,purchasequantity[key],purchaseamount[key]])
                 ct+=1
                 if ct%1000000==0: print 'no of lines read ', ct
       print 'time takes ',time.time() -t_i

   def combine_hist_offer_transaggr(self, infile= '', outfile='', outfile1='hist_offers.csv'):
       #this code reads in the  transactions-aggr file and join with
       #  joined_data  on [customer id,category]
       infile= self.get_filepath(infile)
       outfile= self.get_filepath(outfile)
       if os.path.isfile(infile) is False:
           print 'need to run aggregate_transaction first'
           return 0
       if self._joined_data is None:
           self.combine_trainhistory_offers(outfile=outfile1)

       trans_aggr=pd.read_csv(infile)
       print trans_aggr[:1]
       hist_offers_transaggr= pd.merge(self._joined_data, trans_aggr, left_on=['id','category'],
                                right_on=['id','category'], how='left')
       hist_offers_transaggr.to_csv(outfile)


class AnalyzePredict(DataWrangling):

    def __init__(self, data_dir, infile, date_begin, date_end):
        self._data_dir= data_dir
        if os.path.isfile(self.get_filepath(infile)) is False:
              print 'need to run data wrangling first, hist_offers_transaggr.csv missing'
              exit()
        else:
            self._infile= self.get_filepath(infile)
            self._data =None
            self._date_begin= date_begin
            self._date_end= date_end

    def __date_to_split(self, n_cv_loop):
        begin=datetime.strptime(self._date_begin,'%m-%d-%Y')
        end=datetime.strptime(self._date_end,'%m-%d-%Y')
        if n_cv_loop>1:
           n_days= (end-begin).days
           days_random=np.random.rand()*n_days
           return str(timedelta(days=days_random) + begin)
        else:
           return str(begin)

    def __split_data_into_train_cv(self, date_to_split):
        train_data=self._data[self._data.offerdate <= date_to_split].ix[:,:]
        cv_data=self._data[self._data.offerdate > date_to_split].ix[:,:]
        train_response=train_data['repeater']
        cv_response=cv_data['repeater']
        del train_data['offerdate']
        del train_data['repeater']
        del cv_data['offerdate']
        del cv_data['repeater']
        return train_data, cv_data, train_response, cv_response

    def __onehot_and_normalize_data(self, train_data, cv_data):
        column_names=list(train_data.columns.values)
        columnnames={}
        for e,column_name in enumerate(column_names): columnnames[column_name]=e

        categoricalfeatures= list([columnnames['chain'], columnnames['offer'],
                                   columnnames['market'], columnnames['category'],
                                   columnnames['company'], columnnames['brand']])

        enc = preprocessing.OneHotEncoder(categorical_features=categoricalfeatures)
        enc.fit(np.vstack((train_data,cv_data)))
        train_data_1hot=enc.transform(train_data).toarray()
        cv_data_1hot=enc.transform(cv_data).toarray()

        min_max_scaler = preprocessing.MinMaxScaler()
        min_max_scaler.fit(train_data_1hot)

        train_data_1hot_scaled =min_max_scaler.transform(train_data_1hot)
        cv_data_1hot_scaled= min_max_scaler.transform(cv_data_1hot)
        return train_data_1hot_scaled, cv_data_1hot_scaled


    def read_aggr_data(self):
        print 'reading the aggregated data now...'
        file=self._infile
        self._data=pd.read_csv(file)
        del self._data['Unnamed: 0'] #redundant
        del self._data['quantity']  # only 1 unique val, so useless
        self._data['productsize_q']=self._data['productsize']/self._data['purchasequantity']
        self._data['purchaseamount_q']=self._data['purchaseamount']/self._data['purchasequantity']
        # replace missing vals-> 0
        self._data.replace(to_replace=[np.nan,np.inf,-np.inf],value=[0,0,0],inplace=True)
        self._data.replace(to_replace=['t','f'],value=[1,0],inplace=True) # replace t,f->1,0
        del self._data['productsize'] # redundant w per quantity features above
        del self._data['purchaseamount'] # ditto
        del self._data['id'] # not useful
        del self._data['repeattrips'] # cant use , not present in test
        print '# rows, columns in data: ',self._data.shape

        #for column_name in column_names[:9]:
        #    print column_name,len(set(self._data[column_name])),list(set(self._data[column_name]))[0:10]

        return self._data

    def hyperparam(self, model, n_cv_loop =5):
           print "performing hyperparameter selection..."
           # Hyperparameter selection loop
           score_hist = []
           #Xt = sparse.hstack([Xts[j] for j in good_features]).tocsr()
           #print Xt.shape
           Cvals = np.logspace(-4, 4, 15, base=2)
           for C in Cvals:
               model.C = C
               auc, std_auc = self.cv_loop(model, n_cv_loop=n_cv_loop)
               score_hist.append((auc,C))
               print "C: %f Mean AUC: %f" %(C,auc)
           best_auc, best_c = sorted(score_hist)[-1]
           print "Best C value: %f, %f" % (best_c, best_auc)
           return best_c, best_auc


    def cv_loop(self, model, bestc, n_cv_loop=5):
        if self._data is None:
            print 'run read_aggr_data first'
            exit()
        model.C= bestc
        mean_auc, std_auc= 0, 0
        print 'performing cv loop using model= ',model
        for i in xrange(n_cv_loop):
            print 'starting cv loop ',i
            date=self.__date_to_split(n_cv_loop=n_cv_loop)
            train_data, cv_data, train_response, cv_response= \
                self.__split_data_into_train_cv(date)
            print 'features used: ',train_data.columns.values
            train_data_scaled, cv_data_scaled= self.__onehot_and_normalize_data(train_data.ix[:,:],
                                                                                cv_data.ix[:,:])
            print '#rows, columns of train and cv scaled: ',train_data_scaled.shape, cv_data_scaled.shape,\
                'using date to split: ',date
            model.fit(train_data_scaled, train_response)
            #preds = model.predict_proba(cv_data)[:,1]
            preds_binary= model.predict(cv_data_scaled)
            auc = metrics.roc_auc_score(cv_response, preds_binary)
            print  "auc= ", auc
            mean_auc += auc
            std_auc += auc*auc
        mean_auc/=n_cv_loop
        std_auc=(std_auc/n_cv_loop - mean_auc**2)**0.5
        return mean_auc, std_auc


    """
        SEED=25
        def create_cv_randomsubset_from_cvdata(cv_data, cv_number=8):
        X_cv, X1_cv, y_cv, y1_cv = cross_validation.train_test_split(
        cv_data,cv_response, test_size=.50,
        random_state = i*SEED)
        """
    def feature_importance(self, model):
        date=self.__date_to_split(n_cv_loop=1)
        train_data, cv_data, train_response, cv_response= \
                self.__split_data_into_train_cv(date)
        model.fit(train_data, train_response)
        for (column,imp) in sorted(zip(list(train_data.columns.values), model.feature_importances_)
                          ,key=lambda x: x[1], reverse=True):
              print column, imp




if __name__=="__main__":
   data_dir= 'data/'
   train_file= 'trainHistory.csv'
   test_file= 'testHistory.csv'
   offer_file= 'offers.csv'
   transaction_file= 'transactions.csv'
   hist_offers_file= 'hist_offers.csv'
   reduced_transaction_file= 'reduced.csv'
   transaction_aggr_file= 'transactions-aggr.csv'
   hist_offers_transaction_file= 'hist_offers_transaggr.csv'

   model = linear_model.LogisticRegression()
   date_to_split_begin= '04-01-2013'
   date_to_split_end= '04-20-2013'
   case= None
   best_c=4.876
   model_for_feature_imp= AdaBoostClassifier(n_estimators=100)
   #model_for_feature_imp= ExtraTreesClassifier(n_estimators=100)
   if len(sys.argv)!=2:
       print 'python kaggle-shopper.py {wrangling|analyze|both}'
       exit()
   else:
       case=sys.argv[1]

   if case =='wrangling' or case=='both':
      dw=DataWrangling(data_dir, train_file, test_file, offer_file, transaction_file)
      dw.combine_trainhistory_offers(outfile=hist_offers_file)
      dw.reduce_transaction(outfile=reduced_transaction_file)
      dw.aggregate_transaction(infile=reduced_transaction_file, outfile=transaction_aggr_file)
      dw.combine_hist_offer_transaggr(infile= transaction_aggr_file, outfile= hist_offers_transaction_file)

   if case=='analyze' or case=='both':
       An=AnalyzePredict(data_dir, infile=hist_offers_transaction_file, date_begin=date_to_split_begin
       , date_end=date_to_split_end)
       An.read_aggr_data()
       #An.feature_importance(model=model_for_feature_imp)
       #best_c, best_auc= An.hyperparam(model, n_cv_loop=1)
       #print 'best C, best AUC: ', best_c, best_auc
       mean_auc, std_auc=An.cv_loop(model, bestc=best_c, n_cv_loop=5)
       print 'mean, std AUC: ', mean_auc, std_auc







