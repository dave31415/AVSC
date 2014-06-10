import pandas as pd
import time
import os
import numpy as np
import sys
from collections import defaultdict
import csv
from itertools import combinations
from scipy import sparse
import math

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,\
    AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn import metrics
from sklearn import preprocessing
from scipy.sparse import csr_matrix
from sklearn import linear_model, svm
from datetime import datetime, timedelta, date

class DataWrangling:

   def __init__(self, data_dir, offer_file, transaction_file):
        self._data_dir= data_dir
        self._offer_file= offer_file
        self._transaction_file=transaction_file
        self._joined_data= None

   def get_filepath(self,file):
       return self._data_dir + file

   def combine_history_offers(self, infile,  outfile=''):
       # this code combines trainHistory+offers on offer
       outfile=self.get_filepath(outfile)
       file_to_open=self.get_filepath(infile)
       train_history=pd.read_csv(file_to_open)
       #print train_history[:1]

       file_to_open=self.get_filepath(self._offer_file)
       offers=pd.read_csv(file_to_open)

       self._joined_data= pd.merge(train_history, offers, left_on='offer', right_on='offer', how='inner')
       #print joined_data.shape
       #print joined_data[:1]
       try:
          del self._joined_data['repeattrips']
       except KeyError:
           print 'test file dont have repeattrips'
       del self._joined_data['quantity']
       print self._joined_data[:1]
       self._joined_data.to_csv(outfile)

   def reduce_transaction(self, outfile= ''):
       #this code reads in transactions data, print out the rows that has matching category or company or brand
       # with offers.csv
       outfile=self.get_filepath(outfile)
       offers_cat, offers_comp, offers_brand = {},{},{}

       for e, line in enumerate( open(self.get_filepath(self._offer_file))):
           lsplit=line.split(",")
           offers_cat[lsplit[1]] = 1
           offers_comp[lsplit[3]] = 1
           offers_brand[lsplit[5]] = 1
       print offers_cat, offers_comp, offers_brand
       t_i=time.time()
       with open(outfile, "wb") as outfile:
             [outfile.write(line) for e,line in enumerate(open(self.get_filepath(self._transaction_file)))
              if line.split(",")[3] in offers_cat or line.split(",")[4] in offers_comp or line.split(",")[5]
              in offers_brand]
       print 'time takes ',time.time()-t_i

   def diff_days(self, s1, s2):
        date_format = "%Y-%m-%d"
        a = datetime.strptime(s1, date_format)
        b = datetime.strptime(s2, date_format)
        delta = a - b
        return delta.days

   def add_past_purchase_last_n_days(self, infile='', infile1='', outfile=''):
       infile=self.get_filepath(infile)
       infile1=self.get_filepath(infile1)
       outfile=self.get_filepath(outfile)
       offer_date=defaultdict(str)
       date_list=['0-29 days','30-59 days','60-89 days','90-179 days','>=180 days']
       array_pos={'0':0,'1':1,'2':2,'3':3,'4':3,'5':3}
       for i in xrange(6,20):
           array_pos[str(i)]=4
       boolean_list=["%d" % (0) for x in xrange(len(date_list))]
       arrpos=None
       with open(infile) as file:
           next(file)
           for e, line in enumerate(file ):
               lsplit=line.split(",")
               offer_date[lsplit[0]]=lsplit[6].strip("\n")

       out=open(outfile, "wb")
       ct,ct1=0,0
       t_i=time.time()
       with open(infile1) as infile:
            head=next(infile).replace('\n',',')+'0-29 days,'+'30-59 days,'+'60-89 days,'+\
                     '90-179 days,'+'>=180 days'
            out.write(head+'\n')
            for line in infile:
                lstr=line.strip('\n')
                entries=lstr.split(',')
                if offer_date[entries[0]]:
                     days=self.diff_days(offer_date[entries[0]], entries[6])
                     ct1+=1
                     #print days, array_pos[str(int(math.floor(days/30)))]
                     arrpos=str(int(math.floor(days/30)))
                     if days>=0: boolean_list[array_pos[arrpos]]=str(1)
                     total=lstr+","+",".join(boolean_list)
                     total+='\n'
                     if days>=0: boolean_list[array_pos[arrpos]]=str(0)
                     out.write(total)
                ct+=1
                if ct % 2e6==0:  print ct,ct-ct1
       out.close()
       print time.time()-t_i


   def aggregate_transaction(self, infile='', outfile=''):
       #this code read reduced.csv (obtained from transactions file by keeping the
       # rows whose category exists in offers.csv), create dictionaries which sum over
       # productsize, purchasequantity and purchaseamount for each [customer,category/comp etc] pair
       infile=self.get_filepath(infile)
       outfile=self.get_filepath(outfile)

       if os.path.isfile(infile) is False:
           print 'need to run reduce_transaction first'
           return 0
       productsize, purchasequantity, purchaseamount= defaultdict(float), defaultdict(int), defaultdict(float)

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

   def __data_grp(self, mergedata,data, cat_combinations, cat_combinations_to_join, func_to_apply):
        data_grp=data.groupby(cat_combinations,as_index=False).agg(func_to_apply)
        mergedata=pd.merge(mergedata, data_grp, left_on=cat_combinations,
                               right_on=cat_combinations_to_join, how='left')
        for cat in cat_combinations_to_join: del mergedata[cat]
        return mergedata

   def __rename_data(self, mergedata, names, new_names):
        mergedata.rename(columns={names[-4]: new_names[0], names[-3]: new_names[1],
                                        names[-2]: new_names[2], names[-1]: new_names[3]}, inplace=True)

   def __by_date(self, data, col_names, col_num):
      data_bin=[]
      ct=0
      for j in xrange(12,col_num):
            data_bin.append(data[data[col_names[j]]==1])
            print ct, data_bin[ct].shape
            ct+=1
      return data_bin

   def combine_hist_offer_transaggr_v1(self, infile='', infile1='',infile2='',outfile1='', outfile2=''):
       #extend above aggr_transaction
       #and combinations of cat X comp etc
       # then writes the above dictionaries key, value pair to csv file
       # for brevity, we use the following naming scheme of generated categories:
       # chain, category, company, brand=0,1,2,3
       # cat X comp etc
       # then over a time of last 30 days 2_30, 24_30 etc
       #from trainhist+offer we have id, chain, market, category, company, brand
       #from reduced, we have id, chain, dept, category, company, brand
       #Q. how to combine these two files?
       #common=id, chain, category, company, brand

       infile=self.get_filepath(infile)
       infile1=self.get_filepath(infile1)
       infile2=self.get_filepath(infile2)
       outfile1=self.get_filepath(outfile1)
       outfile2=self.get_filepath(outfile2)

       if os.path.isfile(infile1) is False or os.path.isfile(infile2) is False:
                   print 'run combine_history_offers for train and test first'
                   return 0
       if os.path.isfile(infile) is False:
           print 'need to run reduce_transaction first'
           return 0

       t_i=time.time()

       mergedata_train=pd.read_csv(infile1)
       mergedata_test=pd.read_csv(infile2)
       cat_to_gen=['category','company','brand'] #['chain',
       func_to_apply={'productsize':[len,np.median],
                      'purchasequantity':[np.median],
                      'purchaseamount':[np.median]}
       #lambda x: np.percentile(x['productsize'], q = 95)
       list_cat_to_divide = [('productsize','len')] #,('productsize', 'median'),
                                 #('purchaseamount', 'median'),('purchasequantity','median')]

       cat_combinations, cat_combinations_to_join=[],[]
       for i in xrange(len(cat_to_gen)+1):
            for indicies in combinations(cat_to_gen, i):
               indicies=list(indicies)
               indicies.insert(0,'id')
               cat_combinations.append(indicies)
               cat_combinations_to_join.append([(i,'') for i in indicies] )
       #print cat_combinations
       #print cat_combinations_to_join
       data=pd.read_csv(infile)
       col_names=data.columns.values
       col_num=len(col_names)
       t_1=time.time()-t_i
       data_grp_to_divide=None
       if col_num>11: data_bin=self.__by_date(data,col_names, col_num)
       print t_1
       t_1=time.time()
       for i in xrange(len(cat_combinations)):
            mergedata_train= self.__data_grp(mergedata_train,data,cat_combinations[i],
                                                    cat_combinations_to_join[i], func_to_apply)
            mergedata_test= self.__data_grp(mergedata_test,data,cat_combinations[i],
                                                    cat_combinations_to_join[i], func_to_apply)

            new_names=[name+str(i) for name in ['len_','ps_median_','pq_median_','pa_median_']]
            names=mergedata_train.columns.values
            self.__rename_data(mergedata_train,names,new_names)
            self.__rename_data(mergedata_test,names,new_names)
            print mergedata_train.columns.values, mergedata_test.columns.values
            print i ,' feature aggregation+merging done'

            if col_num>=11:
               for j in xrange(col_num-12):
                   mergedata_train= self.__data_grp(mergedata_train,data_bin[j],cat_combinations[i],
                                                    cat_combinations_to_join[i], func_to_apply)
                   mergedata_test= self.__data_grp(mergedata_test,data_bin[j],cat_combinations[i],
                                                    cat_combinations_to_join[i], func_to_apply)

                   new_names=[name+"_"+col_names[j+12]+"_"+str(i) for name in
                              ['len','ps_median','pq_median','pa_median']]
                   names=mergedata_train.columns.values
                   self.__rename_data(mergedata_train,names,new_names)
                   self.__rename_data(mergedata_test,names,new_names)
                   print mergedata_train.columns.values, mergedata_test.columns.values
                   print j, col_names[j+12],' done'
               print i ,' feature aggregation+merging done'

       mergedata_train.to_csv(outfile1)
       mergedata_test.to_csv(outfile2)
       t_2=time.time()-t_1
       print t_2

   def groupby_func(self, cat_combinations, cat_combinations_to_join):
           data_grp=data.groupby(cat_combinations,as_index=False).agg(func_to_apply)

           #if i==0:  data_grp_to_divide = data_grp[list_cat_to_divide]

           mergedata_train=pd.merge(mergedata_train, data_grp, left_on=cat_combinations,
                               right_on=cat_combinations_to_join, how='left')
           mergedata_test=pd.merge(mergedata_test, data_grp, left_on=cat_combinations[i],
                               right_on=cat_combinations_to_join, how='left')

           #for cat in list_cat_to_divide:
                 #print mergedata_train[cat],data_grp_to_divide[cat]
           #      mergedata_train[cat]=mergedata_train[cat]/data_grp_to_divide[cat]
           #      mergedata_test[cat]=mergedata_test[cat]/data_grp_to_divide[cat]


           for cat in cat_combinations_to_join:
               del mergedata_train[cat]
               del mergedata_test[cat]
           new_names=[name+str(i) for name in ['ps_median_','pq_median_','pa_median_']]
           names=mergedata_train.columns.values
           mergedata_train.rename(columns={names[-3]: new_names[0],
                                names[-2]: new_names[1], names[-1]: new_names[2]}, inplace=True)
           mergedata_test.rename(columns={names[-3]: new_names[0],
                                names[-2]: new_names[1], names[-1]: new_names[2]}, inplace=True)

           #if i==0:
           #     del mergedata_train['len_0']
           #     del mergedata_test['len_0']
           print mergedata_train.columns.values, mergedata_test.columns.values
           print i ,' feature aggregation+merging done'

   def combine_hist_offer_transaggr(self, infile= '', outfile=''):
       #this code reads in the  transactions-aggr file and join with
       #  joined_data  on [customer id,category]
       infile= self.get_filepath(infile)
       outfile= self.get_filepath(outfile)
       if os.path.isfile(infile) is False:
           print 'need to run aggregate_transaction first'
           return 0
       if self._joined_data is None:
          print 'need to uncomment combine_hist_offers first'
          return 0
       trans_aggr=pd.read_csv(infile)
       #print trans_aggr[:1]
       hist_offers_transaggr= pd.merge(self._joined_data, trans_aggr, left_on=['id','category'],
                                right_on=['id','category'], how='left')
       hist_offers_transaggr.to_csv(outfile)


class AnalyzePredict(DataWrangling):

    def __init__(self, dir, date_begin='04-01-2013', date_end='04-20-2013'):
            self._data_dir = dir
            self._traindata = None
            self._trainresponse = None
            self._testdata = None
            self._data_norm = None
            self._data_1hot = None
            self._n_cat = 6
            self._date_begin = date_begin
            self._date_end = date_end
            self._numtrain=None
            self._testid=None
            self._testid_bs=None

    def __date_to_split(self, n_cv_loop):
        begin=datetime.strptime(self._date_begin,'%m-%d-%Y')
        end=datetime.strptime(self._date_end,'%m-%d-%Y')
        if n_cv_loop>1:
           n_days= (end-begin).days
           days_random=np.random.rand()*n_days
           return str(timedelta(days=days_random) + begin)
        else:
           return str(begin)


    def __split_data_into_train_cv(self, data, response, n_cv_loop, fraction=0.67, spread=0.25):
        max_row = data.shape[0]
        if n_cv_loop>1: fraction = np.random.ranf()*spread + fraction
        row_cut = int(math.floor(max_row*fraction))
        print 'row no to split train-cv: ',row_cut
        train_data = data[:row_cut]
        cv_data = data[row_cut:]
        train_response= response[:row_cut]
        cv_response=response[row_cut:]
        return train_data, cv_data, train_response, cv_response

    def __normalize_continuous_features(self, continuous_features, data):
        data_to_norm= data[:,continuous_features].astype(float)
        min_max_scaler = preprocessing.MinMaxScaler() # StandardScaler()
        min_max_scaler.fit(data_to_norm)
        self._data_norm =sparse.csr_matrix(min_max_scaler.transform(data_to_norm))
        del data_to_norm
        print 'normalizing continuous features done...'

    def __onehot_encoder(self,data, keymap=None):
         """
         OneHotEncoder takes data matrix with categorical columns and
         converts it to a sparse binary matrix.

         Returns sparse binary matrix and keymap mapping categories to indicies.
         If a keymap is supplied on input it will be used instead of creating one
         and any categories appearing in the data that are not in the keymap are
         ignored
         """
         if keymap is None:
              keymap = []
              for col in data.T:
                   uniques = set(list(col))
                   keymap.append(dict((key, i) for i, key in enumerate(uniques)))
         total_pts = data.shape[0]
         outdat = []
         for i, col in enumerate(data.T):
              km = keymap[i]
              num_labels = len(km)
              spmat = sparse.lil_matrix((total_pts, num_labels))
              for j, val in enumerate(col):
                   if val in km:
                        spmat[j, km[val]] = 1
              outdat.append(spmat)
         outdat = sparse.hstack(outdat).tocsr()
         return outdat, keymap

    def __onehot_categorical_features(self, categorical_features, data):
        data_to_1hot= data[:, categorical_features]
        self._data_1hot, keymap = self.__onehot_encoder(data_to_1hot)
        #enc = preprocessing.OneHotEncoder()
        #enc.fit(data_to_1hot)
        #self._data_1hot = enc.transform(data_to_1hot)
        del data_to_1hot
        #print self._data_1hot
        #exit()
        print '1hot encoding done...'

    def normalize_data(self,  add_data, case):
        if self._traindata is None:
            print 'run read_aggr_data w case=train first'
            exit()
        self._numtrain = np.shape(self._traindata)[0]
        if add_data!=None:
             if case=='train':
                 traindata_aggr= np.hstack((self._traindata.ix[:,:self._n_cat],add_data[:self._numtrain],
                                   self._traindata.ix[:,self._n_cat:]))
             else:
                 data= np.vstack((self._traindata,self._testdata))
                 traindata_aggr= np.hstack((data[:,:self._n_cat],add_data,data[:,self._n_cat:]))
             categorical = [i for i in xrange(self._n_cat + add_data[:self._numtrain].shape[1])]
             print self._traindata.ix[:,:self._n_cat].columns.values,\
                    self._traindata.ix[:,self._n_cat:].columns.values
        else:
             if case=='train':
                  traindata_aggr= np.array(self._traindata)
             else:
                  traindata_aggr=np.vstack((self._traindata,self._testdata))
             categorical = [i for i in xrange(self._n_cat)]

        continuous = [i for i in range(len(categorical), len(categorical) +
                                                         self._traindata.shape[1] - self._n_cat)]
        print 'categorical: ',categorical,'continuous: ', continuous

        self.__normalize_continuous_features(continuous, traindata_aggr)
        self.__onehot_categorical_features(categorical, traindata_aggr)
        #data_1hot_scaled = self._data_1hot.tocsr() #self._data_norm.tocsr()
        data_1hot_scaled = sparse.hstack((self._data_1hot, self._data_norm)).tocsr()
        print 'normalizing data done...'
        return data_1hot_scaled

    def group_data(self, degree=3, hash=hash):
        """
        numpy.array -> numpy.array
        if h < 0:
          h += sys.maxsize
         Groups all columns of data into all combinations of triples
        """
        data=np.vstack((self._traindata.ix[:,:self._n_cat],self._testdata.ix[:,:self._n_cat]))
        new_data = []
        m,n = data.shape
        for indicies in combinations(range(n), degree):
            #for v in data[:,indicies]:
                #h=hash(tuple(v))
                #if h<0: h += sys.maxsize
                #new_data.append(h)
            new_data.append([hash(tuple(v)) for v in data[:,indicies]])
        print 'shape of group data ',np.array(new_data).T.shape
        return np.array(new_data).T

    def read_aggr_data(self, file, case):
        print  file
        if os.path.isfile(self.get_filepath(file)) is False:
                 print 'need to run data wrangling first, train[test]hist_offers_transaggr.csv missing'
                 exit()

        print 'reading the aggregated data now...'
        data=pd.read_csv(self.get_filepath(file))
        #print data.columns.values[0:4]
        for i in range(2): del data[data.columns.values[i-i]] # redundant
        #del data[data.columns.values[1]] #redundant
        #print data.columns.values[0:2]

        #data['productsize_q']= data['productsize']/data['purchasequantity']
        #data['purchaseamount_q']= data['purchaseamount']/data['purchasequantity']
        # replace missing vals-> 0
        data.replace(to_replace=[np.nan,np.inf,-np.inf], value=[0,0,0], inplace=True)
        data.replace(to_replace=['t','f'], value=[1,0], inplace=True) # replace t,f->1,0
        #del data['productsize'] # redundant w per quantity features above
        #del data['purchaseamount'] # ditto

        dummy=data['offervalue']
        del data['offervalue']
        data['offervalue']= dummy
        data=data.sort('offerdate')
        del data['offerdate']
        if case=='test': self._testid=data['id'].values
        del data['id'] # not useful

        if case=='train':
            self._trainresponse = data['repeater'].values
            del data['repeater']
            self._traindata =data
            print '# rows, columns in train data features and response: ', \
                self._traindata.shape, self._trainresponse.shape
            print 'columns are: ', self._traindata.columns.values
        else:
            self._testdata=data
            print '# rows, columns in test data: ', self._testdata.shape
            print 'columns are: ', self._traindata.columns.values
        #for column_name in column_names[:9]:
        #    print column_name,len(set(self._data[column_name])),list(set(self._data[column_name]))[0:10]


    def hyperparam(self, model, data, n_cv_loop =5):
           print "performing hyperparameter selection..."
           # Hyperparameter selection loop
           score_hist = []
           #Xt = sparse.hstack([Xts[j] for j in good_features]).tocsr()
           #print Xt.shape
           Cvals = np.logspace(-4, 4, 15, base=2)
           for C in Cvals:
               auc, std_auc = self.cv_loop(model, bestc=C, data=data, n_cv_loop=n_cv_loop)
               score_hist.append((auc,C))
               print "C: %f Mean AUC: %f" %(C,auc)
           best_auc, best_c = sorted(score_hist)[-1]
           print "Best C value: %f, %f" % (best_c, best_auc)
           return best_c, best_auc


    def cv_loop(self, model, bestc, data, n_cv_loop=5):
        model.C= bestc
        mean_auc, std_auc= 0, 0

        print '#rows, columns of scaled data: ',data.shape

        print 'performing cv loop using model= ',model
        for i in xrange(n_cv_loop):
            print 'starting cv loop ',i
            train_data, cv_data, train_response, cv_response= \
                self.__split_data_into_train_cv(data, self._trainresponse, n_cv_loop)
            print 'train and cv data shape', train_data.shape, cv_data.shape
            model.fit(train_data, train_response)
            #preds = model.predict_proba(cv_data)[:,1]
            preds_binary= model.predict(cv_data)
            auc = metrics.roc_auc_score(cv_response, preds_binary)
            print  "auc= ", auc
            mean_auc += auc
            std_auc += auc*auc
        mean_auc/=n_cv_loop
        std_auc=(std_auc/n_cv_loop - mean_auc**2)**0.5
        return mean_auc, std_auc

    def predict_test(self, model, data):
        X_train = data[:self._numtrain]
        X_test = data[self._numtrain:]

        model.fit(X_train, self._trainresponse)
        prediction=defaultdict(float)
        print "Making prediction and saving results..."
        preds= model.predict_proba(X_test)[:,1]
        for i,p in enumerate(preds):
            prediction[self._testid[i]]=p

        return prediction

    def create_test_submission(self, filename, prediction, file='sampleSubmission.csv'):
        data=pd.read_csv(self.get_filepath(file))
        id=data.id.values
        content = ['id,repeatProbability']
        for i in xrange(len(id)):
            content.append('%i,%f' %(id[i],prediction[id[i]]))
        f = open(filename, 'w')
        f.write('\n'.join(content))
        f.close()
        print 'Saved'

    def feature_importance(self, model):
        date=self.__date_to_split(n_cv_loop=1)
        train_data, cv_data, train_response, cv_response= \
                self.__split_data_into_train_cv(date)
        model.fit(train_data, train_response)
        for (column,imp) in sorted(zip(list(train_data.columns.values), model.feature_importances_)
                          ,key=lambda x: x[1], reverse=True):
              print column, imp

    """
        SEED=25
        def create_cv_randomsubset_from_cvdata(cv_data, cv_number=8):
        X_cv, X1_cv, y_cv, y1_cv = cross_validation.train_test_split(
        cv_data,cv_response, test_size=.50,
        random_state = i*SEED)
        """
    """
    def identify_cat_cont_features(self):
        column_names=list(self._traindata.columns.values)
        columnnames={}
        for e,column_name in enumerate(column_names): columnnames[column_name]=e

        categorical_features= list([columnnames['chain'], columnnames['offer'],
                                   columnnames['market'], columnnames['category'],
                                   columnnames['company'], columnnames['brand']])

        continuous_features= list([columnnames['offervalue'], columnnames['purchasequantity'],
                                   columnnames['productsize_q'],columnnames['purchaseamount_q']])
        return categorical_features, continuous_features
    """

if __name__=="__main__":
       data_dir= 'data/'
       train_file= 'trainHistory.csv'
       test_file= 'testHistory.csv'
       offer_file= 'offers.csv'
       transaction_file= 'transactions.csv'

       trainhist_offers_file= 'trainhist_offers.csv'
       reduced_transaction_file= 'reduced.csv'
       reduced_dated= 'reduced-dated.csv'
       transaction_aggr_file= 'transactions-aggr.csv'
       trainhist_offers_transaction_file= 'trainhist_offers_transaggr_dated.csv'

       testhist_offers_file= 'testhist_offers.csv'
       testhist_offers_transaction_file= 'testhist_offers_transaggr_dated.csv'


       model = linear_model.LogisticRegression() #GaussianNB() svm.SVC()
       date_to_split_begin= '04-01-2013'
       date_to_split_end= '04-20-2013'
       case= None
       add_data=None
       best_c=None
       model_for_feature_imp= AdaBoostClassifier(n_estimators=100)
       #model_for_feature_imp= ExtraTreesClassifier(n_estimators=100)
       if len(sys.argv)!=2:
           print 'python kaggle-shopper.py {wrangling|analyze|both}'
           exit()
       else:
           case=sys.argv[1]

       if case =='wrangling' or case=='both':
          t_i=time.time()
          dw=DataWrangling(data_dir, offer_file, transaction_file)
          #dw.reduce_transaction(outfile=reduced_transaction_file)
          #dw.add_past_purchase_last_n_days(infile=train_file, infile1=reduced_transaction_file,
                                          # outfile=reduced_dated)

          #dw.combine_history_offers(infile=train_file, outfile=trainhist_offers_file)
          #dw.combine_history_offers(infile=test_file, outfile=testhist_offers_file)

          dw.combine_hist_offer_transaggr_v1(infile=reduced_dated, infile1=trainhist_offers_file,
                                    infile2=testhist_offers_file,outfile1=trainhist_offers_transaction_file,
                                    outfile2=testhist_offers_transaction_file)
          #dw.combine_hist_offer_transaggr(infile= transaction_aggr_file, outfile= trainhist_offers_transaction_file)

          #dw.combine_hist_offer_transaggr(infile= transaction_aggr_file, outfile= testhist_offers_transaction_file)
          print ' time takes to wrangle: ', time.time()-t_i

       if case=='analyze' or case=='both':
           t_i=time.time()
           # initialize
           An=AnalyzePredict(data_dir,date_begin=date_to_split_begin, date_end=date_to_split_end)
           # read train
           An.read_aggr_data(file=trainhist_offers_transaction_file, case='train')
           # read test
           An.read_aggr_data(file=testhist_offers_transaction_file, case='test')
           # feature gen. of degree n
           #add_data=An.group_data(degree=2)
           # normalize the data
           data_aggr_scaled = An.normalize_data(add_data, case='train')
           # do a hyperparameter search
           best_c, best_auc= An.hyperparam(model, data=data_aggr_scaled, n_cv_loop=1)
           print 'best C, best AUC: ', best_c, best_auc
           # cv test using the optimized hyperparameter
           mean_auc, std_auc=An.cv_loop(model, bestc=best_c, data= data_aggr_scaled, n_cv_loop=8)
           print 'mean, std AUC: ', mean_auc, std_auc

           # submission on test data
           print "Training full model..."
           totaldata_agr_scaled = An.normalize_data(add_data, case='test')
           preds=An.predict_test(model, totaldata_agr_scaled)
           An.create_test_submission(filename='submit/run5.csv', prediction=preds)
           #An.feature_importance(model=model_for_feature_imp)
           print 'time takes to analyze: ', time.time() - t_i






