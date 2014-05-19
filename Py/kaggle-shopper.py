import pandas as pd
import time
import os
import numpy as np
import sys
from collections import defaultdict
import csv


class DataWrangling:

   def __init__(self, data_dir, train_file, test_file, offer_file, transaction_file):
        self._data_dir= data_dir
        self._train_file= train_file
        self._test_file= test_file
        self._offer_file= offer_file
        self._transaction_file=transaction_file
        self._joined_data= None

   def __get_filepath(self,file):
       return self._data_dir + file

   def combine_trainhistory_offers(self, outfile=''):
       # this code combines trainHistory+offers on offer
       outfile=self.__get_filepath(outfile)
       file_to_open=data_dir + self._train_file
       train_history=pd.read_csv(file_to_open)
       #print train_history[:1]

       file_to_open=data_dir + self._offer_file
       offers=pd.read_csv(file_to_open)

       self._joined_data= pd.merge(train_history, offers, left_on='offer', right_on='offer', how='inner')
       #print joined_data.shape
       #print joined_data[:1]
       print self._joined_data[:1]
       self._joined_data.to_csv(outfile)

   def reduce_transaction(self, outfile= ''):
       #this code reads in transactions data, print out the rows that has matching category with offers.csv
       outfile=self.__get_filepath(outfile)
       offers_cat = {}
       t_i=time.time()
       for e, line in enumerate( open(self._data_dir + self._offer_file) ):
           offers_cat[ line.split(",")[1] ] = 1
       print offers_cat
       t_i=time.time()
       with open(outfile, "wb") as outfile:
             [outfile.write(line) for e,line in enumerate(open(self._data_dir + self._transaction_file))
              if line.split(",")[3] in offers_cat]
       print 'time takes ',time.time()-t_i

   def aggregate_transaction(self, infile='', outfile=''):
       #this code read reduced.csv (obtained from transactions file by keeping the
       # rows whose category exists in offers.csv), create dictionaries which sum over
       # productsize, purchasequantity and purchaseamount for each [customer,category] pair
       # then writes the above dictionaries key, value pair to csv file
       infile=self.__get_filepath(infile)
       outfile=self.__get_filepath(outfile)

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
       infile= self.__get_filepath(infile)
       outfile= self.__get_filepath(outfile)
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

   dw=DataWrangling(data_dir, train_file, test_file, offer_file, transaction_file)
   dw.combine_trainhistory_offers(outfile=hist_offers_file)
   dw.reduce_transaction(outfile=reduced_transaction_file)
   dw.aggregate_transaction(infile=reduced_transaction_file, outfile=transaction_aggr_file)
   dw.combine_hist_offer_transaggr(infile= transaction_aggr_file, outfile= hist_offers_transaction_file)








