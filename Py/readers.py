#readers for avsc

import os, csv, gzip, json
import pandas as pd

#TODO: remove hardcoded paths, put in parameter file
data_dir="/Users/davej/data/AVSC/"

#Read json data from Parameter file, for now, just need data_dir
JSONDC=json.JSONDecoder()
PARS=JSONDC.decode(open('../PARS.txt','rU').read())

def data_files(name=None):
    ''' returns a dictionary with full path to data files. Or if name is given, just
        return that one'''
    data_dir=PARS['data_dir']
    files= {"offers": data_dir+"offers.csv",
            "history": data_dir+"trainHistory.csv",
            "history_test": data_dir+"testHistory.csv",
            "sample_sub": data_dir+"sampleSubmission.csv",
            #"transactions_gzip": data_dir+"transactions.csv",
            "leaderboard":data_dir+"leaderboard_May12.csv"
            }	
    for key, file in files.iteritems(): assert(os.path.exists(file))
    if name : return files[name]
    return files  

def read_files_pandas(name):
    '''read data into pandas data frame'''
    filename=data_files(name)
    return pd.read_csv(filename)

def stream_data(name):
    '''Return generator/stream to data as dictionaries'''
    #TODO: handle gzip 
    filename=data_files(name)
    return csv.DictReader(open(filename,'rU'))

