#readers for avsc

import os, csv, json
import pandas as pd
import random

#Read json data from Parameter file, for now, just need data_dir
JSONDC=json.JSONDecoder()
PARS=JSONDC.decode(open('../PARS.json','rU').read())

def data_files(name=None):
    ''' returns a dictionary with full path to data files. Or if name is given, just
        return that one'''
    data_dir=PARS['data_dir']
    if data_dir[-1] != '/' : data_dir+='/'
    
    files= {"offers": data_dir+"offers.csv",
            "history": data_dir+"trainHistory.csv",
            "history_test": data_dir+"testHistory.csv",
            "sample_sub": data_dir+"sampleSubmission.csv",
            "transactions": data_dir+"transactions.csv",
            "leaderboard":data_dir+"leaderboard_May12.csv"
            }	
    for key, file in files.iteritems(): assert(os.path.exists(file))
    if name : return files[name]
    return files  

def read_files_pandas(name):
    '''read data into pandas data frame'''
    filename=data_files(name)
    return pd.read_csv(filename)

def stream_data(name,frac=1.0):
    '''Return generator/stream to data as dictionaries
       If frac < 0, it will sample a random fraction of customers 
    '''
    #TODO: handle gzip?
    filename=data_files(name)
    for line in csv.DictReader(open(filename,'rU')):
        if 'id' in line:
            if hash_frac(line['id'],frac=frac) : yield line
        else:
            yield line

def hash_frac(input,frac=1.0):
    '''Useful utility for sampling some fraction of things deterministically'''
    if frac == 1.0 : return True
    if frac == 0.0 : return False
    h=abs(hash(str(input)))
    random.seed(h)
    r=random.random()
    return r < frac

