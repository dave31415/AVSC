import pandas
from readers import data_files,PARS
import numpy
import time

def breakup():
    start=time.time()
    df = pandas.read_csv(data_files('offers'))
    categories = df.category.tolist()
 
    outfile_name=PARS['data_dir']+'subset.csv'
    subset = open(outfile_name, 'w')
    fl = open(data_files('transactions'), 'r')
    fl.readline()
    i=0
    while True:
        i+=1
        if i % 100000 ==0: 
            tdiff_min=(time.time()-start)/60.0
            rate=i/tdiff_min/1e6
            print 'line: %s'%i
            print 'time: %0.2f minutes'%tdiff_min
            print 'rate: %0.2f million lines/minute'%rate
        l = fl.readline()
        if l == '':
            break
        if numpy.int64(l.split(',')[3]) in categories:
            subset.write(l)
    fin=time.time()
    time_diff_min=(fin-start)/60.0
    print 'done'
    print 'runtime: %0.2f minutes'%time_diff_min