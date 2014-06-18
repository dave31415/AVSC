from operator import add
from pyspark import SparkConf, SparkContext
from collections import defaultdict

conf = (SparkConf()
         .setMaster("local")
         .setAppName("My app")
         .set("spark.executor.memory", "8g")
         .set("spark.storage.memoryFraction","0.3")
)
sc = SparkContext(conf = conf)
"""
def cust_dept():
     key_val=entries_by_line.map(lambda x:(x[0],x[2]))
     ct_keyval=key_val.map(lambda x:(x,1)).reduceByKey(add)
     for k,v in ct_keyval.collect():
         cust_dept[k]+=v
     return cust_dept
"""
directory = "/mnt/data/reducedpart/" #transactions-parts/"  # Should be some file on your system
filepart=directory+'reduced_part_'
total_files=175

cust_dept, cust_cat, cust_comp, cust_brand=defaultdict(int),defaultdict(int),defaultdict(int),defaultdict(int)

for i in xrange(4):
     counter='%d'%(i)
     infile=filepart +str(counter) #'trainHistory.csv'
     print infile
     logData = sc.textFile(infile)
     #line= logData.flatMap(lambda x:x.split('\n'))
     entries_by_line= logData.map(lambda x:x.split(','))  #.take(2000000)

     key_val=entries_by_line.map(lambda x:(x[0],x[2]))
     ct_keyval=key_val.map(lambda x:(x,1)).reduceByKey(add)
     for k,v in ct_keyval.collect():
         cust_dept[k]+=v

     key_val=entries_by_line.map(lambda x:(x[0],x[3]))
     ct_keyval=key_val.map(lambda x:(x,1)).reduceByKey(add)
     for k,v in ct_keyval.collect():
         cust_cat[k]+=v

     key_val=entries_by_line.map(lambda x:(x[0],x[4]))
     ct_keyval=key_val.map(lambda x:(x,1)).reduceByKey(add)
     for k,v in ct_keyval.collect():
         cust_comp[k]+=v

     key_val=entries_by_line.map(lambda x:(x[0],x[5]))
     ct_keyval=key_val.map(lambda x:(x,1)).reduceByKey(add)
     for k,v in ct_keyval.collect():
         cust_brand[k]+=v     
     #logData.unpersist()
     #print cust_dept.take(10)
     #if j==0: ct_distinct[j-2]-=1
print len(cust_dept.keys()),len(cust_cat.keys()),len(cust_comp.keys()),len(cust_brand.keys())
