from operator import add
from pyspark import SparkConf, SparkContext
from collections import defaultdict
import csv
import sys
import numpy
from datetime import datetime, timedelta, date

conf = (SparkConf()
         .setMaster("local")
         .setAppName("My app")
         .set("spark.executor.memory", "40g")
         .set("spark.storage.memoryFraction","0.1")
)
sc = SparkContext(conf = conf)

def sumThis(entries_by_line,out_dict,*col_no):
     key_val=entries_by_line.map(lambda x:(x[col_no[0]],x[col_no[1]]))
     ct_keyval=key_val.map(lambda x:(x,1)).reduceByKey(add)
     for k,v in ct_keyval.collect():
         out_dict[k]+=v

def compBrCat(entries_by_line, out_dict, *col_no):
     key_val=entries_by_line.map(lambda x:(x[col_no[0]],x[col_no[1]],x[col_no[2]],x[col_no[3]]))
     ct_keyval=key_val.map(lambda x:(x,1)).reduceByKey(add)
     for k,v in ct_keyval.collect():
         out_dict[k]+=v

def compCat(entries_by_line, out_dict, *col_no):
     key_val=entries_by_line.map(lambda x:(x[col_no[0]],x[col_no[1]],x[col_no[2]]))
     ct_keyval=key_val.map(lambda x:(x,1)).reduceByKey(add)
     for k,v in ct_keyval.collect():
         out_dict[k]+=v
     
def purchaseQuantity(entries_by_line,out_dict,*col_no):
     key_val_2=entries_by_line.map(lambda x:((x[col_no[0]],x[col_no[1]],x[col_no[2]],x[col_no[3]]),int(x[col_no[4]])))
     ct_keyval_2=key_val_2.reduceByKey(lambda x,y:x+y)
     for k,v in ct_keyval_2.collect():
         out_dict[k]+=v

def appendThis(entries_by_line, out_dict,*col_no):
    ct_keyval=entries_by_line.map(lambda x:((x[col_no[0]],x[col_no[1]]),x[col_no[2]]))
    for k,v in ct_keyval.collect():
         out_dict[k].append(v)

def purchaseAmount(entries_by_line,out_dict,*col_no):
     key_val=entries_by_line.map(lambda x:((x[col_no[0]],x[col_no[1]]),float(x[col_no[2]])))
     ct_keyval=key_val.reduceByKey(lambda x,y:x+y)
     for k,v in ct_keyval.collect():
         out_dict[k]+=v
    
def writeOutfile(outfile,data,*head):
     with open(outfile,'w') as f:
        write=csv.writer(f,delimiter=',')
        write.writerow(list((head)))
        for keys,value in data.iteritems():
            row=["".join(key) for key in list(keys)]
            if not isinstance(value,tuple): row.append(str(value))
            else: row.extend([str(v) for v in value])
            write.writerow(row)

def getFilepath(directory,filename):
       return directory + filename 

def diffMonths(s1):
   delta=[]
   date_format = "%Y-%m-%d"
   b = datetime.strptime("2014-01-01",date_format)
   for a in s1:
        a1=datetime.strptime(a, date_format)
        delta.append(int((b-a1).days)/30.0)
   return delta   

case=5

directory = "/mnt/data/transactions-parts/"  # Should be some file on your system
filepart=directory+'transactions_part_'
total_files=175

outdir="/mnt/data/transactions-aggr/"
file_cust_dept,file_cust_cat,file_cust_comp,file_cust_brand,file_avg_basket_size="sum_dept.csv","sum_cat.csv","sum_comp.csv","sum_brand.csv","avg_basket_size.csv"
file_comp_br_cat,file_comp_cat="count_trans_comp_br_cat.csv","count_trans_comp_cat.csv"
file_monthly_avg_pa="monthly_avg_pa.csv"
file_sum_pq="sum_pq_comp_br_cat.csv"

cust_dept, cust_cat, cust_comp, cust_brand=defaultdict(int),defaultdict(int),defaultdict(int),defaultdict(int)
visits, cust_trans,avg_basket_size=defaultdict(list),defaultdict(int), defaultdict(float)
cust_trans_cat_comp_br,cust_trans_cat_comp,cust_sum_pq_cat_comp_br=defaultdict(int),defaultdict(int),defaultdict(int)
#cat_comp_br=defaultdict(int)
monthly_avg_pa,sum_pa,sum_pq=defaultdict(float), defaultdict(float),defaultdict(int)

for i in xrange(total_files):
     counter='%003d'%(i)
     infile=filepart +str(counter) 
     print infile
     logData = sc.textFile(infile)
     entries_by_line= logData.map(lambda x:x.split(','))
     if case==0: sumThis(entries_by_line,cust_dept,0,2)
     if case==1: sumThis(entries_by_line,cust_cat,0,3)
     if case==2: sumThis(entries_by_line,cust_comp,0,4)
     if case==3: sumThis(entries_by_line,cust_brand,0,5)
     if case==4:
           sumThis(entries_by_line,cust_trans,0,1) 
           appendThis(entries_by_line,visits,0,1,6)
     if case==5: compBrCat(entries_by_line,cust_trans_cat_comp_br,0,3,4,5)
     if case==6: 
         purchaseAmount(entries_by_line,sum_pa,0,1,10)
         appendThis(entries_by_line, visits,0,1,6)
     if case==7:
         purchaseQuantity(entries_by_line,sum_pq,0,3,4,5,9)
     if case==8:
         compCat(entries_by_line,cust_trans_cat_comp,0,3,4)
if case==4:
   for key in cust_trans.keys():
       count_visits=len(set(visits[key]))
       avg_basket_size[key]=(cust_trans[key]/(float(count_visits)+1),count_visits)
#if case==5:
#   for key in cust_trans_cat_comp_br.keys():
#       cust_trans_notthisbrand=cust_trans_cat_comp[(key[0],key[1],key[2])] - cust_trans_cat_comp_br[key]   
#       cat_comp_br[key]=(cust_trans_cat_comp_br[key],cust_trans_notthisbrand)
if case==6:
     for key in visits.keys():
            list_months=diffMonths(list(set(visits[key])))
            max_min= max(list_months)-min(list_months)  
            #print sum_pa[key],sum_pa[key]/(max_min+1),max_min
            monthly_avg_pa[key]=(sum_pa[key]/(max_min+1),max_min)

if case==0: writeOutfile(getFilepath(outdir,file_cust_dept),cust_dept,"id","dept","sum_dept")
if case==1: writeOutfile(getFilepath(outdir,file_cust_cat),cust_cat,"id","category","sum_category")
if case==2: writeOutfile(getFilepath(outdir,file_cust_comp),cust_comp,"id","company","sum_company")
if case==3: writeOutfile(getFilepath(outdir,file_cust_brand),cust_brand,"id","brand","sum_brand")
if case==4:
   writeOutfile(getFilepath(outdir,file_avg_basket_size),avg_basket_size,"id","chain","avg_basket_size","visits")
if case==5:
   writeOutfile(getFilepath(outdir,file_comp_br_cat),cust_trans_cat_comp_br,"id","category","company","brand","count_trans")
if case==6:
   writeOutfile(getFilepath(outdir,file_monthly_avg_pa),monthly_avg_pa,"id","chain","monthly_avg_pa","max_min_months")
if case==7:
   writeOutfile(getFilepath(outdir,file_sum_pq),sum_pq,"id","category","company","brand","sum_pq")
if case==8:
  writeOutfile(getFilepath(outdir,file_comp_cat),cust_trans_cat_comp,"id","category","company","count_trans")
print len(cust_dept.keys()),len(cust_cat.keys()),len(cust_comp.keys()),len(cust_brand.keys()), len(avg_basket_size.keys()),len(sum_pa.keys())
