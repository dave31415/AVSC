#AVSC
library(data.table)
library('bit64')

#TODO: remove hardcode path
#data.dir="/Users/davej/data/AVSC/"
data.dir="/home/ubuntu/data/"

read.offers<-function(){
  file=paste(data.dir,'offers.csv',sep='')
  return(fread(file))
}

read.history<-function(){
  file=paste(data.dir,'trainHistory.csv',sep='')
  o=read.offers()
  setkey(o,offer)
  h=fread(file)
  setkey(h,offer)
  #join on offer
  data=o[h]
  data[,repeater:=repeattrips>0]
  data$offer=factor(data$offer)
  data$category=factor(data$category)
  data$company=factor(data$company)
  data$id=factor(data$id)
  data$brand=factor(data$brand)
  data$chain=factor(data$chain)
  data$market=factor(data$market)
  data[,group:=as.integer(id) %% 100]
  add.item(data)
  #data[,item:=factor(paste(as.character(category),as.character(brand),sep='_'))]
  return(data)
}

add.item<-function(data){
  data[,item:=factor(paste(as.character(category),as.character(brand),sep='_'))]
  data[,id.item:=factor(paste(as.character(id),as.character(item),sep='_'))]
}

first<-function(x) x[1]

fast.agg<-function(){
   start=Sys.time()
   trans.file=paste(data.dir,'reduced.csv',sep='')
   print('reading history')
   hist=read.history()
   #assert that customers only appear once
   stopifnot(nrow(hist) == length(unique(hist$id)))
   print("reading transactions")
   trans=fread(trans.file)
   print(Sys.time()-start)
   print("adding item")
   add.item(trans)
   print("aggregating")
   trans.agg=trans[,list(N.item.ids=.N,N.purchases=sum(purchasequantity),Tot.amount=sum(purchaseamount),
	Min.price=min(purchaseamount), Max.price=max(purchaseamount), 
	id=first(id),item=first(item)),
	by=id.item]

   print(Sys.time()-start)
   print("reaggregating")
   #roll up to id
   trans.agg.all=trans.agg[,list(N.unique=.N,N.purchases.all=sum(N.purchases),Tot.amount.all=sum(Tot.amount),
        Min.price.all=min(Min.price), Max.price.all=max(Max.price)),
        by=id]	
   print(Sys.time()-start)
   print("joining")
   setkey(hist,id.item)
   setkey(trans.agg,id.item)
   agg=trans.agg[hist]
   print("joining with agg.all")
   setkey(agg,id)
   setkey(trans.agg.all)
   agg=trans.agg.all[agg]
   print("done")
   print("runtime")
   print(Sys.time()-start)
   #use Laplace smoothing for ratios
   diversity.prior=0.1
   alpha=30.0
   agg[,diversity:=(N.purchases+alpha)/(N.purchases+alpha/diversity.prior)
   return(agg)
}


cust.count<-function(){
   start=Sys.time()
   trans.file=paste(data.dir,'reduced.csv',sep='')
   print("reading transactions")
   trans=fread(trans.file)
   print(Sys.time()-start)
   print("aggregating")
   customer.counts=trans[,.N,by=id]
   print(Sys.time()-start)
   print("writing to a file")
   out.file=paste(data.dir,'customer.counts.csv',sep='')
   write.csv(customer.counts,out.file)
   print("done")
   print(Sys.time()-start)
}

   