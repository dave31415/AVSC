#AVSC
library(data.table)
library(bit64)

#TODO: remove hardcode path
data.dir="/Users/davej/data/AVSC/"
#data.dir="/home/ubuntu/data/"

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
  
  data[,group:=as.numeric(id) %% 100]
  add.item(data)
  return(data)
}

add.item<-function(data){
   data[,item.family:=paste(category,brand,company ,chain,sep='_')]
  
   data[,id.item.family:=paste(id,item.family,sep='_')]
  
   if ('productmeasure' %in% names(data) & 'productsize' %in% names(data)) {

     data[,item:=paste(category,brand,company,chain,productsize,productmeasure,sep='_')]
  
     data[,id.item:=paste(id,item,sep='_')]
    }
}

first <-function(x) x[1]

fast.agg<-function(){
   start=Sys.time()
   trans.file=paste(data.dir,'reduced.csv',sep='')
   print('reading history')
   hist=read.history()
   #assert that customers only appear once
   stopifnot(nrow(hist) == length(unique(hist$id)))

   print("reading transactions")
   trans=fread(trans.file)
   trans=trans[purchasequantity > 1 & purchaseamount > 0.01,]
   trans[,price:=purchaseamount/purchasequantity]

   print(Sys.time()-start)

   print("adding item")
   add.item(trans)

   print("aggregating by id")
   
   trans.agg.by.id=trans[,list( 
   	N.tot.by.id=sum(purchasequantity),Spend.tot.by.id=sum(purchaseamount),
	N.unique.by.id=length(!duplicated(item)) 
	),by=id]
   alpha.N=10.0
   #TODO: pick a good prior
   prior.diversity=0.5
   trans.agg.by.id[,diversity.by.id:=(N.unique.by.id+alpha.N)/(N.tot.by.id + alpha.N/prior.diversity)]

   print("aggregating by item")
     
   trans.agg.by.item=trans[,list( 
   	N.all.purchases.by.item=sum(purchasequantity),Spend.all.by.item=sum(purchaseamount),
	N.unique.by.item=length(!duplicated(id)),N.tot.by.item=.N,
  Mean.price.item.by.item=mean(price,trim=0.05,na.rm=T),Med.price.by.item=median(price))
,by=item]  

   alpha.id=100.0
   #TODO: pick a good prior
   prior.diversity=0.1
   trans.agg.by.item[,diversity.by.item:=(N.unique.by.item+alpha.id)/(N.tot.by.item + alpha.N/prior.diversity)]

   print("aggregating by id.item")
   trans.agg.by.id.item=trans[,list (Mean.price.item.by.id.item=mean(price,trim=0.05,na.rm=T),
                                     id=first(id),item=first(item),id.item.family=first(id.item.family))
                             ,by=id.item]

   print("joining aggregates")

   setkey(trans.agg.by.id.item,item)
   setkey(trans.agg.by.item,item)
   trans.agg.by.id.item=trans.agg.by.item[trans.agg.by.id.item]

   setkey(trans.agg.by.id.item,id)
   setkey(trans.agg.by.id,id)
   trans.agg.by.id.item=trans.agg.by.id[trans.agg.by.id.item]

   print("Calculating Discounts")

   trans.agg.by.id.item[,Price.Discount:=Mean.price.item.by.item-Mean.price.item.by.id.item]
   trans.agg.by.id.item[,Price.Discount.Percent:= 100.0*Price.Discount/Mean.price.item.by.item]

   print("Aggregating id.item down to less granular id.item.family")
   #add other stuff ???
   trans.agg.by.id.item.family = trans.agg.by.id.item[,list(
     Mean.Discount.Percent=mean(Price.Discount.Percent,na.rm=T),
     id=first(id)
     ),by=id.item.family]    

   print("joining Mean.Discount.Percent to training")
  
   agg=trans.agg.by.id.item.family[,list(Mean.Discount.Percent.By.Customer=mean(Mean.Discount.Percent)),by=id]

   setkey(hist,id)
   setkey(agg,id)  
   hist=agg[hist]

   print(Sys.time()-start)
   print("done")
   p<-ggplot(hist,aes(Mean.Discount.Percent.By.Customer,fill=repeater))+geom_density(alpha=0.2)
   p+xlim(-20,20)

   return(hist)
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

roc<-function(prob,actual){
  prob.min=min(prob)
  prob.max=max(prob)
  
}



   