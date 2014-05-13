#AVSC
library(data.table)

#TODO: remove hardcode path
data.dir="/Users/davej/data/AVSC/"

read.offers<-function(){
  file=paste(data.dir,'offers.csv',sep='')
  return(data.table(read.csv(file,as.is=T)))
}

read.history<-function(){
  file=paste(data.dir,'trainHistory.csv',sep='')
  o=read.offers()
  setkey(o,offer)
  h=data.table(read.csv(file,as.is=T))
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
  
  return(data)
}
