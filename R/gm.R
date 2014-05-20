#AVSC
library(data.table)
library('bit64')
#Bayesian network for kaggle

library(gRbase)
library(bnlearn)
library(igraph)

source('io.R')

get.bayes.net<-function(history){
  dat=history[,list(category,brand,company,repeater,market,offervalue,chain,item)]
  dat$repeater=factor(dat$repeater)
  dat$offervalue=factor(dat$offervalue)
  white_from=c("category","brand")
  white_to=c("item","item")
  whitelist=data.frame(from=white_from,to=white_to)
  net=hc(dat,whitelist=whitelist)
  plot(as(amat(net),"graphNEL"))
}


