setwd('~/kaggle')
#import trainHistory
trainHistory=as.data.frame(read.csv('trainHistory',header=T,sep=','))

#import offer file
offerData=as.data.frame(read.csv('offers',header=T,sep','))

##inner merge on offer

joinedData<-merge(trainHistory,offerData,by="offer")

## Transform the categorical variables
joinedData_t=transform(joinedData,id=as.character(id),chain=as.factor(chain),offer=as.factor(offer),market=as.factor(market),category=as.factor(category),company=as.factor(company),brand=as.factor(brand))

##Remove quantity
joinedData_t<-joinedData_t[,-joinedData_t$quantity]

##create the contigency tables...: Not needed since chisq.test inherently is making it so we can use directly the test.


##chi-sq tests
##This surely will be written as a function..
## Offer-Repeater
off_rep<-chisq.test(joinedData_t$offer,joinedData_t$repeater,simulate.p.value=TRUE)
##Chain-Repeater
chain_rep<-chisq.test(joinedData_t$chain,joinedData_t$repeater,simulate.p.value=TRUE)
##market-Repeater
market_rep<-chisq.test(joinedData_t$market,joinedData_t$repeater,simulate.p.value=TRUE)
##repeattrips
reptrips_rep<-chisq.test(joinedData_t$repeattrips,joinedData_t$repeater,simulate.p.value=TRUE)
##offerdate
offD_rep<-chisq.test(joinedData_t$offerdate,joinedData_t$repeater,simulate.p.value=TRUE)
##category-Repeater
cat_rep<-chisq.test(joinedData_t$category,joinedData_t$repeater,simulate.p.value=TRUE)
##Company-Repeater
comp_rep<-chisq.test(joinedData_t$company,joinedData_t$repeater,simulate.p.value=TRUE)
##offervalue-Repeater
offV_rep<-chisq.test(joinedData_t$offervalue,joinedData_t$repeater,simulate.p.value=TRUE)
##Brand-rep
brand_rep<-chisq.test(joinedData_t$brand,joinedData_t$repeater,simulate.p.value=TRUE)

##make a table of chi-squre values:
chisq.table<-rbind(off_rep$statistic,chain_rep$statistic,brand_rep$statistic,cat_rep$statistic,offD_rep$statistic,market_rep$statistic,comp_rep$statistic,offV_rep$statistic,reptrips_rep$statistic)
##add p values to the chisq.table

##name the rows
rownames(chisq.table)<-c("offer","chain","brand","category","offer Date","market","company","offer Value","repeat trips")

##plot residulas within each to see the maximum contributions to the chi-squared

par(mfrow=c(3,3))
barplot(off_rep$residuals[,2],names.arg=rownames(off_rep$residuals),main="Offers")
barplot(chain_rep$residuals[,2],names.arg=rownames(chain_rep$residuals),main="Chain")
barplot(brand_rep$residuals[,2],names.arg=rownames(brand_rep$residuals),main="Brand")
barplot(cat_rep$residuals[,2],names.arg=rownames(cat_rep$residuals),main="Category")
barplot(offD_rep$residuals[,2],names.arg=rownames(offD_rep$residuals),main="Offer Date")
barplot(market_rep$residuals[,2],names.arg=rownames(market_rep$residuals),main="Market")
barplot(comp_rep$residuals[,2],names.arg=rownames(comp_rep$residuals),main="Company")
barplot(offV_rep$residuals[,2],names.arg=rownames(offV_rep$residuals),main="Offer Value")
barplot(reptrips_rep$residuals[,2],names.arg=rownames(reptrips_rep$residuals),main="Repeat trips")
title("feature Residuals from chi-square test", outer=TRUE)






#barplot(chisq.test(joinedData_t$market,joinedData_t$repeater,simulate.p.value=TRUE)$residuals[,2],names.arg=rownames(chisq.test(joinedData_t$market,joinedData_t$repeater,simulate.p.value=TRUE)$residuals))
