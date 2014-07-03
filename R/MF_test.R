#average price per item
library(data.table)
library(ggplot2)
library(caTools)

read<-function(tmax=5,train=T) {
  file="/Users/davej/data/AVSC/trainHistory_with_MF_features.csv"
  data=data.table(read.csv(file,as.is=T))
  data[,Repeat.Trips:=ifelse(repeattrips<tmax,repeattrips,tmax)]
  return(data)
  
  dat=data[,list(N=.N,score.mean=mean(score,na.rm=T,trim=0.01),),by=repeattrips]
  dat=dat[N>10,]
  dat[,trips:=repeattrips+1]
  dat[,scaled.trips:=log(1.0+trips)/log(2)]
  dat[,score.mean.err:=score.sigma/sqrt(N)]
  return(data)
}

doplot<-function(data){
  p<-ggplot(data,aes(scaled.trips,score.mean))+geom_point(size=3)
  p=p+geom_errorbar(aes(ymin=score.mean-score.mean.err,ymax=score.mean+score.mean.err))
  #p=p+geom_errorbar(aes(ymin=score.mean-score.sigma,ymax=score.mean+score.sigma))
  print(p)
}

doplot2<-function(data){
  p<-ggplot(data,aes(score,fill=factor(Repeat.Trips)))+geom_density(alpha=0.1)
  p=p+xlab("MF Score")+xlim(-0.2,0.8)
  print(p)
}

prep<-function(data){
  d.yes=data[repeater=='t',]
  d.no=data[repeater=='f',]
  d.yes=d.yes[sample(nrow(d.yes)),]
  d.no=d.no[sample(nrow(d.no)),]
  row.max=min(nrow(d.yes),nrow(d.no))
  d.yes=d.yes[1:row.max,]
  d.no=d.no[1:row.max,]
  d=rbind(d.yes,d.no)
  d=d[sample(nrow(d)),]
  return(d)
}

run.kmeans<-function(data,k=100){
  ylim=c(0,70)
  d=data[,list(MF0,MF1,MF2,MF3,MF4,MF5,MF6,MF7,MF8,MF9)]
  km=kmeans(d,k,iter.max=50,nstart=1)
  print(nrow(d))
  print(length(km$cluster))
  data[,cluster:=km$cluster]
  data[,random.cluster:=sample(cluster)]
  stats=data[,list(N=.N,frac.repeat=mean(repeater=='t')),by=cluster]
  stats.random=data[,list(N=.N,frac.repeat=mean(repeater=='t')),by=random.cluster]
  stats=stats[order(frac.repeat)]
  stats[,frac.err:=1.0/sqrt(N*frac.repeat)]
  stats.random=stats.random[order(frac.repeat)]

  par(mfrow=c(2,2))
  barplot(stats$frac.repeat*100.0,ylab="% Repeater",xlab="k means cluster #",ylim=ylim)
  barplot(stats.random$frac.repeat*100.0,ylab="% Repeater",xlab="randomized k means cluster #",ylim=ylim)
 
  setkey(stats,cluster)
  setkey(data,cluster)
  data=stats[data]
  roc(data$frac.repeat,data$repeater=='t')
  stats.random$cluster=stats.random$random.cluster
  setkey(stats.random,cluster)
  data.random=stats.random[data]
  roc(data.random$frac.repeat,data.random$repeater=='t',tit='(random)')
  return(list(data=data,km=km))
}

roc<-function(prob,actual,tit=''){
  # ROC curve is x=False Positive rate, y= True Positive rate
  num=50
  prob.min=min(prob)
  prob.max=max(prob)
  ord=rev(order(prob))
  x=prob[ord]
  a=actual[ord]
  not.a=!a
  a.cum=cumsum(a)
  not.a.cum=cumsum(not.a)
  a.tot=sum(a)
  not.a.tot=sum(not.a)
  
  TP=a.cum
  FP=not.a.cum
  
  #these not needed, I guess
  TN=not.a.tot-not.a.cum
  FN=a.tot-a.cum
  
  TPR=TP/a.tot
  FPR=FP/not.a.tot
 
  TPR=c(0,TPR,1)
  FPR=c(0,FPR,1)
  
  auroc=trapz(FPR,TPR)
  title=paste("AUROC: ",strtrim(auroc,6),tit)
  print(title)
  plot(FPR,TPR,type='l')
  title(main=title)
  #lines(FPR,TPR)
  lines(c(0,1),c(0,1),col="red",lty=2)
}

score.kmeans<-function(k=60){
  data=read()
  dk=run.kmeans(data,k=k)
  data=dk$data
  km=dk$km
  clust=predict(km,data)
}


