#average price per item
library(data.table)
library(ggplot2)

read<-function(tmax=5) {
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

run.kmeans<-function(data,k=15){
  ylim=c(15,65)
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
  par(mfrow=c(1,2))
  barplot(stats$frac.repeat*100.0,ylab="% Repeater",xlab="k means cluster #")
  barplot(stats.random$frac.repeat*100.0,ylab="% Repeater",xlab="randomized k means cluster #")
  #p<-ggplot(stats,aes(frac.repeat))+geom_bar(stat="identity")
  #print(p)
  return(stats)
}
