#install.packages('ROCR')

library(ROCR)
setwd('/Users/Thoughtworker/Programming/Kaggle/avsc/data')
predictions <- read.csv('predictions.csv', header=T)
labels <- read.csv('labels.csv', header=F)
pred <- prediction(predictions$V2, labels$V2)
perf <- performance(pred, measure = "tpr", x.measure = "fpr") 
plot(perf, col=rainbow(10))
abline(0,1)
performance(pred, measure = "auc")

#predictions$label <- labels$V2
#neg.scores <- predictions[which(as.numeric(predictions$label) == 0),]$V2
#pos.scores <- predictions[which(as.numeric(predictions$label) == 1),]$V2
#mean(replicate(1000,mean(sample(pos.scores,2*length(pos.scores),replace=T) > sample(neg.scores,2*length(pos.scores),replace=T))))



