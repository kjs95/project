rm(list=ls())
setwd("D:/4학년 1학기/데이터과학/Term")

rawdata = read.csv("Dataset.csv")

colnames(rawdata) = c("item" , "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23")

for(i in 1:nrow(rawdata)){
  if(rawdata[i,1] == 0 ){
    rawdata[i,1] = "Snack"
  }else if(rawdata[i,1] == 1){
    rawdata[i,1] = "Beer"
  }else if(rawdata[i,1] == 2){
    rawdata[i,1] = "Soju"
  }else if(rawdata[i,1] == 3){
    rawdata[i,1] = "ramen"
  }else if(rawdata[i,1] == 4){
    rawdata[i,1] = "IceCream"
  }else
    rawdata[i,1] = "MilkProduct"
}

for (i in 1:nrow(rawdata)){
  sum = 0
  for(j in 2:ncol(rawdata)){
    sum = sum + rawdata[i,j]
  }
  if(sum == 0){
    rawdata = rawdata[-i,]
  }
}

for(i in 2:ncol(rawdata)){
  rawdata[,i] = (rawdata[,i] - min(rawdata[,i])) / (max(rawdata[,i]) - min(rawdata[,i]))
}

#prdata = cbind(rawdata, new_item)
prdata = rawdata

#tmp_idx = union(which(prdata$new_item == 2), which(prdata$new_item == 3))
#prdata = prdata[tmp_idx,]
#prdata$new_item[which(prdata$new_item == 3)] =0
#prdata$new_item = as.factor(prdata$new_item)

trn_ratio = 0.7
trn_idx = sample(1:nrow(prdata) , round(trn_ratio*nrow(prdata)))
tst_idx = setdiff(1:nrow(prdata) , trn_idx)

trn_data = prdata[trn_idx,]
tst_data = prdata[tst_idx,]

library(nnet)

model_nn = nnet(class.ind(item) ~ ., data = trn_data , size = 30 ,linout = TRUE , maxit = 1000)
out_nn = predict(model_nn , tst_data)

library(devtools)
source_url("https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r")
library(reshape2)
plot.nnet(model_nn)


target = tst_data[,ncol(tst_data)]
outs = cbind(target,out_nn)
##for(i in 1:nrow(outs)){
#  temp = round(outs[i,2],digits = 1)
#  if(temp < 0.1){
#    outs[i,2] = round(0.0,digits = 1)
#  }else if ( 0.1 < temp && temp < 0.3){
#    outs[i,2] = round(0.2,digits = 1)
#  }else if ( 0.3 < temp && temp < 0.5){
#    outs[i,2] = round(0.4,digits = 1)
#  }else if ( 0.5 < temp && temp < 0.7){
#    outs[i,2] = round(0.6,digits = 1)
#  }else if ( 0.7 < temp && temp < 0.9){
#    outs[i,2] = round(0.8,digits = 1)
#  }else
#    outs[i,2] = round(1.0,digits = 1)
##}
for(i in 1:nrow(outs)){
  for(j in 2:ncol(outs)){
    temp = round(outs[i,j], digits = 1)
    if(temp < 0.1){
      outs[i,j] = round(0.0,digits = 1)
    }else if ( 0.1 < temp && temp < 0.3){
      outs[i,j] = round(0.2,digits = 1)
    }else if ( 0.3 < temp && temp < 0.5){
      outs[i,j] = round(0.4,digits = 1)
    }else if ( 0.5 < temp && temp < 0.7){
      outs[i,j] = round(0.6,digits = 1)
    }else if ( 0.7 < temp && temp < 0.9){
      outs[i,j] = round(0.8,digits = 1)
    }else
      outs[i,j] = round(1.0,digits = 1) 
  }
}

library(caret)
confusionMatrix(factor(outs[,2]), factor(outs[,1]))
confusionMatrix(factor(outs[,3]), factor(outs[,1]))
confusionMatrix(factor(outs[,4]), factor(outs[,1]))
confusionMatrix(factor(outs[,5]), factor(outs[,1]))
confusionMatrix(factor(outs[,6]), factor(outs[,1]))
confusionMatrix(factor(outs[,7]), factor(outs[,1]))
