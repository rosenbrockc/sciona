# Native reference-only randomForest Gini importance for interaction selection.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==8)
suppressPackageStartupMessages(library(randomForest,lib.loc=args[1]))
x <- as.matrix(read.csv(args[2],header=FALSE,check.names=FALSE))
y <- factor(scan(args[3],quiet=TRUE),levels=0:8)
set.seed(as.integer(args[5]))
model <- randomForest(x=x,y=y,ntree=as.integer(args[6]),
                      mtry=as.integer(args[7]),nodesize=as.integer(args[8]))
scores <- as.numeric(importance(model,type=2))
stopifnot(length(scores)==ncol(x),all(is.finite(scores)),all(scores>=0))
write.table(scores,args[4],row.names=FALSE,col.names=FALSE)
