# Native R randomForest fit, using private runtime CSV transport only.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==9)
suppressPackageStartupMessages(library(randomForest, lib.loc=args[1]))
x <- as.matrix(read.csv(args[2], header=FALSE, check.names=FALSE))
y <- factor(scan(args[3], quiet=TRUE), levels=0:8)
q <- as.matrix(read.csv(args[4], header=FALSE, check.names=FALSE))
set.seed(as.integer(args[6]))
model <- randomForest(x=x, y=y, ntree=as.integer(args[7]),
                      mtry=as.integer(args[8]), nodesize=as.integer(args[9]),
                      replace=TRUE, keep.forest=TRUE)
p <- predict(model, q, type="prob")
stopifnot(identical(colnames(p), as.character(0:8)), all(is.finite(p)))
write.table(p, args[5], sep=",", row.names=FALSE, col.names=FALSE)
