# Private runtime transport for nine independent RSofia logistic models.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==8)
.libPaths(c(args[1],.libPaths()))
suppressPackageStartupMessages(library(RSofia))
x <- as.matrix(read.csv(args[2],header=FALSE,check.names=FALSE))
y <- scan(args[3],quiet=TRUE)
q <- as.matrix(read.csv(args[4],header=FALSE,check.names=FALSE))
scores <- matrix(NA_real_,nrow(q),9)
for (label in 0:8) {
  model <- sofia.fit(x,as.numeric(ifelse(y==label,1,-1)),
                    random_seed=as.numeric(args[6]),lambda=as.numeric(args[7]),
                    iterations=as.numeric(args[8]),learner_type='logreg-pegasos',
                    eta_type='pegasos',loop_type='balanced-stochastic',
                    no_bias_term=FALSE,dimensionality=ncol(x)+1)
  # Native exp(margin)/(1+exp(margin)) overflows for large margins.
  margin <- suppressWarnings(predict.sofia(model,q,prediction_type='linear'))
  stopifnot(all(is.finite(margin)))
  scores[,label+1] <- plogis(margin)
}
stopifnot(all(is.finite(scores)),all(scores>=0),all(scores<=1))
write.table(scores,args[5],sep=',',row.names=FALSE,col.names=FALSE)
