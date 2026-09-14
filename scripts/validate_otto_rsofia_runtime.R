# Synthetic native runtime validation only; not a published Otto model.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==1)
.libPaths(c(args[1], .libPaths()))
suppressPackageStartupMessages(library(RSofia))
y <- rep(0:8, 30)
x <- diag(9)[y+1,,drop=FALSE]
center <- colMeans(x)
spread <- apply(x, 2, sd)
x <- sweep(sweep(x, 2, center, '-'), 2, spread, '/')
q <- sweep(sweep(diag(9), 2, center, '-'), 2, spread, '/')
fit <- function(label, loop) {
  sofia.fit(x, as.numeric(ifelse(y==label, 1, -1)),
            random_seed=12, lambda=.01, iterations=10000,
            learner_type='logreg-pegasos', eta_type='pegasos',
            loop_type=loop, rank_step_probability=.5,
            no_bias_term=FALSE, dimensionality=ncol(x)+1)
}
for (loop in c('balanced-stochastic', 'combined-roc')) {
  scores <- matrix(NA_real_,9,9)
  for (label in 0:8) {
    model <- fit(label,loop)
    scores[,label+1] <- predict.sofia(model,q,prediction_type='logistic')
    single <- predict.sofia(model,q[1,,drop=FALSE],prediction_type='logistic')
    stopifnot(isTRUE(all.equal(single[1],scores[1,label+1],tolerance=0)))
    repeated <- fit(label,loop)
    stopifnot(identical(model$weights,repeated$weights))
  }
  stopifnot(all(is.finite(scores)),all(scores>=0),all(scores<=1),
            identical(as.integer(max.col(scores)-1L),0:8))
}
cat('PASS: two sampling loops, 36 binary fits, nine-class learning, repeatability and query batch invariance\n')
