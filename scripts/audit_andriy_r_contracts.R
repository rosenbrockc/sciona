# Synthetic API/label audit only; not a full source-model validation.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==1)
suppressPackageStartupMessages(library(xgboost))
suppressPackageStartupMessages(library(LiblineaR))
suppressPackageStartupMessages(library(glmnet))
suppressPackageStartupMessages(library(jsonlite))
set.seed(661)
x <- matrix(rnorm(300*8),ncol=8)
y <- as.numeric(x[,1]+.2*x[,2]>0)
d <- xgb.DMatrix(x,label=y)
model <- xgb.train(params=list(objective='binary:logistic',max_depth=3,eta=.3,nthread=1),data=d,nrounds=30,verbose=0)
importance <- xgb.importance(model=model)
source_indices <- suppressWarnings(as.numeric(importance$Feature))
source_selection <- tryCatch({selected <- x[,source_indices,drop=FALSE]; list(status='accepted',all_missing=all(is.na(selected)),columns=ncol(selected))},error=function(e) list(status='error',message=conditionMessage(e)))
named_importance <- xgb.importance(model=model,feature_names=as.character(seq_len(ncol(x))))
stopifnot(all(as.integer(named_importance$Feature)>=1),all(as.integer(named_importance$Feature)<=ncol(x)))

# Source training stacks positive examples before negative examples. Check both
# encounter orders to distinguish class labels from incidental output position.
label_cases <- list()
for (positive_first in c(TRUE,FALSE)) {
  order <- order(y,decreasing=positive_first)
  train <- scale(x[order,])
  target <- as.factor(y[order])
  for (kind in c(5,6)) {
    set.seed(662)
    fitted <- LiblineaR(data=train,target=target,type=kind,cost=if(kind==5) 100 else 10,bias=TRUE,verbose=FALSE)
    predicted <- predict(fitted,train,decisionValues=TRUE,proba=(kind==6))
    decision <- predicted$decisionValues[,1]
    probabilities <- if(kind==6) predicted$probabilities else NULL
    stopifnot(if(positive_first) mean(decision[y[order]==1])>mean(decision[y[order]==0]) else mean(decision[y[order]==1])<mean(decision[y[order]==0]))
    label_cases[[length(label_cases)+1]] <- list(type=kind,positive_first=positive_first,
      model_classes=as.character(fitted$ClassNames),decision_columns=colnames(predicted$decisionValues),
      probability_columns=if(kind==6) colnames(probabilities) else NULL,
      positive_mean_decision=mean(decision[y[order]==1]),negative_mean_decision=mean(decision[y[order]==0]),
      source_column_positive_mean=if(kind==6) mean(probabilities[y[order]==1,1]) else mean(plogis(decision[y[order]==1])),
      source_column_negative_mean=if(kind==6) mean(probabilities[y[order]==0,1]) else mean(plogis(decision[y[order]==0])))
  }
}

ridge <- glmnet(x=x,y=y,alpha=0,family='binomial')
ridge_prediction <- as.numeric(predict(ridge,x,s=.0001,type='response'))
stopifnot(all(is.finite(ridge_prediction)),all(ridge_prediction>=0),all(ridge_prediction<=1))
ridge_minimum_prediction <- as.numeric(predict(ridge,x,s=min(ridge$lambda),type='response'))
ridge_clamp_error <- max(abs(ridge_prediction-ridge_minimum_prediction))
stopifnot(ridge_clamp_error==0)
report <- list(r_version=R.version.string,
  package_versions=lapply(c('xgboost','glmnet','LiblineaR','R.matlab','Matrix'),function(p) list(package=p,version=as.character(packageVersion(p)))),
  unnamed_importance_features=as.character(importance$Feature),source_numeric_indices=source_indices,
  source_selection_result=source_selection,explicit_one_based_importance_features=as.character(named_importance$Feature),
  label_cases=label_cases,ridge_lambda_range=range(ridge$lambda),ridge_requested_lambda=.0001,
  ridge_probability_range=range(ridge_prediction),ridge_minimum_path_clamp_error=ridge_clamp_error,
  limitations=c('Synthetic runtime API audit; does not execute complete source model rounds or all-population aggregation.',
                'Current R package behavior only; historical package equivalence remains unproven.'))
write_json(report,args[1],pretty=TRUE,auto_unbox=TRUE,na='null',digits=16)
cat('R API and label audit completed\n')
