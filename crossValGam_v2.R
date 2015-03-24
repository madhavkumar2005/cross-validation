




#==============================================================================================
# Filename:   crossValGam_v2.R
# Objective:  1. Build and validate gam using n-fold cross validation
#			  2. Build gam on entire training sample and score on testing
# Date:       2013-09-26
# Version:    v2
# Depends:    gam
# Author:     Madhav Kumar
#
#
# Inputs:    01. Formula
#            02. Training data
#            03. Target variable
#            04. Testing data (op)
#            05. ID (op) unique id variable(s) -- must
#               be present in train and test
#               -- If not supplied, it is internally created
#                  with the original sort of train and test
#            06. Distribution (op) - family
#            07. Model name (op) - currently ineffective
#            08. Seed (op) - seed for reproducibility
#
# op: optional
# 
# Outputs:   00. Ouput is list object
#            01. var.pred - cross validation predictions
#            02. test.pred - predictions on test data
#            03. importance - average importance of variables 
#                accross folds
#			 04. test.model - gam model using entire training data
#            05. gam model, val predictions, and 
#				 importance for each fold
# 
#==============================================================================================


crossValGam <- function(formula, train, train.label, test= NULL, 
                        id= NULL, nfolds= 2, distribution= "gaussian",
                        model.name= "gam_model", seed= 314159){
  require(gam)
  
  # start time counter
  ptm <- proc.time()
  
  if (nfolds <= 1){
    stop("nfolds should be greater than 1")
  }
  if(is.null(id)) {
    id.train <- data.frame(1:nrow(train))
    names(id.train) <- "id"
    if (!is.null(test)){
      id.test <- data.frame(1:nrow(test))
      names(id.test) <- "id" 
    }
  } else{
    id.train <- data.frame(train[, id])
    names(id.train) <- id
    if (!is.null(test)){
      id.test <- data.frame(test[, id])
      names(id.test) <- id  
    }
  }
  
  if (nfolds > 1){
    splitData <- function(data, nfolds, seed){
      set.seed(seed)
      rows <- nrow(data)
      # folds
      folds <- rep(1:nfolds, length.out= rows)[sample(rows, rows)]
      lapply(unique(folds), function(x) which(folds == x))
    }
    foldid <- splitData(train, nfolds, seed) 
    # empty list to store results
    result <- lapply(1:nfolds, function(x) vector('list', 3))
    names(result) <- paste("fold", 1:nfolds, sep= "")
    
    for (i in 1:nfolds){
      print(paste('Fold', i, 'of', nfolds, sep= " "))
      
      # rows for training and vaidation
      r <- sort(foldid[[i]])
      
      # train model
      set.seed(seed)
      result[[i]][[1]] <-  gam(formula, train[-r,], family= distribution, trace= TRUE)
      
      # score on val
      result[[i]][[2]] <- predict(result[[i]][[1]], train[r, ], type= "response")
      result[[i]][[2]] <- data.frame(id.train[r,], train.label[r], result[[i]][[2]])
      names(result[[i]][[2]]) <- c(names(id.train), "pred")
      
      # important variables
      n <- length(row.names(summary(result[[i]][[1]])$parametric.anova))
      result[[i]][[3]] <- data.frame(var= row.names(summary(result[[i]][[1]])$parametric.anova)[1:(n-1)],
                                     imp= summary(result[[i]][[1]])$parametric.anova[1:(n-1),4]) 
    }
  }
  
  # list for aggregating outputs
  output <- list()
  output$val.pred <- NA
  output$test.pred <- NA
  output$importance <- NA
  output$test.model <- NA
  output$vars <- formula
  
  print('Building final model on entire training data')
  output$test.model <- gam(formula, train, family= distribution, trace= TRUE)
  if (!is.null(test)){
    output$test.pred <- predict(output$test.model, test, type= "response")
    output$test.pred <- data.frame(id.test, output$test.pred)
    names(output$test.pred) <- c(names(id.train), "pred") 
  }
  
  # validation
  output$val.pred <- do.call(rbind, lapply(1:nfolds, function(x) result[[x]][[2]]))
  names(output$val.pred) <- c(names(id.train), "target", "pred")
  output$val.pred <- output$val.pred[order(output$val.pred[, names(id.train)]), ]
  
  # importance
  output$importance <- do.call(rbind, lapply(1:nfolds, function(x) result[[x]][[3]]))
  output$importance <- aggregate(output$importance[, "imp"], 
                                 by= list(output$importance$var), mean, na.rm= TRUE)
  names(output$importance) <- c("var", "avg.imp")
  output$importance <- output$importance[order(-output$importance$avg.imp), ]
  
  # collate output
  if(nfolds > 1){
    output <- c(output, result) 
  }
  # time elapsed
  print(proc.time() - ptm)
  output
}
