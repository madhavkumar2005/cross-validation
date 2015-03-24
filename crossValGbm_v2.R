




#==============================================================================================
# Filename:   crossValGbm_v2.R
# Objective:  1. Build and validate gbm using n-fold cross validation
#			  2. Build gbm on entire training sample and score on testing
# Date:       2013-09-26
# Version:    v2
# Depends:    gbm (2.1)
# Author:     Madhav Kumar
#
#
# Inputs:    01. Training data frame
#            02. Target variable
#            03. Testing data (op)
#            04. Features (op) -- can be either a
#               character vector of names or 
#               integer vector of positions
#               -- If not supplied, all variables 
#                 present in train are considered
#            05. ID (op) unique id variable(s) -- must
#               be present in train and test
#               -- If not supplied, it is internally created
#                  with the original sort of train and test
#            06. No. of cross-validation folds
#            07. Iter (op) - number of trees to build in each repition
#            08. Distribution (op) - gbm loss function
#            09. Shrinkage (op) - gbm shrinkage
#            10. Depth (op) - gbm interaction depth
#            11. bag.frac (op) - gbm bag fraction
#            12. Minimumn observations (op) - gbm n.minobsinnode
#            13. Model name (op) - currently ineffective
#            14. Seed (op) - seed for reproducibility
#
# op: optional
# 
# Outputs:   00. Ouput is list object
#            01. var.pred - cross validation predictions
#            02. test.pred - predictions on test data
#            03. importance - average importance of variables 
#                accross folds
#			 04. test.model - gbm model using entire training data
#            05. gbm model, val predictions, and 
#				 importance for each fold
# 
#==============================================================================================


crossValGbm <- function(train, train.label, test= NULL, vars= NULL, id= NULL,
                        nfolds= 2, iter= 500,
                        distribution= "gaussian", shrinkage= 0.05, depth= 2,
                        bag.frac= 0.5, min.obs.node= 10, 
                        model.name= "gbm_model", seed= 314159){
  require(gbm)
  
  # start time counter
  ptm <- proc.time()
  
  if (nfolds <= 1){
    stop("nfolds should be greater than 1")
  }
  if(is.null(vars)) vars <- names(train)
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
      result[[i]][[1]] <- gbm.fit(train[-r, vars], train.label[-r],
                                  distribution= distribution, n.trees= iter,
                                  shrinkage= shrinkage, 
                                  bag.fraction= bag.frac,
                                  interaction.depth= depth, 
                                  n.minobsinnode= min.obs.node, 
                                  keep.data= FALSE)
      
      # score on val
      result[[i]][[2]] <- predict(result[[i]][[1]], train[r, vars], n.trees= iter, type= "response")
      result[[i]][[2]] <- data.frame(id.train[r,], train.label[r], result[[i]][[2]])
      names(result[[i]][[2]]) <- c(names(id.train), "target", "pred")
      
      # important variables
      result[[i]][[3]] <- data.frame(summary(result[[i]][[1]], plotit= FALSE)) 
    }
  }
  
  # list for aggregating outputs
  output <- list()
  output$val.pred <- NA
  output$test.pred <- NA
  output$importance <- NA
  output$test.model <- NA
  output$vars <- vars
  
  print('Building final model on entire training data')
  output$test.model <- gbm.fit(train[, vars], train.label,
                               distribution= distribution, n.trees= iter,
                               shrinkage= shrinkage, 
                               bag.fraction= bag.frac,
                               interaction.depth= depth, 
                               n.minobsinnode= min.obs.node, 
                               keep.data= FALSE)
  if (!is.null(test)){
    output$test.pred <- predict(output$test.model, test[, vars], n.trees= iter, type= "response")
    output$test.pred <- data.frame(id.test, output$test.pred)
    names(output$test.pred) <- c(names(id.train), "pred") 
  } 
  
  # validation
  output$val.pred <- do.call(rbind, lapply(1:nfolds, function(x) result[[x]][[2]]))
  names(output$val.pred) <- c(names(id.train), "target","pred")
  output$val.pred <- output$val.pred[order(output$val.pred[, names(id.train)]), ]
  
  # importance
  output$importance <- do.call(rbind, lapply(1:nfolds, function(x) result[[x]][[3]]))
  output$importance <- aggregate(output$importance[, "rel.inf"], 
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