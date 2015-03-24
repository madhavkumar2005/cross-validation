




#==============================================================================================
# Filename:   crossValRf_v2.R
# Objective:  1. Build and validate randomized trees using n-fold cross validation
#			  2. Build randomized trees on entire training sample and score on testing
# Date:       2013-09-26
# Version:    v2
# Depends:    randomForest
# Author:     Madhav Kumar
#
#
# Inputs:    01. Training data frame
#            02. Target variable
#            03. Testing data (op)
#            04. Features (op) -- can be either a
#                character vector of names or 
#                integer vector of positions
#                -- If not supplied, all variables 
#                 present in train are considered
#            05. ID (op) unique id variable(s) -- must
#                be present in train and test
#                -- If not supplied, it is internally created
#                with the original sort of train and test
#            06. No. of cross-validation folds (op)
#            07. Iter (op) - number of trees to build in each fold
#            08. bag.frac (op) - sample proportion for each tree
#            09. Minimumn observations (op) - rf nodesize
#            10. mtry (op) - number of features used for each tree
#            11. Model name (op) - currently ineffective
#            12. Seed (op) - seed for reproducibility
#
# op: optional
# 
# Outputs:   00. Ouput is list object
#            01. var.pred - cross validation predictions
#            02. test.pred - predictions on test data
#            03. importance - average importance of variables 
#                accross folds
#			 04. test.model - rf model using entire training data
#            05. rf model, val predictions, and 
#				 importance for each fold
# 
#==============================================================================================


crossValRf <- function(train, train.label, test= NULL, vars= NULL, id= NULL,
                       nfolds= 2, iter= 100,
                       bag.frac= NULL, min.obs.node= NULL, mtry= NULL,
                       model.name= "rf_model", seed= 314159){
  require(randomForest)
  
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
    
  if(is.null(min.obs.node)){
    if(is.factor(train.label)){
      min.obs.node <- 1
    } else{
      min.obs.node <- 5
    }
  }
  if(is.null(mtry)){
    if(is.factor(train.label)){
      mtry <- floor(sqrt(length(vars)))
    } else{
      mtry <- max(floor(length(vars)/3), 1)
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
      
      if(is.null(bag.frac)){
        bf <- ceiling(.632*nrow(train[-r,]))
      } else{
        bf <- ceiling(bag.frac*nrow(train[-r,]))
      }
      # train model
      set.seed(seed)
      result[[i]][[1]] <- randomForest(train[-r, vars], train.label[-r],
                                       ntree= iter,
                                       sampsize= bf, 
                                       nodesize= min.obs.node,
                                       mtry= mtry,
                                       importance= TRUE, 
                                       do.trace= TRUE)
      
      # score on val
      if (is.factor(train.label)) {
        result[[i]][[2]] <- predict(result[[i]][[1]], train[r, vars], type= "prob")[,2]
      } else{
        result[[i]][[2]] <- predict(result[[i]][[1]], train[r, vars], type= "response") 
      }
      result[[i]][[2]] <- data.frame(id.train[r,], train.label[r], result[[i]][[2]])
      names(result[[i]][[2]]) <- c(names(id.train), "target", "pred")
      
      # important variables
      result[[i]][[3]] <- data.frame(importance(result[[i]][[1]]), 
                                     var= row.names(importance(result[[i]][[1]]))) 
    }
  }
  
  # aggregate outputs
  output <- list()
  output$val.pred <- NA
  output$test.pred <- NA
  output$importance <- NA
  output$test.model <- NA
  output$vars <- vars
  
  print('Building final model on entire training data')
  if(is.null(bag.frac)){
    bf <- ceiling(.632*nrow(train))
  } else{
    bf <- ceiling(bag.frac*nrow(train))
  }
  output$test.model <- randomForest(train[, vars], train.label,
                                    ntree= iter,
                                    sampsize= bf, 
                                    nodesize= min.obs.node,
                                    mtry= mtry,
                                    importance= TRUE, 
                                    do.trace= TRUE)
  if (!is.null(test)){
    if (is.factor(train.label)) {
      output$test.pred <- predict(output$test.model, test[, vars], type= "prob")[,2]
    } else{
      output$test.pred <- predict(output$test.model, test[, vars], type= "response") 
    }
    output$test.pred <- data.frame(id.test, output$test.pred)
    names(output$test.pred) <- c(names(id.train), "pred") 
  }
  
  # validation
  output$val.pred <- do.call(rbind, lapply(1:nfolds, function(x) result[[x]][[2]]))
  names(output$val.pred) <- c(names(id.train), "target","pred")
  output$val.pred <- output$val.pred[order(output$val.pred[, names(id.train)]), ]
  
  # importance
  output$importance <- do.call(rbind, lapply(1:nfolds, function(x) result[[x]][[3]]))
  output$importance <- aggregate(output$importance[, 1:2], 
                                 by= list(output$importance$var), mean, na.rm= TRUE)
  names(output$importance) <- c("var", "IncMSE", "IncNodePurity")
  output$importance <- output$importance[order(-output$importance$IncMSE), ]
  
  # collate output
  if(nfolds > 1){
    output <- c(output, result) 
  }
  # time elapsed
  print(proc.time() - ptm)
  output
}
