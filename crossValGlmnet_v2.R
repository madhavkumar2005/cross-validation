




#==============================================================================================
# Filename:   crossValGlmnet_v2.R
# Objective:  1. Build and validate regularized regression using 
#				 n-fold cross validation
#			  2. Build regularized regression on entire training sample
#				 and score on testing sample
# Date:       2013-09-26
# Version:    v2
# Depends:    glmnet
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
#            06. No. of cross-validation folds (op)
#  		 	 07. iter (op): No. of folds for cv.glmnet
#            08. distribution (op) - family for glm
#            09. Error measure (op)
#            10.  Observation weights (op)
#            11. alpha (op) - regularization norm
#            12. model name - currently ineffective
#            13. Seed (op) - seed for reproducibility
#
# op: optional
# 
# Outputs:   00. Ouput is list object
#            01. var.pred - cross validation predictions
#            02. test.pred - predictions on test data
#			 04. test.model - glmnet model using entire training data
#            05. glmnet model, val predictions, and 
#				 importance for each fold
# 
#==============================================================================================


crossValGlmnet <- function(train, train.label, test= NULL, vars= NULL, id= NULL,
                           nfolds= 2, iter= 5, 
                           distribution= "gaussian", measure= "mse", 
                           w= NULL, alpha= 0.5, lambda.min.ratio= 0.0001,
                           model.name= "enet_model", seed= 314159){
  require(glmnet)
  
  # # start time counter
  ptm <- proc.time()
  
  if (nfolds < 1){
    stop("nfolds should be greater than 0")
  }
  if(is.null(vars)) vars <- colnames(train)
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
  
  
  if (is.null(w)){
    w <- rep(1, nrow(train))
  }
  if (lambda.min.ratio == 0.0001){
    lambda.min.ratio <- ifelse(nrow(train) < length(vars), 0.01, 0.0001) 
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
      result[[i]][[1]] <- cv.glmnet(data.matrix(train[-r, vars]), 
                                    train.label[-r], 
                                    family= distribution, alpha= alpha,
                                    weights= w[-r],
                                    nfolds= iter, type.measure= measure,
                                    lambda.min.ratio= lambda.min.ratio)
      
      # score on val
      result[[i]][[2]] <- predict(result[[i]][[1]], data.matrix(train[r, vars]), type= 'response', 
                                  s= 'lambda.min')[,1]
      result[[i]][[2]] <- as.numeric(result[[i]][[2]])
      result[[i]][[2]] <- data.frame(id.train[r,], train.label[r], result[[i]][[2]])
      names(result[[i]][[2]]) <- c(names(id.train), "target", "pred")
      
      # important variables
      result[[i]][[3]] <- names(which(
        result[[i]][[1]]$glmnet.fit$beta[ ,which(result[[i]][[1]]$lambda == 
                                                   result[[i]][[1]]$lambda.min)] != 0))  
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
  output$test.model <- cv.glmnet(data.matrix(train[, vars]), 
                                 train.label, 
                                 family= distribution, alpha= alpha,
                                 weights= w,
                                 nfolds= iter, type.measure= measure,
                                 lambda.min.ratio= lambda.min.ratio)
  if (!is.null(test)){
    output$test.pred <- predict(output$test.model, data.matrix(test[, vars]), type= 'response', 
                                s= 'lambda.min')[,1]
    output$test.pred <- as.numeric(output$test.pred)
    output$test.pred <- data.frame(id.test, output$test.pred)
    names(output$test.pred) <- c(names(id.train), "pred") 
  }
  
  # validation
  output$val.pred <- do.call(rbind, lapply(1:nfolds, function(x) result[[x]][[2]]))
  names(output$val.pred) <- c(names(id.train), "target", "pred")
  output$val.pred <- output$val.pred[order(output$val.pred[, names(id.train)]), ]
  
  # importance
  output$importance <- do.call(rbind, lapply(1:nfolds, function(x){
    data.frame(result[[x]][[3]])
  }))
  output$importance <- aggregate(output$importance[,1], 
                                 by= list(output$importance[,1]), length)
  names(output$importance) <- c("var", "times.select")
  output$importance <- output$importance[order(-output$importance$times.select), ]
  
  # collate output
  if(nfolds > 1){
    output <- c(output, result) 
  }
  # time elapsed
  print(proc.time() - ptm)
  output
}
