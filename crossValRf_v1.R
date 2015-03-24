




#=====================================================================
# Filename:   crossValRf_v1.R
# Objective:  Build and score a randomForest using n-fold 
#             cross-validation
# Date:       2013-08-22
# Version:    v1
# Depends:    randomForest
# Author:     Madhav Kumar
#
#
# Inputs:    01. Training data frame
#            02. Target variable
#            03. Testing data (op) -- if not supplied 
#               20% of training data is randomly
#               selected for testing
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
#            07. Iter (op) - number of trees to build in each fold
#            09. bag.frac (op) - sample proportion for each tree
#            09. Minimumn observations (op) - rf nodesize
#            10  mtry (op) - number of features used for each tree
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
#            04. gbm model, val predictions, test predictions, and 
#                importance for each repition
# 
#=====================================================================


crossValRf <- function(train, train.label, test= NULL, vars= NULL, id= NULL,
                       nfolds= 2, iter= 100,
                       bag.frac= NULL, min.obs.node= NULL, mtry= NULL,
                       model.name= "rf_model", seed= 314159){
  require(randomForest)
  
  # start time counter
  start.time <- Sys.time()
  
  splitData <- function(data, nfolds, seed){
    set.seed(seed)
    rows <- nrow(data)
    # folds
    folds <- rep(1:nfolds, length.out= rows)[sample(rows, rows)]
    lapply(unique(folds), function(x) which(folds == x))
  }
  foldid <- splitData(train, nfolds, seed)
  
  # empty list to store results
  result <- lapply(1:nfolds, function(x) vector('list', 4))
  names(result) <- paste("fold", 1:nfolds, sep= "")
  
  if(is.null(test)){
    d <- sort(sample(nrow(train), nrow(train)*0.2))
    test <- train[d,]
    train <- train[-d,]
  }
  if(is.null(vars)) vars <- names(train)
  if(is.null(id)) {
    id.train <- data.frame(1:nrow(train))
    names(id.train) <- "id"
    id.test <- data.frame(1:nrow(test))
    names(id.test) <- "id"
  } else{
    id.train <- data.frame(train[, id])
    names(id.train) <- id
    id.test <- data.frame(test[, id])
    names(id.test) <- id
  }
  if(is.null(bag.frac)){
    bag.frac <- ceiling(.632*nrow(train))
  } else{
    bag.frac <- ceiling(bag.frac*nrow(train))
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
  
  for (i in 1:nfolds){
    print(paste('Fold', i, 'of', nfolds, sep= " "))
    
    # rows for training and vaidation
    r <- sort(foldid[[i]])
    
    # train model
    set.seed(seed)
    result[[i]][[1]] <- randomForest(train[-r, vars], train.label[-r],
                                     ntree= iter,
                                     sampsize= bag.frac, 
                                     nodesize= min.obs.node,
                                     mtry= mtry,
                                     importance= TRUE, 
                                     do.trace= TRUE)
    
    # score on val
    val.pred <- predict(result[[i]][[1]], train[r, vars], type= "response")
    if (is.factor(train.label)) {
      val.pred <- predict(result[[i]][[1]], train[r, vars], type= "prob")[,2]
    }
    result[[i]][[2]] <- data.frame(id.train[r,], val.pred)
    names(result[[i]][[2]]) <- c(names(id.train), "pred")
    
    # score on test
    test.pred <- predict(result[[i]][[1]], test[, vars], type= "response")
    if (is.factor(train.label)) {
      test.pred <- predict(result[[i]][[1]], test[, vars], type= "prob")[,2]
    }
    result[[i]][[3]] <- data.frame(id.test, test.pred)
    names(result[[i]][[3]]) <- c(names(id.train), "pred")
    
    # important variables
    result[[i]][[4]] <- data.frame(importance(result[[i]][[1]]), 
                                   var= row.names(importance(result[[i]][[1]]))) 
  }
  
  # aggregate outputs
  output <- list()
  
  # validation
  output[[1]]<- do.call(rbind, lapply(1:nfolds, function(x) result[[x]][[2]]))
  names(output[[1]]) <- c(names(id.train), "pred")
  output[[1]] <- output[[1]][order(output[[1]][, names(id.train)]), ]
  
  # test
  test.out <- do.call(cbind, lapply(1:nfolds, 
                                    function(x){
                                      if (x == 1){
                                        result[[x]][[3]]
                                      } else {
                                        result[[x]][[3]][,ncol(result[[x]][[3]])]
                                      }             
                                    }))
  output[[2]] <- data.frame(id.test, pred= rowMeans(test.out[, (dim(id.train)[2] + 1):ncol(test.out)]))
  
  # importance
  imp.out <- do.call(rbind, lapply(1:nfolds, function(x) result[[x]][[4]]))
  imp.out <- aggregate(imp.out[, 1:2], by= list(imp.out$var), mean, na.rm= TRUE)
  names(imp.out) <- c("var", "IncMSE", "IncNodePurity")
  imp.out <- imp.out[order(-imp.out$IncMSE), ]
  output[[3]] <- imp.out
  
  # collate output
  names(output) <- c("val.pred", "test.pred", "importance")
  output <- c(output, result)
  
  # time elapsed
  end.time <- Sys.time()
  print(end.time - start.time)
  output
}
