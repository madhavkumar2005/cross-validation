




#=====================================================================
# Filename:   crossValGam_v1.R
# Objective:  Build and score a gam using n-fold cross-validation
# Date:       2013-09-10
# Version:    v1
# Depends:    gam
# Author:     Madhav Kumar
#
#
# Inputs:    01. Formula
#            02. Training data
#            03. Testing data (op) -- if not supplied 
#               20% of training data is randomly
#               selected for testing
#            04. ID (op) unique id variable(s) -- must
#               be present in train and test
#               -- If not supplied, it is internally created
#                  with the original sort of train and test
#            05. Distribution (op) - family
#            06. Model name (op) - currently ineffective
#            07. Seed (op) - seed for reproducibility
#
# op: optional
# 
# Outputs:   00. Ouput is list object
#            01. var.pred - cross validation predictions
#            02. test.pred - predictions on test data
#            03. importance - average importance of variables 
#                accross folds
#            04. gam model, val predictions, test predictions, and 
#                importance for each repition
# 
#=====================================================================


crossValGam <- function(formula, train, test= NULL, id= NULL,
                        nfolds= 2, distribution= "gaussian",
                        model.name= "gam_model", seed= 314159){
  require(gam)
  
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
  
  for (i in 1:nfolds){
    print(paste('Fold', i, 'of', nfolds, sep= " "))
    
    # rows for training and vaidation
    r <- sort(foldid[[i]])
    
    # train model
    set.seed(seed)
    result[[i]][[1]] <- gam(formula, train[-r,], family= distribution, trace= TRUE)
    
    # score on val
    val.pred <- predict(result[[i]][[1]], train[r, ], type= "response")
    result[[i]][[2]] <- data.frame(id.train[r,], val.pred)
    names(result[[i]][[2]]) <- c(names(id.train), "pred")
    
    # score on test
    test.pred <- predict(result[[i]][[1]], test, type= "response")
    result[[i]][[3]] <- data.frame(id.test, test.pred)
    names(result[[i]][[3]]) <- c(names(id.train), "pred")
    
    # important variables
    n <- length(row.names(summary(result[[i]][[1]])$parametric.anova))
    result[[i]][[4]] <- data.frame(var= row.names(summary(result[[i]][[1]])$parametric.anova)[1:(n-1)],
                                   imp= summary(result[[i]][[1]])$parametric.anova[1:(n-1),4]) 
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
  imp.out <- aggregate(imp.out[, "imp"], by= list(imp.out$var), mean, na.rm= TRUE)
  names(imp.out) <- c("var", "avg.imp")
  imp.out <- imp.out[order(-imp.out$avg.imp), ]
  output[[3]] <- imp.out
  
  # collate output
  names(output) <- c("val.pred", "test.pred", "importance")
  output <- c(output, result)
  
  # time elapsed
  end.time <- Sys.time()
  print(end.time - start.time)
  output
}
