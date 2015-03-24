




#=====================================================================
# Filename:   featureBaggedGbm_v2.R
# Objective:  Build and score feature bagged stochastic
#             gradient boosting models
# Date:       2013-10-14
# Version:    v2
# Depends:    gbm (2.1)
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
#            06. tprop (op) - fraction of randomly selected
#               training data to be used in each repition 
#            07. fprop (op) - fraction of randomly selected
#               features to be used in each repition
#            08. Iter (op) - number of trees to build in each repition
#            09. Distribution (op) - gbm loss function
#            10. Shrinkage (op) - gbm shrinkage
#            11. Depth (op) - gbm interaction deptj
#            12. bag.frac (op) - gbm bag fraction
#            13. Minimumn observations (op) - gbm n.minobsinnode
#            14. Model name (op) - currently ineffective
#            15. Seed (op) - seed for reproducibility
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


featureBaggedGbm <- function(train, train.label, test= NULL, vars= NULL, id= NULL, 
                             tprop= 0.2, fprop= 0.2, nfolds= 4, iter= 500,
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
  
  # empty list to store results
  result <- lapply(1:nfolds, function(x) vector('list', 4))
  names(result) <- paste("rep", 1:nfolds, sep= "")
  
  for (i in 1:nfolds){
    print(paste('Fold', i, 'of', nfolds, sep= " "))
    
    # sample rows based on tprop
    set.seed(i*seed)
    r <- sort(sample(nrow(train), nrow(train)*tprop))
    
    # sample features based on fprop
    f <- sample(length(vars), length(vars)*fprop)
    f <- vars[f]
    
    # train model
    set.seed(seed)
    result[[i]][[1]] <- gbm.fit(train[r, f], train.label[r],
                                distribution= distribution, n.trees= iter,
                                shrinkage= shrinkage, 
                                bag.fraction= bag.frac,
                                interaction.depth= depth, 
                                n.minobsinnode= min.obs.node, 
                                keep.data= FALSE)
    
    # score on val
    val.pred <- predict(result[[i]][[1]], train[-r, f], n.trees= iter, type= "response")
    result[[i]][[2]] <- data.frame(id.train[-r,], val.pred)
    names(result[[i]][[2]]) <- c(names(id.train), "pred")
    
    # important variables
    result[[i]][[3]] <- data.frame(summary(result[[i]][[1]], plotit= FALSE))
    
    # vars used
    result[[i]][[4]] <- f
  }
  
  # list for aggregating outputs
  output <- list()
  output$val.pred <- NA
  output$test.pred <- NA
  output$importance <- NA
  output$vars <- vars
  
  # validation
  output$val.pred <- do.call(rbind, lapply(1:nfolds, function(x) result[[x]][[2]]))
  output$val.pred <- aggregate(output$val.pred[,"pred"], 
                               by= list(output$val.pred[, names(id.train)]),
                               mean, na.rm= TRUE)
  names(output$val.pred) <- c(names(id.train), "pred")
  temp <- data.frame(id.train, target= train.label)
  output$val.pred <- merge(temp, output$val.pred, by= names(id.train), all.x= TRUE)
  output$val.pred$pred <- ifelse(is.na(output$val.pred$pred), 
                                 mean(output$val.pred$pred, na.rm= TRUE), 
                                 output$val.pred$pred)
  output$val.pred <- output$val.pred[order(output$val.pred[, names(id.train)]), ]
  
  # score on test
  if(!is.null(test)){
    output$test.pred <- data.frame(row.names= 1:nrow(test))
    for(i in 1:nfolds){
      f <- result[[i]][[4]]
      output$test.pred[,paste("fold", i, sep= ".")] <- 
        predict(result[[i]][[1]], test[, f], n.trees= iter, type= "response")
    }
    output$test.pred <- data.frame(id.test, rowMeans(output$test.pred))
    names(output$test.pred) <- c(names(id.test), "pred")
  }
  
  # importance
  output$importance <- do.call(rbind, lapply(1:nfolds, function(x) result[[x]][[3]]))
  output$importance <- aggregate(output$importance[, "rel.inf"], 
                                 by= list(output$importance$var), 
                                 function(x) c(mean(x, na.rm= TRUE), length(x)))
  output$importance <- do.call("data.frame", output$importance)
  names(output$importance) <- c("var", "avg.imp", "times.select")
  output$importance <- output$importance[order(-output$importance$avg.imp), ]
  
  # collate output
  output <- c(output, result)
  
  # time elapsed
  print(proc.time() - ptm)
  output
}
