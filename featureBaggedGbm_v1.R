




#=====================================================================
# Filename:   featureBaggedGbm_v1.R
# Objective:  Build and score feature bagged stochastic
#             gradient boosting models
# Date:       2013-08-21
# Version:    v1
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
                             tprop= 0.2, fprop= 0.2, reps= 2, iter= 500,
                             distribution= "gaussian", shrinkage= 0.05, depth= 2,
                             bag.frac= 0.5, min.obs.node= 10, 
                             model.name= "gbm_model", seed= 314159){
  require(gbm)
  
  # start time counter
  start.time <- Sys.time()
  
  # empty list to store results
  result <- lapply(1:reps, function(x) vector('list', 4))
  names(result) <- paste("rep", 1:reps, sep= "")
  
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
  
  for (i in 1:reps){
    print(paste('Rep', i, 'of', reps, sep= " "))
    
    # sample rows based on tprop
    set.seed(i*37)
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
    
    # score on test
    test.pred <- predict(result[[i]][[1]], test[, f], n.trees= iter, type= "response")
    result[[i]][[3]] <- data.frame(id.test, test.pred)
    names(result[[i]][[3]]) <- c(names(id.train), "pred")
    
    # important variables
    result[[i]][[4]] <- data.frame(summary(result[[i]][[1]], plotit= FALSE)) 
  }
  
  # aggregate outputs
  output <- list()
  
  # validation
  val.out <- do.call(rbind, lapply(1:reps, function(x) result[[x]][[2]]))
  output[[1]] <- aggregate(val.out$pred, 
                           by= list(val.out[, -ncol(val.out)]),
                           mean, na.rm= TRUE)
  names(output[[1]]) <- c(names(id.train), "pred")
  output[[1]] <- output[[1]][order(output[[1]][, names(id.train)]), ]
  
  # test
  test.out <- do.call(cbind, lapply(1:reps, 
                                          function(x){
                                            if (x == 1){
                                              result[[x]][[3]]
                                            } else {
                                              result[[x]][[3]][,ncol(result[[x]][[3]])]
                                            }             
                                          }))
  output[[2]] <- data.frame(id.test, pred= rowMeans(test.out[, (dim(id.train)[2] + 1):ncol(test.out)]))
  
  # importance
  imp.out <- do.call(rbind, lapply(1:reps, function(x) result[[x]][[4]]))
  imp.out <- aggregate(imp.out[, "rel.inf"], by= list(imp.out$var), function(x) c(mean(x), length(x)))
  imp.out <- do.call("data.frame", imp.out)
  names(imp.out) <- c("var", "avg.imp", "times.select")
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
