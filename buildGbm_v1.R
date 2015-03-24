




#==========================================================================================
# Filename:   buildGbm_v1.R
# Objective:  1. Build gbm on subsample of training and score on validation
#			  2. Build gbm on complete training and score on testing
# Date:       2013-09-26
# Version:    v1
# Depends:    gbm (2.1)
# Author:     Madhav Kumar
#
#
# Inputs:    01. Training data frame
#            02. Target variable
#            03. Features (op) -- can be either a
#                character vector of names or 
#                integer vector of positions
#                 -- If not supplied, all variables 
#                 present in train are considered
#            04. mode (op) -- can be either:
#                 a. "val": build model on subset of train
#                           and score on validation data
#                 b. "test": build model on complete train
#                           and score on testing data
#                 c. "both": do both a. and b.
#            05. Column in training to identify train and val split
#            06. Value that identifies validation rows 
#            07. Testing data -- required is mode is "test" or "both"
#            08. ID (op) unique id variable(s) -- must
#               be present in train and test
#               -- If not supplied, it is internally created
#                  with the original sort of train and test
#            09. Iter (op) - number of trees
#            10. distribution (op) - loss function for gbm
#            11. Shrinkage (op) - gbm shrinkage
#            12. Depth (op) - gbm interaction depth
#            13. bag.frac (op) - gbm bag fraction
#            14. Minimumn observations (op) - gbm n.minobsinnode
#            15. Model name (op) - currently ineffective
#            16. Seed (op) - seed for reproducibility
#
# op: optional
# 
# Output:    Output is a list object with basic function call information and 
#            01. model(s) - model built for validation and testing
#            02. pred(s) - predictions on validation and test data
#            03. importance - variable importance from validation and test data
# 
#==========================================================================================

buildGbm <- function(train, train.label, vars= NULL,
                     mode= "both",
                     val.id= NULL, val.value= NULL, 
                     test= NULL, id= NULL,
                     iter= 5, distribution= "gaussian", measure= "mse", 
                     w= NULL, alpha= 0.5, lambda.min.ratio= 0.0001,
                     model.name= "enet_model", seed= 314159){
  require(gbm)
  
  # start time counter
  ptm <- proc.time()
  
  # sanity checks
  if (mode == "both" & (is.null(val.id) || is.null(val.value) || is.null (test))){
    stop("If mode is set to both then val.id, val.value, and test data need to be supplied")
  } else if (mode == "val" & (is.null(val.id) || is.null(val.value))){
    stop("If mode is set to val then val.id and val.value need to be supplied")
  } else if (mode == "test" & (is.null(test))){
    stop("If mode is set to test then test data need to be supplied")
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
  
  # list for storing results
  output <- list()
  output$vars <- vars
  output$mode <- mode
  output$val.model <- NA
  output$val.pred <- NA
  output$val.imp <- NA
  output$test.model <- NA
  output$test.pred <- NA
  output$test.imp <- NA
  
  if (mode %in% c("val", "both")){
    r <- sort(which(train[,val.id] == val.value))
    output$val.model <- gbm.fit(train[-r, vars], train.label[-r],
                                distribution= distribution, n.trees= iter,
                                shrinkage= shrinkage, 
                                bag.fraction= bag.frac,
                                interaction.depth= depth, 
                                n.minobsinnode= min.obs.node, 
                                keep.data= FALSE)
    output$val.pred <- predict(output$val.model, train[r, vars], n.trees= iter, type= "response")
    output$val.pred <- data.frame(id.train[r,], train.label[r], output$val.pred)
    names(output$val.pred) <- c(names(id.train), "target", "pred")
    output$val.imp <- data.frame(summary(output$val.model, plotit= FALSE)) 
    print("Trained and scored for validation")
  }
  
  if (mode %in% c("test", "both")){
    output$test.model <- gbm.fit(train[, vars], train.label,
                                 distribution= distribution, n.trees= iter,
                                 shrinkage= shrinkage, 
                                 bag.fraction= bag.frac,
                                 interaction.depth= depth, 
                                 n.minobsinnode= min.obs.node, 
                                 keep.data= FALSE)
    output$test.pred <- predict(output$test.model, test[, vars], n.trees= iter, type= "response")
    output$test.pred <- data.frame(id.test, output$test.pred)
    names(output$test.pred) <- c(names(id.test), "pred")
    output$test.imp <- data.frame(summary(output$test.model, plotit= FALSE)) 
    print("Trained and scored for testing")
  }  
  print(proc.time() - ptm)
  output
}