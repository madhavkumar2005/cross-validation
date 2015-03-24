




#==========================================================================================
# Filename:   buildRf_v1.R
# Objective:  1. Build randomized trees on subsample of training and score on validation
#			  2. Build randomized trees on complete training and score on testing
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
#            09. Iter (op) - number of trees to build in each fold
#            10. bag.frac (op) - sample proportion for each tree
#            11. Minimumn observations (op) - rf nodesize
#            12  mtry (op) - number of features used for each tree
#            13. Model name (op) - currently ineffective
#            14. Seed (op) - seed for reproducibility
#
# op: optional
# 
# Output:    Output is a list object with basic function call information and 
#            01. model(s) - model built for validation and testing
#            02. pred(x) - predictions on validation and test data
#            03. importance - variable importance from validation and test data
# 
#==========================================================================================

buildRf <- function(train, train.label, vars= NULL,
                    mode= "both",
                    val.id= NULL, val.value= NULL, 
                    test= NULL, id= NULL,
                    iter= 100, bag.frac= NULL, min.obs.node= NULL, mtry= NULL,
					model.name= "rf_model", seed= 314159){
  require(randomForest)
  
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
    if(is.null(bag.frac)){
      bf <- ceiling(.632*nrow(train[-r,]))
    } else{
      bf <- ceiling(bag.frac*nrow(train[-r,]))
    }
    output$val.model <- randomForest(train[-r, vars], train.label[-r],
                                     ntree= iter,
                                     sampsize= bf, 
                                     nodesize= min.obs.node,
                                     mtry= mtry,
                                     importance= TRUE, 
                                     do.trace= TRUE)
    if (is.factor(train.label)) {
      output$val.pred <- predict(output$val.model, train[r, vars], type= "prob")[,2]
    } else{
      output$val.pred <- predict(output$val.model, train[r, vars], type= "response") 
    }
    output$val.pred <- data.frame(id.train[r,], train.label[r], output$val.pred)
    names(output$val.pred) <- c(names(id.train), "target", "pred")
    output$val.imp <- data.frame(importance(output$val.model), 
                                 var= row.names(importance(output$val.model)))
    print("Trained and scored for validation")
  }
  
  if (mode %in% c("test", "both")){
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
    if (is.factor(train.label)) {
      output$test.pred <- predict(output$test.model, test[, vars], type= "prob")[,2]
      } else{
       output$test.pred <- predict(output$test.model, test[, vars], type= "response") 
     }
    output$test.pred <- data.frame(id.test, output$test.pred)
    names(output$test.pred) <- c(names(id.train), "pred")
    output$test.imp <- data.frame(importance(output$test.model), 
                                 var= row.names(importance(output$test.model)))
    print("Trained and scored for testing")
  }  
  print(proc.time() - ptm)
  output
}