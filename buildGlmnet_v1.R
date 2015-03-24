




#==========================================================================================
# Filename:   buildGlmnet_v1.R
# Objective:  1. Build regularized regression on subsample of training and score on validation
#			  2. Build regularized regression on complete training and score on testing
# Objective:  Build and score a regularized regression model 
# Date:       2013-09-17
# Version:    v1
# Depends:    glmnet
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
#            09. Iter (op) - number of cv folds in cv.glmnet
#            10. distribution (op) - family for glm
#            11. Error measure (op)
#            12  Observation weights (op)
#            13. alpha (op) - regularization norm
#            14. Minimum lambda ratio (op)
#            15. model name - currently ineffective
#            16. Seed (op) - seed for reproducibility
#
# op: optional
# 
# Output:    Output is a list object with basic function call information and 
#            01. model(s) - model built for validation and testing
#            02. pred(x) - predictions on validation and test data
#            03. importance - variable importance from validation and test data
# 
#==========================================================================================


buildGlmnet <- function(train, train.label, vars= NULL,
                        mode= "both",
                        val.id= NULL, val.value= NULL, 
                        test= NULL, id= NULL,
                        iter= 5, distribution= "gaussian", measure= "mse", 
                        w= NULL, alpha= 0.5, lambda.min.ratio= 0.0001,
                        model.name= "enet_model", seed= 314159){
  require(glmnet)
  
  # start time counter
  ptm <- proc.time()
  
  # sanity checks
  if (mode == "both" & (is.null(val.id) || is.null(val.value) || is.null (test))){
    stop("If mode is set to both then val.id, val.value, and test data need to be supplied")
  } else if (mode == "val" & (is.null(val.id) || is.null(val.value))){
    stop("If mode is set to val then val.id and val.value need to be supplied")
  } else if (mode == "test" & (is.null(test))){
  } else if (mode == "val" & (is.null(test))){
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
  
  # model specific
  if (is.null(w)){
    w <- rep(1, nrow(train))
  }
  if (lambda.min.ratio == 0.0001){
    lambda.min.ratio <- ifelse(nrow(train) < length(vars), 0.01, 0.0001) 
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
    output$val.model <- cv.glmnet(data.matrix(train[-r, vars]), 
                                  train.label[-r], 
                                  family= distribution, alpha= alpha,
                                  weights= w[-r],
                                  nfolds= iter, type.measure= measure,
                                  lambda.min.ratio= lambda.min.ratio)
    output$val.pred <- predict(output$val.model, 
                               data.matrix(train[r, vars]), 
                               type= 'response', 
                               s= 'lambda.min')[,1]
    output$val.pred <- data.frame(id.train[r,], train.label[r], output$val.pred)
    names(output$val.pred) <- c(names(id.train), "target", "pred")
    output$val.imp <- names(which(
      output$val.model$glmnet.fit$beta[ ,which(output$val.model$lambda == 
                                                 output$val.model$lambda.min)] != 0))
    output$val.imp <- data.frame(var = output$val.imp)
    print("Trained and scored for validation")
  }
  
  if (mode %in% c("test", "both")){
    output$test.model <- cv.glmnet(data.matrix(train[, vars]), 
                                   train.label, 
                                   family= distribution, alpha= alpha,
                                   weights= w,
                                   nfolds= iter, type.measure= measure,
                                   lambda.min.ratio= lambda.min.ratio)
    output$test.pred <- predict(output$test.model, 
                                data.matrix(test[,vars]), 
                                type= 'response', 
                                s= 'lambda.min')[,1]
    output$test.pred <- data.frame(id.test, output$test.pred)
    names(output$test.pred) <- c(names(id.test), "pred")
    output$test.imp <- names(which(
      output$test.model$glmnet.fit$beta[ ,which(output$test.model$lambda == 
                                                  output$test.model$lambda.min)] != 0))
    output$test.imp <- data.frame(var = output$test.imp)
    print("Trained and scored for testing")
  }  
  print(proc.time() - ptm)
  output
}
