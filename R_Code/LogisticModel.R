##########################
# Generalized Linear regression function. (Logistic Regression)
##########################
logisticRegressionModel = function(dataSet, targetVariable) {
  glmFit = glm(formula = targetVariable ~ ., data = dataSet, family = binomial)
  return(glmFit)
}

##########################
# VIF
##########################
library(car)

checkVarianceInflationFactor = function(model) {
  vif(model)
}

##########################
# Stepwise regression function.
##########################
stepwiseRegression = function(model) {
  stepwiseModel = stepAIC(model)
  return(stepwiseModel)
}

##########################
# Predict on Train/Test with model. return probabilities
# Get a list of predictions (probability scores) using the predict() function
# Use the argument 'type = "response"' in the predict function to get a list of predictions between 0 and 1
##########################
predictProbability = function(model, dataSet) {
  probPredict = predict(model, dataSet, type = "response")
  return(probPredict)
}

##########################
# Assessing performance with ROC curve
# Using the ROCR package create a "prediction()" object
##########################
library(ROCR)

rocrPerformance = function(trainedModel, testData, targetVariable) {
  
  # 1. Prepare Probability Matrix. requires -> trainedModel, testData without y
  # or get probabilities.
  # 2. ROCR prediction
  # 3. Prepare ROCR performance object for ROC curve (tpr, fpr) ans AUC
  # The performance() function from the ROCR package helps us extract metrics such as True positive rate, False positive rate etc. from the prediction object, we created above.
  # Two measures (y-axis = tpr, x-axis = fpr) are extracted
  # 4. plot ROCR curve.
  
  # 1.
  train.model.predict.probs = predictProbability(trainedModel, testData)
  # 2.
  train.model.probability.rocr = prediction(train.model.predict.probs, testData[targetVariable])
  # 3.
  train.model.performance = performance(train.model.probability.rocr, "tpr", "fpr")
  train.model.auc.perf = performance(train.model.probability.rocr, measure = "auc", x.measure = "cutoff")
  # 4.
  plot(train.model.performance, col = 2, colorize = TRUE, 
       main = paste("AUC", train.model.auc.perf@y.values))
  auc <- train.model.auc.perf@y.values[[1]]
  print(auc)
  return(list(train.rocr.perf = train.model.performance, auc = auc)) 
  #return(train.model.performance)
}

choosingCutOffValues = function(train.rocr.perf) {
  # For different threshold values identifying the tpr and fpr
  cutoffs <- data.frame(cut= train.rocr.perf@alpha.values[[1]], fpr= train.rocr.perf@x.values[[1]], 
                        tpr=train.rocr.perf@y.values[[1]])
  
  # Sorting the data frame in the decreasing order based on tpr
  cutoffs <- cutoffs[order(cutoffs$tpr, decreasing=TRUE),]
  
  # Plotting the true positive rate and false negative rate based based on the cutoff       
  # increasing from 0.1-1
  plot(train.rocr.perf, colorize = TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))
}