######
# Paalkhile Kamaraj
# Saruk Shaikh


# XGBoost

# Importing the dataset
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[4:14]
str(dataset)
# Encoding the categorical variables as factors
dataset$Geography = as.numeric(factor(dataset$Geography,
                                      levels = c('France', 'Spain', 'Germany'),
                                      labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                                   levels = c('Female', 'Male'),
                                   labels = c(1, 2)))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

## XGBOOST
library(xgboost)
list<-list("eta" = 0.01,
               "max_depth" = 8,
               "gamma" = 0.1,
               "subsample" = 0.5,
               "colsample_bytree" = 1,
               "seed" = 333,
               "objective" ="multi:softmax",
               "eval_metric" = "mlogloss",
               "num_class"= 2)

classifier = xgboost(data = as.matrix(training_set[,-11]), 
                     label = training_set$Exited, 
                     missing = NA,
                     nrounds =500,
                     params = list
                     )


# Predicting the Test set results
y_pred = predict(classifier, newdata = as.matrix(test_set[,-11]))
#y_pred = (y_pred >= 0.5)
predictions <-data.frame(test_set$Exited,y_pred)
View(predictions)

# Making the Confusion Matrix
cm = table(test_set[, 11], y_pred)
cm
accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
accuracy


# Applying k-Fold Cross Validation
# install.packages('caret')
library(caret)
folds = createFolds(training_set$Exited, k = 10)

cv = lapply(folds, function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier = xgboost(data = as.matrix(training_set[-11]), label = training_set$Exited,params = list, nrounds = 500)
  y_pred = predict(classifier, newdata = as.matrix(test_fold[-11]))
  y_pred = (y_pred >= 0.5)
  cm = table(test_fold[, 11], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})
accuracy = mean(as.numeric(cv))
accuracy
