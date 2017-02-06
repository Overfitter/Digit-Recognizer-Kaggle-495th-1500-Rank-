setwd("H:/Kaggle/Digit_Recognizer")
test <- read_csv("H:/Kaggle/Digit_Recognizer/test.csv")
train <- read_csv("H:/Kaggle/Digit_Recognizer/train.csv")

#PCA Algorithm
library(pca)
 #load data
label <- train$label
train$label<-NULL
combi <- rbind(train, test)
pca.train <- combi[1:nrow(train),]
pca.test <- combi[-(1:nrow(train)),]
head(train) #show sample data
dim(train) #check dimensions
str(train) #show structure of the data
sum(train) 
colnames(train)
apply(train,2,var) #check the variance accross the variables
pca =prcomp(pca.train) #applying principal component analysis on crimtab data
par(mar = rep(2, 4)) #plot to show variable importance
plot(pca) 
'below code changes the directions of the biplot, if we donot include
the below two lines the plot will be mirror image to the below one.'
pca$rotation=-pca$rotation
pca$x=-pca$x
biplot (pca , scale =0) #plot pca components using biplot in r
train.data <- predict(pca, newdata = pca.train)
test.data <- predict(pca, newdata = pca.test)
train.data <- as.data.frame(train.data)
test.data <- as.data.frame(test.data)
train.data <- train.data[,1:50]
test.data <- test.data[,1:50]


#SVM
library(e1071)
svm_model <- svm(label~.,train.data)
pred_svm <- predict(svm_model,test.data)
predictions <- data.frame(ImageId=1:nrow(test), Label=pred_svm)

write.csv(predictions, "nnet.csv")
##0.9827

# Creates a simple random forest benchmark

library(randomForest)
library(readr)

set.seed(0)

numTrain <- 10000
numTrees <- 25

rows <- sample(1:nrow(train), numTrain)
labels <- as.factor(train[rows,1])
train <- train[rows,-1]
##Random Forest Model
rf <- randomForest(train, labels, xtest=test, ntree=numTrees)
predictions <- data.frame(ImageId=1:nrow(test), Label=levels(labels)[rf$test$predicted])
head(predictions)

write.csv(predictions, "Naive_Bayes.csv")
##KNN Algo
set.seed(107)
inTrain <- createDataPartition(y = train$label, p = .75, list = FALSE)
training <- train[ inTrain,]
testing <- train[-inTrain,]

ytrain = training$label
library(class)
knn_model <- knn(training,testing,ytrain,k=5)
summary(knn_model)
#xgboost
#convert = c(1:784)
#`test`[,convert] = sapply(`test`[,convert],as.numeric)
#dtrain <- xgb.DMatrix(data = data.matrix(train), label = data.matrix(train$label))
#dtest <- xgb.DMatrix(data = data.matrix(test), missing = NA)
PARAM <- list(
  # General Parameters
  booster            = "gbtree",          # default
  silent             = 0,                 # default
  # Booster Parameters
  eta                = 0.05,              # default = 0.30
  gamma              = 0,                 # default
  max_depth          = 5,                 # default = 6
  min_child_weight   = 1,                 # default
  subsample          = 0.70,              # default = 1
  colsample_bytree   = 0.95,              # default = 1
  num_parallel_tree  = 1,                 # default
  lambda             = 0,                 # default
  lambda_bias        = 0,                 # default
  alpha              = 0,                 # default
  # Task Parameters
  objective          = "multi:softmax",   # default = "reg:linear"
  num_class          = 10,                # default = 0
  base_score         = 0.5,               # default
  eval_metric        = "merror"           # default = "rmes"
)
TRAIN_SMM <- sparse.model.matrix(label ~ ., data = train.data)
TRAIN_XGB <- xgb.DMatrix(data = TRAIN_SMM, label = label)
set.seed(1)

# train xgb model
MODEL <- xgb.train(params      = PARAM, 
                   data        = TRAIN_XGB, 
                   nrounds     = 50, # change this to 400
                   verbose     = 2,
                   watchlist   = list(TRAIN_SMM = TRAIN_XGB)
)

# attach a predictions vector to the test dataset
test.data$label <- 0

# use the trained xgb model ("MODEL") on the test data ("TEST") to predict the response variable ("LABEL")
TEST_SMM <- sparse.model.matrix(label ~ ., data = test.data)
PRED <- predict(MODEL, TEST_SMM)

# create submission file
SUBMIT <- data.frame(ImageId = c(1:length(PRED)), Label = PRED)
write.csv(SUBMIT, "xgb.csv")
#GBM
train$label<- as.factor(train$label)
library(gbm)
library(caret)
gbm = gbm(label ~., distribution="multinomial", data=train.data, n.trees=5000, interaction.depth =6, shrinkage=0.05, n.minobsinnode = 10)
#outputfile 
yhat <- predict.gbm(gbm,testing_with_dummy, n.trees = 5000)

importance = summary(gbmFit, plotit=TRUE)
#H2O Model
require(h2o)

localH2O = h2o.init(max_mem_size = '6g', # use 6GB of RAM of *GB available
                    nthreads = -1) # use all CPUs (8 on my personal computer :3)
train[,1] = as.factor(train[,1]) # convert digit labels to factor for classification
train_h2o = as.h2o(train)
test_h2o = as.h2o(test)
s <- proc.time()

## train model
model =
  h2o.deeplearning(x = 2:785,  # column numbers for predictors
                   y = 1,   # column number for label
                   training_frame = train_h2o, # data in H2O format
                   activation = "RectifierWithDropout", # algorithm
                   input_dropout_ratio = 0.2, # % of inputs dropout
                   hidden_dropout_ratios = c(0.5,0.5), # % for nodes dropout
                   balance_classes = TRUE, 
                   hidden = c(100,100), # two layers of 100 nodes
                   momentum_stable = 0.99,
                   nesterov_accelerated_gradient = T, # use it for speed
                   epochs = 15) # no. of epochs
h2o.confusionMatrix(model)
s - proc.time()
## classify test set
h2o_y_test <- h2o.predict(model, test_h2o)

## convert H2O format into data frame and  save as csv
df_y_test = as.data.frame(h2o_y_test)
df_y_test = data.frame(ImageId = seq(1,length(df_y_test$predict)), Label = df_y_test$predict)
write.csv(df_y_test, file = "submission-r-h2o.csv", row.names=F)

## shut down virutal H2O cluster
h2o.shutdown(prompt = F)
#set parameter space
activation_opt <- c("Rectifier","RectifierWithDropout", "Maxout","MaxoutWithDropout")
hidden_opt <- list(c(10,10),c(20,15),c(50,50,50))
l1_opt <- c(0,1e-3,1e-5)
l2_opt <- c(0,1e-3,1e-5)

hyper_params <- list( activation=activation_opt,
                      hidden=hidden_opt,
                      l1=l1_opt,
                      l2=l2_opt )

#set search criteria
search_criteria <- list(strategy = "RandomDiscrete", max_models=10)

#train model
dl_grid <- h2o.grid("deeplearning"
                    ,grid_id = "deep_learn"
                    ,hyper_params = hyper_params
                    ,search_criteria = search_criteria
                    ,training_frame = trainh2o
                    ,x = 2:785 
                    ,y = 1
                    ,nfolds = 5
                    ,epochs = 100)

#get best model
d_grid <- h2o.getGrid("deep_learn",sort_by = "accuracy")
best_dl_model <- h2o.getModel(d_grid@model_ids[[1]])
h2o.performance (best_dl_model,xval = T) #CV Accuracy - 84.7%
h2o.performance(deepmodel,xval = T) #84.5 % CV accuracy

#load package
require(mxnet)

#convert target variables into numeric
train[,target := as.numeric(target)-1]
test[,target := as.numeric(target)-1]

#convert train data to matrix
train.x <- data.matrix(train[,-c("target"),with=F])
train.y <- train$target

#convert test data to matrix
test.x <- data.matrix(test[,-c("target"),with=F])
test.y <- test$target
#set seed to reproduce results
mx.set.seed(1)

mlpmodel <- mx.mlp(data = train.x
                   ,label = train.y
                   ,hidden_node = 3 #one layer with 10 nodes
                   ,out_node = 2
                   ,out_activation = "softmax" #softmax return probability
                   ,num.round = 100 #number of iterations over training data
                   ,array.batch.size = 20 #after every batch weights will get updated
                   ,learning.rate = 0.03 #same as step size
                   ,eval.metric= mx.metric.accuracy
                   ,eval.data = list(data = test.x, label = test.y))
#create NN structure
data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, num_hidden=3) #3 neuron in one layer
lrm <- mx.symbol.SoftmaxOutput(fc1)