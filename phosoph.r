
library(ggplot2)
library(tidyverse)
library(randomForest)
library(MLmetrics)
library(caret)
library(class)
library(caTools)

test <- read.csv("test.csv")
test$Response = NA
train <- read.csv("train.csv")

complete <- rbind(test,train)
complete$Response <- as.factor(complete$Response)




train <- complete %>% filter(!is.na(PSSM))
test <- complete %>% filter(is.na(PSSM))


set.seed(222)
train.imputed <- rfImpute(PSSM ~ SVM + ANN, train)
set.seed(333)
completed.rf <- randomForest(PSSM ~ SVM + ANN, train)


test$PSSM <- predict(completed.rf, test)
test$Consensus <- (test$ANN + test$PSSM + test$SVM) / 3

completed <- rbind(test,train)

f1 <- function(data, lev = NULL, model = NULL) {
  f1_val <- F1_Score(y_pred = data$pred, y_true = data$obs, positive = "Yes") 
  c(F1 = f1_val)
}

myControl_model <- trainControl(method = "repeatedcv", 
                                number = 5,
                                repeats = 4,
                                classProbs = TRUE,
                                summaryFunction = f1)

completed <- completed %>% 
  mutate(Response = ifelse(Response == 0, "No", "Yes"))
train <- completed %>% filter(!is.na(Response))
test <- completed %>% filter(is.na(Response))


############################
### Using XG Boost ###
############################
xgb_model <- train(as.factor(Response)~.-SiteNum,
                   data = train, 
                   method = "xgbTree",
                   tuneLength = 2,
                   metric = "F",
                   trControl = myControl_model
)
xgb_model

preds <- predict(xgb_model, newdata = test)
preds.frame <- data.frame(Id = test$SiteNum, Predicted = preds)
preds.frame <- preds.frame %>% 
  mutate(Predicted = ifelse(Predicted == "No", 0, 1))

write_csv(preds.frame, "xgb_model_rfImpute.csv")




suppressWarnings(suppressMessages(library(kknn)))
#################
### Using KNN ###
#################
knn_model <- train.kknn(as.factor(Response)~.-SiteNum,data = train,kmax=5)
knn_model  


preds <- predict(knn_model, newdata = test)
preds.frame <- data.frame(Id = test$SiteNum, Predicted = preds)
preds.frame <- preds.frame %>% 
  mutate(Predicted = ifelse(Predicted == "No", 0, 1))


write_csv(preds.frame, "knn_model_rfImpute2.csv")


##############################
### Using Boosted Logistic ###
##############################

LogitBoost_model <- train(as.factor(Response)~.-SiteNum,
                   data = train, 
                   method = "LogitBoost",
                   nIter=100,
                   metric = "F1",
                   trControl = myControl_model
)
LogitBoost_model

preds <- predict(LogitBoost_model, newdata = test)
preds.frame <- data.frame(Id = test$SiteNum, Predicted = preds)
preds.frame <- preds.frame %>% 
  mutate(Predicted = ifelse(Predicted == "No", 0, 1))

write_csv(preds.frame, "LogitBoost_model_rfImpute.csv")


############################
### Using Gradient Boost ###
############################


gradientBoost <- train(as.factor(Response)~.-SiteNum,
                            data = train, 
                            method = "gbm",
                            verbose = FALSE,
                            trControl = myControl_model,
                            metric="F1"
)
preds <- predict(gradientBoost, newdata = test)
preds.frame <- data.frame(Id = test$SiteNum, Predicted = preds)
preds.frame <- preds.frame %>% 
  mutate(Predicted = ifelse(Predicted == "No", 0, 1))

write_csv(preds.frame, "gradientBoost2_model_rfImpute.csv")





