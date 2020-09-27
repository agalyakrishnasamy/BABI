
library(ggplot2)
library(gridExtra)
library(corrplot)
library(dplyr)
library(DataExplorer)
library(car)
library(DMwR)
library(caret)
library(ROCR)
library(pROC)


getwd()
data <-read.csv("Cars_edited.csv")

str(data)
summary(data)
head(data)
dim(data)

prop.table(table(Transport))
#check NA values

sapply(data,function(x) sum(is.na(x)))

any(is.na(data))

plot_intro(data)

##detect outliers 
plot_histogram(data)

##Density Plots

plot_density(data)

attach(data)

##Univariate analysis: Numeric variables
colnames(data[,sapply(data, is.numeric)]) 

p1=ggplot(data, aes(x =Age )) + geom_histogram(bins = 30, fill = "lightblue", col = "blue")
p2=ggplot(data, aes(x = Gender,fill = Gender)) + geom_bar()
p3=ggplot(data, aes(x =Engineer )) + geom_histogram(bins = 30, fill = "lightblue", col = "blue")
p4=ggplot(data, aes(x =MBA )) + geom_histogram(bins = 30, fill = "lightblue", col = "blue")
p5=ggplot(data, aes(x =Work.Exp )) + geom_histogram(bins = 30, fill = "lightblue", col = "blue")
p6=ggplot(data, aes(x =Salary )) + geom_histogram(bins = 30, fill = "lightblue", col = "blue")
p7=ggplot(data, aes(x =Distance )) + geom_histogram(bins = 30, fill = "lightblue", col = "blue")
p8=ggplot(data, aes(x =license )) + geom_histogram(bins = 30, fill = "lightblue", col = "blue")
p9=ggplot(data, aes(x = Transport,fill = Transport)) + geom_bar()


grid.arrange(p1, p2, p3, p4,p5,p6,p7,p8,p9, ncol = 3, nrow = 3)

## The columns Engineer,MBA and license need to be converted into factors
data$Engineer<-as.factor(data$Engineer)
data$MBA<-as.factor(data$MBA)
data$license<-as.factor(data$license)

#Univariate analysis: Categorical variables
b1=ggplot(data, aes(x = Engineer,fill = Engineer)) + geom_bar()
b2=ggplot(data, aes(x = MBA,fill = MBA)) + geom_bar()  ##NA value available
b3=ggplot(data, aes(x = license,fill = license)) + geom_bar()

grid.arrange(b1,b2,b3, ncol = 2, nrow = 2)

#Bivariate Analysis:

b5=ggplot(data, aes(x = Transport)) + geom_bar(aes(fill = Gender), position = "dodge")
b6=ggplot(data, aes(x = Transport)) + geom_bar(aes(fill = Engineer), position = "dodge")
b7=ggplot(data, aes(x = Transport)) + geom_bar(aes(fill = MBA), position = "dodge")

b8=ggplot(data, aes(x = Transport)) + geom_bar(aes(fill = license), position = "dodge")
b9=ggplot(data, aes(x = Transport, y = Age)) + geom_boxplot()
b10=ggplot(data, aes(x = Transport, y = Work.Exp)) + geom_boxplot()
b11=ggplot(data, aes(x = Transport, y = Salary)) + geom_boxplot()
b12=ggplot(data, aes(x = Transport, y = Distance)) + geom_boxplot()


grid.arrange(b5, b6, b7,b8 , ncol = 2, nrow = 2)
grid.arrange(b9, b10, b11,b12 , ncol = 2, nrow = 2)

##NA value treatment

data<-knnImputation(data)
any(is.na(data))

##collinearity: age ,salary & work.exp is correlated
data.numeric = data %>% select_if(is.numeric)
corr_data= round(cor(data.numeric),2)
corrplot(corr_data,method="number")




table(Transport)
data$TransportUsage<-ifelse(data$Transport =='Car',1,0)
table(data$TransportUsage)


sum(data$TransportUsage=="1")/nrow(data)
data$TransportUsage<-as.factor(data$TransportUsage)

data=select(data, -Transport)

prop.table(table(data$TransportUsage))


str(data)
summary(data)

##Split the data into test and train
set.seed(777)
carindex<-createDataPartition(data$TransportUsage, p=0.7,list = FALSE)
data_train<-data[carindex,]
data_test<-data[-carindex,]
prop.table(table(data_train$TransportUsage))
prop.table(table(data_test$TransportUsage))
table(data_train$TransportUsage)
table(data_test$TransportUsage)

dim(data_train)
dim(data_test)
table(data_test$TransportUsage)

43/269

#Data Preparation: SMOTE
##install.packages('DMwR')
set.seed(777)
library(DMwR)

SMOTE_TRain <- SMOTE(TransportUsage ~., data_train, perc.over =200 , perc.under = 319)
table(SMOTE_TRain$TransportUsage)
prop.table(table(SMOTE_TRain$TransportUsage))
summary(SMOTE_TRain)
summary(data_train)



##Building Logistic regression model on balanced data
set.seed(777)
library(nnet)

logit_model1 = glm(TransportUsage ~ ., data = SMOTE_TRain, 
                   family = "binomial" (link="logit"))

summary(logit_model1)
vif(logit_model1)

logit_model2 = step(glm(TransportUsage ~ .-Age -Work.Exp , data = SMOTE_TRain, 
                   family = "binomial" (link="logit")))

summary(logit_model2)
vif(logit_model2)


pred = predict(logit_model2, data=SMOTE_TRain, type="response")
y_pred_num = ifelse(pred>0.5,1,0)
y_pred = factor(y_pred_num, levels=c(0,1))
y_actual =SMOTE_TRain$TransportUsage
confusionMatrix(y_pred,y_actual,positive="1")


#Prediction with test data

data_test$log.pred<-predict(logit_model2, data_test, type="response")

table(data_test$TransportUsage,data_test$log.pred>0.5)

predictions <- ifelse(data_test$log.pred>0.5,1,0)
confusionMatrix(table(data_test$TransportUsage,predictions))

## Area under the curve for LR model
ROC = prediction(data_test$log.pred, data_test$TransportUsage)
AUC=as.numeric(performance(ROC, "auc")@y.values)
AUC

### Roc curve for the Log Reg:
performanceroc = performance(ROC, "tpr","fpr")
plot(performanceroc, col="red",lty=2, lwd=2,colorize=T, main="ROC curve", xlab="Specificity", 
     ylab="Sensitivity")

abline(0,1)

# KS
ks.train <- performance(ROC, "tpr", "fpr")
train.ks <- max(attr(ks.train, "y.values")[[1]] - (attr(ks.train, "x.values")[[1]]))
train.ks

# Gini
train.gini = (2 * AUC) - 1
train.gini

##Naive Bayes

set.seed(777)
library(e1071)

NBmodel=naiveBayes(SMOTE_TRain$TransportUsage~.,data=SMOTE_TRain)

print(NBmodel)

pred = predict(NBmodel, newdata = SMOTE_TRain)
confusionMatrix(pred,SMOTE_TRain$TransportUsage,positive="1")

# Performance metrics (test data)
pred2 = predict(NBmodel, newdata = data_test)
confusionMatrix(pred2,data_test$TransportUsage,positive="1")




##KNN model
set.seed(777)

trainct = trainControl(method = "repeatedcv", number = 10, repeats = 3)

set.seed(777)
knn_SMOTE = train(TransportUsage~., data = SMOTE_TRain, method="knn",
            trControl= trainct, preProcess = c("center", "scale"),
            tuneLength= 10)
knn_SMOTE

##Accuracy of KNN:
knn.pred = predict(knn_SMOTE, data_test)
mean(knn.pred == data_test$TransportUsage)

#CM for train kodel
knn.pred_train = predict(knn_SMOTE, SMOTE_TRain)
knn.CM_train = confusionMatrix(knn.pred_train, SMOTE_TRain$TransportUsage, positive = "1")
knn.CM_train


## Confusion Matrix for kNN model:
knn.CM = confusionMatrix(knn.pred, data_test$TransportUsage, positive = "1")
knn.CM

## Area under the curve for KNN model

ROC_knn = prediction(as.numeric(knn.pred), data_test$TransportUsage)
AUC_knn=as.numeric(performance(ROC_knn, "auc")@y.values)
AUC_knn

### Roc curve for the Log Reg:
performanceroc_knn = performance(ROC_knn, "tpr","fpr")
plot(performanceroc_knn, col="red",lty=2, lwd=2,colorize=T, main="ROC curve", xlab="Specificity", 
     ylab="Sensitivity")

abline(0,1)


####Bagging and boosting models


library(gbm)        

library(xgboost)    

## Bagging

library(ipred)
library(rpart)

set.seed(777)
Transport.bagging <- bagging(as.numeric(TransportUsage) ~.,
                          data=data_train,
                          control=rpart.control(maxdepth=15, minsplit=4))


data_test$pred.Transusage <- predict(Transport.bagging, data_test)



table(data_test$TransportUsage,data_test$pred.Transusage>0.5)



    summary(data_test$TransportUsage)            
summary(data_test$pred.Transusage>0.5)
summary(Transport.bagging)

table(data_train$TransportUsage)
table(data_test$TransportUsage)

#boosting
set.seed(111)
library(xgboost)

data_train$Gender=as.numeric(data_train$Gender)
data_train$Engineer=as.numeric(data_train$Engineer)
data_train$MBA=as.numeric(data_train$MBA)
data_train$license=as.numeric(data_train$license)
data_train$TransportUsage=as.numeric(data_train$TransportUsage)


data_test$Gender=as.numeric(data_test$Gender)
data_test$Engineer=as.numeric(data_test$Engineer)
data_test$MBA=as.numeric(data_test$MBA)
data_test$license=as.numeric(data_test$license)
data_test$TransportUsage=as.numeric(data_test$TransportUsage)


data_train$TransportUsage[data_train$TransportUsage == 1] = 0
data_train$TransportUsage[data_train$TransportUsage == 2] = 1
data_test$TransportUsage[data_test$TransportUsage == 1] = 0
data_test$TransportUsage[data_test$TransportUsage == 2] = 1


features_train = as.matrix(data_train[,1:8])
label_train = as.matrix(data_train[,9])
features_test = as.matrix(data_test[,1:8])

XGBmodel = xgboost(
  data = features_train,
  label = label_train,
  eta = .001,
  max_depth = 5,
  min_child_weight = 3,
  nrounds = 10,
  #nfold = 0,
  objective = "binary:logistic",  # for regression models
  verbose = 0,               # silent,
  early_stopping_rounds = 10 # stop if no improvement for 10 consecutive trees
)


XGBpredTest = predict(XGBmodel, features_test)
tabXGB = table(data_test$TransportUsage, XGBpredTest>0.5)
tabXGB


table(XGBpredTest>0.5,data_test$TransportUsage)

#Accuracy: 95.45%
sum(diag(tabXGB))/sum(tabXGB)

#specificity : 83.33%

15/18

#sensitivty :83.33% tp/p

15/18


