############### Logistic Regression ###############


###################################################
# Step 1: Read in and Examine the Data
###################################################
Mydata <- read.csv("Telco-Customer-Churn.csv",header=TRUE,sep=",")

#Explore data
summary(Mydata)
head(Mydata)
table(Mydata$Churn)
table(Mydata$Contract,Mydata$Churn)
table(Mydata$tenure_interval,Mydata$Churn)
table(Mydata$InternetService,Mydata$Churn)

# Check for any missing values
any(is.na(Mydata))
# Omit all rows with missing values
Mydata_noNA <- na.omit(Mydata)



###################################################
# Step 2: Converting variables to factors
###################################################
## Assign data as factor
Mydata_noNA$MultipleLines <- as.factor(Mydata_noNA$MultipleLines)
Mydata_noNA$OnlineSecurity <- as.factor(Mydata_noNA$OnlineSecurity)
Mydata_noNA$OnlineBackup <- as.factor(Mydata_noNA$OnlineBackup)
Mydata_noNA$DeviceProtection <- as.factor(Mydata_noNA$DeviceProtection)
Mydata_noNA$TechSupport <- as.factor(Mydata_noNA$TechSupport)
Mydata_noNA$StreamingTV <- as.factor(Mydata_noNA$StreamingTV)
Mydata_noNA$StreamingMovies <- as.factor(Mydata_noNA$StreamingMovies)

###################################################
#Step 3: Build and Review the Logistic Regression Model
###################################################

# sample the input data with 75% for training and 25% for testing
library(caTools)
sample <- sample.split(Mydata_noNA$Churn,SplitRatio=0.75)
trainData <- subset(Mydata_noNA,sample==TRUE)
testData <- subset(Mydata_noNA,sample==FALSE)

# Building model with all variables
mylogit <- glm(Churn ~ .,
           data =trainData, family=binomial(link="logit"),
           na.action=na.pass)
summary(mylogit)
# Based on p value we can see that there are number of insignificant variables.


###################################################
#Step 4: Variable selection using stepwise regression
###################################################
## The Intercept Model
model.null = glm(Churn ~ 1, 
                 data=trainData,
                 family = binomial(link="logit"))

## Model with all required attributes included
## . in the formula indicates usage of all available metrics
model.full = glm(Churn ~ .,
                 data=trainData,
                 family = binomial(link="logit"))


## Stepwise regression based on Chisq selection
?step
step(model.null,
     scope = list(upper=model.full),
     direction="forward",
     test="Chisq",
     data=trainData)

# Forward selection shows us the best variables to be included.
# Based on business criteria we could select the best model from the list
# Based on AIC from Step:  AIC=5833.71 the reduction in AIC is minimal
# Selecting this model we run the glm

mylogit <- glm(Churn ~ Contract + InternetService + tenure_interval + PaperlessBilling + 
                 SeniorCitizen + StreamingTV + TechSupport + PaymentMethod + 
                 StreamingMovies + MultipleLines + MonthlyCharges + TotalCharges,
               data =trainData, family=binomial(link="logit"),
               na.action=na.pass)
summary(mylogit)


###################################################
#Step 5: Review the Results 
###################################################

# Compute Pseudo R-squared
attributes(mylogit)  # get me the names of the 'class members'
1- with(mylogit, deviance/null.deviance)


###################################################
#Step 6: Use relevel Function to re-level the Price
#   factor with value 30 as the base reference
###################################################
#We will now change the reference level at price point 30

trainData$Contract = relevel(as.factor(trainData$Contract), "Two year")

mylogit2 = glm(Churn ~ Contract + InternetService + tenure_interval + PaperlessBilling + 
                 SeniorCitizen + StreamingTV + TechSupport + PaymentMethod + 
                 StreamingMovies + MultipleLines + MonthlyCharges + TotalCharges ,
            data= trainData,family=binomial(link="logit"), na.action=na.pass)
summary(mylogit2)

###################################################
#Step 7: Plot the ROC Curve
###################################################
library(bitops)
library(caTools)
library(ROCR)

pred = predict(mylogit,newdata = testData, type="response") # this returns the probability scores on the training data
predObj = prediction(pred, testData$Churn) # prediction object needed by ROCR

?performance

rocObj = performance(predObj, measure="tpr", x.measure="fpr")
aucObj = performance(predObj, measure="auc") 
auc = aucObj@y.values[[1]]  
auc  

# plot the roc curve
plot(rocObj, main = paste("Area under the curve:", auc))

# if the prediction probability is greater than 0.5 then those 
# customers are classified as churned customer 
# less than 0.5 are classified as not churning customer
fitted.results <- ifelse(pred > 0.5,"Yes","No")

# calculating the misclassfication rate
misClasificationError <- mean(fitted.results!=testData$Churn)
misClasificationError

# calculating the accuracy rate
accuracyRate <- 1-misClasificationError
accuracyRate

###################################################
#Step 8: Predict Outcome for a Sequence of MonthlyCharges Values 
#         at tenure_interval 12-24 Month and Contract One year 
#         and InternetService Fiber optic
###################################################
newdata1 <- data.frame(MonthlyCharges=seq(min(Mydata_noNA$MonthlyCharges),max(Mydata_noNA$MonthlyCharges),10),
                       tenure_interval=c("12-24 Month"),Contract=c("One year"),
                       InternetService=c("Fiber optic"),
                       PaperlessBilling=c("Yes"), 
                       SeniorCitizen=0, 
                       StreamingTV =c("Yes"), 
                       TechSupport = c("Yes"), 
                       PaymentMethod = c("Electronic check"),
                       StreamingMovies = c("Yes"), 
                       MultipleLines = c("Yes"), 
                       TotalCharges=1000)
newdata1
newdata1$MonthlyChargesP<-predict(mylogit,newdata=newdata1,type="response")
cbind(newdata1$MonthlyCharges,newdata1$MonthlyChargesP)
plot(newdata1$MonthlyCharges,newdata1$MonthlyChargesP)

