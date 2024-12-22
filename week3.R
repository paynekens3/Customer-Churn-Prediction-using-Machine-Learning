install.packages(c("tidyverse", "caret", "randomForest", "xgboost", "e1071", "pROC"))
library(tidyverse)
library(caret)
library(randomForest)
library(xgboost)
library(e1071)
library(pROC)
data <- read.csv("C:/Users/mcken/OneDrive/Desktop/WA_Fn-UseC_-Telco-Customer-Churn.csv")
# Preview the data
head(data)
# Check for missing values
colSums(is.na(data))
# Handling missing values (if any) in 'TotalCharges' column
data$TotalCharges[is.na(data$TotalCharges)] <- mean(data$TotalCharges, na.rm = TRUE)
# Convert categorical variables to factors
data$Churn <- as.factor(data$Churn)
data$SeniorCitizen <- as.factor(data$SeniorCitizen)
data$Partner <- as.factor(data$Partner)
data$Dependents <- as.factor(data$Dependents)
data$PhoneService <- as.factor(data$PhoneService)
data$MultipleLines <- as.factor(data$MultipleLines)
data$InternetService <- as.factor(data$InternetService)
data$OnlineSecurity <- as.factor(data$OnlineSecurity)
data$TechSupport <- as.factor(data$TechSupport)
data$Contract <- as.factor(data$Contract)
data$PaperlessBilling <- as.factor(data$PaperlessBilling)
data$PaymentMethod <- as.factor(data$PaymentMethod)
# Checking data structure after conversion
str(data)
# Univariate Analysis: Distribution of key variables
ggplot(data, aes(x=tenure)) + geom_histogram(bins=30, fill="skyblue", color="black") + ggtitle("Distribution of Tenure")
ggplot(data, aes(x=MonthlyCharges)) + geom_histogram(bins=30, fill="salmon", color="black") + ggtitle("Distribution of Monthly Charges")
# Bivariate Analysis: Relationship between churn and other features
ggplot(data, aes(x=tenure, fill=Churn)) + geom_histogram(position="stack", bins=30) + ggtitle("Tenure vs Churn")
ggplot(data, aes(x=MonthlyCharges, fill=Churn)) + geom_histogram(position="stack", bins=30) + ggtitle("Monthly Charges vs Churn")
data$ChargeRatio <- data$MonthlyCharges / data$TotalCharges
head(data)
# Check class distribution
table(data$Churn)
# Apply oversampling using the caret package
set.seed(123)
oversample <- upSample(x = data[, -ncol(data)], y = data$Churn)
# Check the balanced dataset
table(oversample$Class)
# Distribution of Tenure
ggplot(data, aes(x=tenure)) +
  geom_histogram(bins=30, fill="skyblue", color="black") +
  ggtitle("Distribution of Tenure") +
  xlab("Tenure (Months)") +
  ylab("Frequency")
# Monthly Charges vs. Churn
ggplot(data, aes(x=MonthlyCharges, fill=Churn)) +
  geom_histogram(position="stack", bins=30) +
  ggtitle("Monthly Charges vs. Churn") +
  xlab("Monthly Charges") +
  ylab("Frequency")
# Total Charge vs. Churn
ggplot(data, aes(x=TotalCharges, fill=Churn)) +
  geom_histogram(position="stack", bins=30) +
  ggtitle("Total Charge vs. Churn") +
  xlab("Total Charge") +
  ylab("Frequency")


