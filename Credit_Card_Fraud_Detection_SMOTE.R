credit_card <- read.csv("C:/UC/Extra Skills/Credit Card Fraud Detection/creditcard.csv")

str(credit_card)
# Class = 0 implies legitimate transactions

#---------------------------------------------------------
# Creating training and test data sets

install.packages('caTools')
library(caTools)

set.seed(123)
data_sample = sample.split(credit_card$Class, SplitRatio = 0.8)
train_data = subset(credit_card, data_sample == TRUE)
test_data = subset(credit_card, data_sample == FALSE)

dim(train_data)
dim(test_data)


#---------------------------------------------------------------
# Random Over Sampling - need to increase fraud cases in the train dataset

table(train_data$Class)  # 227452 -> 0 transactions
n_legit <- 227452
new_frac_legit <- 0.5
new_n_total <-  n_legit/new_frac_legit

install.packages("ROSE")
library("ROSE")
oversampling_result <- ovun.sample(Class ~. , data = train_data,
                                   method = "over", N = new_n_total,
                                   seed = 2019)

# Storing the result of above in oversampled_credit variable
oversampled_credit <- oversampling_result$data

table(oversampled_credit$Class) # Same number of 0,1 transactions

#-----------------------------------------------------------------------
# Random Under Sampling - decrease legitimate cases in the train dataset

table(train_data$Class) # 594 fraud cases
n_fraud <- 394
new_frac_fraud <- 0.5
new_n_total <- n_fraud / new_frac_fraud #394/0.5  

library("ROSE")
undersampling_result <- ovun.sample(Class ~., data = train_data,
                                    method = "under",
                                    N = new_n_total,
                                    seed = 2019)

undersampled_credit <- undersampling_result$data
table(undersampled_credit$Class)

#---------------------------------------------------------------------
# both ROS and RUS
n_new <- nrow(train_data) # 227846
fraction_fraud_new <- 0.5

sampling_result <- ovun.sample(Class~., data = train_data,
                               method = "both",
                               N = n_new,
                               p = fraction_fraud_new,
                               seed = 2019)

sampled_credit <- sampling_result$data
table(sampled_credit$Class)

#----------------------------------------------------------------------
# Using SMOTE to balance the dataset

install.packages("smotefamily")
library("smotefamily")

table(train_data$Class)

# Set the number of fraud and legitimate cases, and the desired percentage
# of legitimate cases

n0 <- 227452
n1 <- 394
r0 <- 0.6 # ratio that we want after the SMOTE

# Calculate the value for the dup_size paramter of the SMOTE
ntimes <- ((1 - r0) / r0) * (n0 / n1) - 1
# my SMOTE process below will run for ntimes i.e. 383.85 time

smote_output = SMOTE(X = train_data[, -c(1, 31)],
                     target = train_data$Class,
                     K = 5,
                     dup_size = ntimes)
# k was KNNth k

credit_smote <- smote_output$data
colnames(credit_smote)[30] <- "Class"
str(credit_smote)
prop.table(table(credit_smote$Class))

library(ggplot2)
ggplot(credit_smote, aes(x = V1, y= V2, color = Class)) + 
  geom_point() + 
  scale_color_manual(values = c('dodgerblue2', 'red'))

#---------------------------------------------------------------------

library("rpart")
library('rpart.plot')

CART_model <- rpart(Class ~ . , credit_smote)
rpart.plot(CART_model, extra = 0, type = 5, tweak = 1.2)

# predict fraud classes
predicted_val <- predict(CART_model, test_data, type = 'class')

# Build Confusion Matrix
library(caret)
test_data$Class <- as.factor(test_data$Class)
confusionMatrix(predicted_val, test_data$Class)

str(predicted_val)
(test_data$Class)
