#install packages for assignment
install.packages("ripa")
library(ripa)

install.packages("jpeg")
library(jpeg)

source("http://bioconductor.org/biocLite.R")
biocLite("EBImage")
library(EBImage)

#create a matrix with the files and source names
image_names <- list.files(path = "images_training_rev1")

#declare variables
x = 0
v = 0
q10 = 0
q25 = 0
q75 = 0
q90 = 0

for(n in 1:61578)
{
  image = readJPEG(paste0("images_training_rev1/", image_names[n]))
  #convert images to grayscale
  image1 <- as.matrix(rgb2grey(image))
  #resize images to 50x50
  resize(image1, 50, 50)
  #compute mean, variance, 10th, 25th, 75th, 90th quantiles images
  x[n] = mean(image1)
  v[n] = var(image1)
  q10[n] = quantile(image1, probs = .1)
  q25[n] = quantile(image1, probs = .25)
  q75[n] = quantile(image1, probs = .75)
  q90[n] = quantile(image1, probs = .9)
}

#store features into dataframe
images <- data.frame(x, v, q10, q25, q75, q90)
#save features so that they are not lost and don't need to rerun this code
write.csv(images, "GalaxyImages.csv", row.names = FALSE)

#pull up features and include the names of images
images <- read.csv("GalaxyImages.csv")
images["GalaxyID"] <- image_names
images$GalaxyID <- sub(".jpg", "", images$GalaxyID)

#bring in the training probabilities
images_prob <- read.csv("galaxy_train.csv")

#merge the features with the probabilities, separate training and testing images
images_all <- merge(x = images, y = images_prob, by = "GalaxyID", all = TRUE)
images_train <- na.omit(images_all)
images_test <- subset(images_all, is.na(images_all$Prob_Smooth))

#bring in caret package
library(caret)

#Use data splitting
DataSplit1 <- createDataPartition(y=images_train$Prob_Smooth, p=0.75, list=FALSE)

train1 <- images_train[DataSplit1,]
test1 <- images_train[-DataSplit1]

Galaxy1 <- train(Prob_Smooth~x+v+q10+q25+q75+q90, data = images_train, method = "glm")
summary(Galaxy1)

#Coefficients:
#                Estimate Std. Error t value Pr(>|t|)    
#  (Intercept)   0.363455   0.007821  46.471  < 2e-16 ***
#  x            -4.213173   0.578963  -7.277 3.49e-13 ***
#  v            -2.181923   0.345315  -6.319 2.67e-10 ***
#  q10         -11.915672   1.198295  -9.944  < 2e-16 ***
#  q25          13.506510   1.217020  11.098  < 2e-16 ***
#  q75          -1.715120   0.279108  -6.145 8.09e-10 ***
#  q90           2.901330   0.157485  18.423  < 2e-16 ***

#We can see here that all the features are significant for estimating Prob_Smooth

#Predict for the test data
Galaxy_Predictions1 <- predict(Galaxy1, newdata = images_test)

#Put predictions in correct format
Galaxy_Info1 <- data.frame("GalaxyID" = images_test$GalaxyID, "Prob_Smooth" = Galaxy_Predictions1)

#Output predictions
write.csv(Galaxy_Info1, "Galaxy_Predict1.csv", row.names = FALSE)

#Score: 0.27655

#
#
#
#
#
#
#
#
#
#

#install necessary packages
install.packages("kernlab")
library(kernlab)

#Use data splitting
DataSplit2 <- createDataPartition(y=images_train$Prob_Smooth, p=0.75, list=FALSE)

train2 <- images_train[DataSplit2,]
test2 <- images_train[-DataSplit2]

Galaxy2 <- train(Prob_Smooth~x+v+q10+q25+q75+q90, data = images_train, method = "gaussprLinear")
summary(Galaxy2)

Galaxy_Predictions2 <- predict(Galaxy2, newdata = images_test)

Galaxy_Info2 <- data.frame("GalaxyID" = images_test$GalaxyID, "Prob_Smooth" = Galaxy_Predictions2)
write.csv(Galaxy_Info2, "Galaxy_Predict2.csv", row.names = FALSE)

#Did not work, error when tried to train data with guassprLinear method

#
#
#
#
#
#
#
#
#
#

#declare variable for variance of transpose, since that is the only feeature that
#will change when taking the transpose
vt = 0

for(n in 1:61578)
{
  image = readJPEG(paste0("images_training_rev1/", image_names[n]))
  #convert images to grayscale
  image1 <- as.matrix(rgb2grey(image))
  #resize images to 50x50
  resize(image1, 50, 50)
  #take the transpose
  image2 <- t(image1)
  #compute variance
  vt[n] = var(image2)
}

#copy images_all into images_all_t and update variance
images_all_t <- images_all
images_all_t$v <- vt

#store all the results
write.csv(images_all_t, "GalaxyImagesT.csv", row.names = FALSE)

#bring in the data and split it between training and testing
images_all_t <- read.csv("GalaxyImagesT.csv")
images_train_t <- na.omit(images_all_t)
images_test_t <- subset(images_all_t, is.na(images_all_t$Prob_Smooth))

#install necessary packages
install.packages("caret")
install.packages("quantreg")
library(caret)

#use a gbm model
set.seed(123)
Galaxy_control_t <- trainControl(method = "repeatedcv", number = 2, repeats = 1, verbose = TRUE)
gbmGrid_t <-  expand.grid(interaction.depth = c(1, 2), n.trees = seq(200,10000, by=200), shrinkage = c(0.1,.05), n.minobsinnode=1)
gbm_fit_t <- train(Prob_Smooth~x+v+q10+q25+q75+q90, data = images_train_t, method = "gbm", trControl = Galaxy_control_t, verbose = FALSE, tuneGrid = gbmGrid_t)

#use the model to predict the test data
Galaxy_Predictions_t <- predict(gbm_fit_t, newdata = images_test_t)

#output the predictions in the correct formal
Galaxy_Info_t <- data.frame("GalaxyID" = images_test$GalaxyID, "Prob_Smooth" = Galaxy_Predictions_t)
write.csv(Galaxy_Info_t, "Galaxy_Predict_t.csv", row.names = FALSE)

#Score: 0.25561

#
#
#
#
#
#
#
#
#
#

#install necessary package for cropping
install.packages("fields")
library(fields)

#Set the size for cropping
galaxy_crop = matrix(c(10, 40, 10, 40), nrow = 2, ncol = 2)

for(n in 1:61578)
{
  image = readJPEG(paste0("images_training_rev1/", image_names[n]))
  #convert images to grayscale
  image1 <- as.matrix(rgb2grey(image))
  #resize images to 50x50
  resize(image1, 50, 50)
  
  #crop image
  image2 <- crop.image(image1, loc=galaxy_crop)
  
  #compute mean, variance, 10th, 25th, 75th, 90th quantiles images
  x[n] = mean(image2)
  v[n] = var(image2)
  q10[n] = quantile(image2, probs = .1)
  q25[n] = quantile(image2, probs = .25)
  q75[n] = quantile(image2, probs = .75)
  q90[n] = quantile(image2, probs = .9)
}

#Did not work, could not figure out how to crop image right, this was best guess
#in attempting this feature

#
#
#
#
#
#
#
#
#
#

#install necessary packages
install.packages("caret")
install.packages("quantreg")
library(caret)

#fit a gbm model on original features
set.seed(123)
Galaxy_control <- trainControl(method = "repeatedcv", number = 2, repeats = 1, verbose = TRUE)
gbmGrid <-  expand.grid(interaction.depth = c(1, 2), n.trees = seq(200,10000, by=200), shrinkage = c(0.1,.05), n.minobsinnode=1)
gbm_fit <- train(Prob_Smooth~x+v+q10+q25+q75+q90, data = images_train, method = "gbm", trControl = Galaxy_control, verbose = FALSE, tuneGrid = gbmGrid)

#use model to predict test data
Galaxy_Predictions4 <- predict(gbm_fit, newdata = images_test)

#output predictions in correct format
Galaxy_Info4 <- data.frame("GalaxyID" = images_test$GalaxyID, "Prob_Smooth" = Galaxy_Predictions4)
write.csv(Galaxy_Info4, "Galaxy_Predict4.csv", row.names = FALSE)

#Score: 0.25603

#
#
#
#
#
#
#
#
#
#

#try scaling images, declare variables
xe = 0
ve = 0
q10e = 0
q25e = 0
q75e = 0
q90e = 0

for(n in 1:61578)
{
  image = readJPEG(paste0("images_training_rev1/", image_names[n]))
  #convert images to grayscale
  image1 <- as.matrix(rgb2grey(image))
  #resize images to 50x50
  resize(image1, 50, 50)
  #centered matrix
  image2 <- scale(image1, center = TRUE, scale = TRUE)
  image2
  #compute mean, variance, 10th, 25th, 75th, 90th quantiles images
  xe[n] = mean(image2)
  ve[n] = var(image2)
  q10e[n] = quantile(image2, probs = .1)
  q25e[n] = quantile(image2, probs = .25)
  q75e[n] = quantile(image2, probs = .75)
  q90e[n] = quantile(image2, probs = .9)
}

#there was an error when n was in the 17,000s with the scaling formula

#
#
#
#
#
#
#
#
#
#

#try a glm model with gaussian family
Galaxy_fit5 <- glm(Prob_Smooth~x+v+q10+q25+q75+q90, data = images_train, family = "gaussian")

#predict test values using the model
Galaxy_Predictions_5 <- predict(Galaxy_fit5, images_test, type = "response")

#output predictions in proper format
Galaxy_Info_5 <- data.frame("GalaxyID" = images_test$GalaxyID, "Prob_Smooth" = Galaxy_Predictions_5)
write.csv(Galaxy_Info_5, "Galaxy_Predict_5.csv", row.names = FALSE)

#Score: 0.27655

#
#
#
#
#
#
#
#
#
#

#try glm gaussian model with the transpose
Galaxy_fit6 <- glm(Prob_Smooth~x+v+q10+q25+q75+q90, data = images_train_t, family = "gaussian")

#make predictions
Galaxy_Predictions_6 <- predict(Galaxy_fit6, images_test_t, type = "response")

#output predictions
Galaxy_Info_6 <- data.frame("GalaxyID" = images_test_t$GalaxyID, "Prob_Smooth" = Galaxy_Predictions_6)
write.csv(Galaxy_Info_6, "Galaxy_Predict_6.csv", row.names = FALSE)

#Score: 0.27648

#
#
#
#
#
#
#
#
#
#

#download necessary packages
install.packages("caret")
install.packages("quantreg")
library(caret)

#Use a gbm train model with a higher number of repeated CV and with the transpose
set.seed(123)
Galaxy_control_t3 <- trainControl(method = "repeatedcv", number = 10, repeats = 1, verbose = TRUE)
gbmGrid_t3 <-  expand.grid(interaction.depth = c(1, 2), n.trees = seq(200,10000, by=200), shrinkage = c(0.1,.05), n.minobsinnode=1)
gbm_fit_t3 <- train(Prob_Smooth~x+v+q10+q25+q75+q90, data = images_train_t, method = "gbm", trControl = Galaxy_control_t3, verbose = FALSE, tuneGrid = gbmGrid_t3)

#make predictions
Galaxy_Predictions_t3 <- predict(gbm_fit_t3, newdata = images_test_t)

#output predictions
Galaxy_Info_t3 <- data.frame("GalaxyID" = images_test_t$GalaxyID, "Prob_Smooth" = Galaxy_Predictions_t3)
write.csv(Galaxy_Info_t3, "Galaxy_Predict_t3.csv", row.names = FALSE)

#Score: 0.25541





