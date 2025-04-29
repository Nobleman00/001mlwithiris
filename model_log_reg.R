# Importing libraries
library(caret) # Package for machine learning algorithms
library(nnet)

# Importing the Iris data set
irs<-datasets::iris
irs<-data.frame(irs)

# missing values
# sum(is.na(iris))

# apply column transformation
cl_irs_col <- function(df) {
   new_names <- names(df)
   new_names <- gsub("\\.", "", new_names)
   
   names(df) <- new_names
   return(df)
}

irs<-cl_irs_col(irs)

# To achieve reproducible model; set the random seed number
set.seed(250)

# Performs stratified random split of the data set
TrainingIndex<-createDataPartition(iris$Species,p=0.8,list = F)
TrainingSet<-irs[TrainingIndex,] # Training Set
TestingSet<-irs[-TrainingIndex,] # Test Set



# Logistic Regression model

# Build Training Model
LogRegModel <- train(Species ~ ., data = TrainingSet,
                     method = "multinom",  # Multinomial logistic regression
                     preProcess = c("scale", "center"),
                     trControl = trainControl(method = "none"))

# Build CV model
LogRegModel.cv <- train(Species ~ ., data = TrainingSet,
                        method = "multinom",  # Multinomial logistic regression
                        preProcess = c("scale", "center"),
                        trControl = trainControl(method = "cv", number = 10))

# Apply model for prediction
LogRegModel.training <- predict(LogRegModel, TrainingSet)
LogRegModel.testing <- predict(LogRegModel, TestingSet)
LogRegModel.cv <- predict(LogRegModel.cv, TrainingSet)

# Model performance(Displays confusion matrix and statistics)
LogRegModel.training.confusion <- confusionMatrix(LogRegModel.training, TrainingSet$Species)
LogRegModel.testing.confusion <- confusionMatrix(LogRegModel.testing, TestingSet$Species)
LogRegModel.cv.confusion <- confusionMatrix(LogRegModel.cv, TrainingSet$Species)

print(LogRegModel.training.confusion)
print(LogRegModel.testing.confusion)
print(LogRegModel.cv.confusion)

# Calculate additional metrics: Accuracy, Precision, Recall, F1 Score
calculate_metrics<-function(conf_matrix) {
   accuracy<-round(conf_matrix$overall["Accuracy"], 2)
   precision<-round(mean(conf_matrix$byClass[, "Precision"]), 2)
   recall<-round(mean(conf_matrix$byClass[, "Recall"]), 2)
   f1_score<-2 * (precision * recall) / (precision + recall)
   
   metrics<-data.frame(
      Metric = c("Accuracy", "Precision", "Recall", "F1 Score"),
      Value = c(accuracy, precision, recall, f1_score)
   )
   return(metrics)
}

# Training metrics
training_metrics_logreg <- calculate_metrics(LogRegModel.training.confusion)
print("Logistic Regression Training Metrics:")
print(training_metrics_logreg)


# Visualize confusion matrix for training set
visualize_confusion_matrix <- function(conf_matrix, title) {
   cm <- as.data.frame(conf_matrix$table)
   colnames(cm) <- c("Actual", "Prediction", "Freq")
   
   ggplot(cm, aes(x = Prediction, y = Actual)) +
      geom_tile(aes(fill = Freq), color = "white") +
      scale_fill_gradient(low = "white", high = "blue") +
      geom_text(aes(label = Freq), vjust = 1) +
      ggtitle(title) +
      theme_minimal()
}

# Plot confusion matrix for training and testing
plot_logreg_training_cm <- visualize_confusion_matrix(LogRegModel.training.confusion, "Logistic Regression Training Set Confusion Matrix")
plot_logreg_testing_cm <- visualize_confusion_matrix(LogRegModel.testing.confusion, "Logistic Regression Testing Set Confusion Matrix")

print(plot_logreg_training_cm)
print(plot_logreg_testing_cm)