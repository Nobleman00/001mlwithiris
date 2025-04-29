# Importing libraries
library(caret) # Package for machine learning algorithms

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



# KNN model

# Build Training Model
Model.knn <- train(Species ~ ., data = TrainingSet,
                   method = "knn",
                   na.action = na.omit,
                   preProcess = c("scale", "center"),
                   trControl = trainControl(method = "none"),
                   tuneGrid = data.frame(k = 5)  # You can change k as needed
)

# Build CV model
Model.knn.cv <- train(Species ~ ., data = TrainingSet,
                      method = "knn",
                      na.action = na.omit,
                      preProcess = c("scale", "center"),
                      trControl = trainControl(method = "cv", number = 10),
                      tuneGrid = data.frame(k = 5)
)


# Apply model for prediction
Model.knn.training <- predict(Model.knn, TrainingSet)
Model.knn.testing <- predict(Model.knn, TestingSet)
Model.knn.cv <- predict(Model.knn.cv, TrainingSet)

# Model performance(Displays confusion matrix and statistics)
Model.knn.training.confusion <- confusionMatrix(Model.knn.training, TrainingSet$Species)
Model.knn.testing.confusion <- confusionMatrix(Model.knn.testing, TestingSet$Species)
Model.knn.cv.confusion <- confusionMatrix(Model.knn.cv, TrainingSet$Species)

# Print confusion matrices
print(Model.knn.training.confusion)
print(Model.knn.testing.confusion)
print(Model.knn.cv.confusion)

# Calculate additional metrics: Accuracy, Precision, Recall, F1 Score
calculate_metrics<-function(conf_matrix) {
   accuracy<-round(conf_matrix$overall["Accuracy"],2)
   precision<-round(mean(conf_matrix$byClass[, "Precision"]),2)
   recall<-round(mean(conf_matrix$byClass[, "Recall"]),2)
   f1_score<-2 * (precision * recall) / (precision + recall)
   
   metrics<-data.frame(
      Metric = c("Accuracy", "Precision", "Recall", "F1 Score"),
      Value = c(accuracy, precision, recall, f1_score)
   )
   return(metrics)
}

# Training metrics
training_metrics_knn <- calculate_metrics(Model.knn.training.confusion)
print("KNN - Training Metrics:")
print(training_metrics_knn)


# Visualize the confusion matrix
visualize__confusion_matrix<-function(conf_matrix, title) {
   cm<-as.data.frame(conf_matrix$table)
   colnames(cm)<-c("Actual", "Prediction", "Freq")
   
   ggplot(cm, aes(x = Prediction, y = Actual)) +
      geom_tile(aes(fill = Freq), color = "white") +
      scale_fill_gradient(low = "white", high = "blue") +
      geom_text(aes(label = Freq), vjust = 1) +
      ggtitle(title) +
      theme_minimal()
}

# Plot confusion matrix
plot_knn_training_cm <- visualize__confusion_matrix(Model.knn.training.confusion,
                                                    "KNN Training Set Confusion Matrix")
plot_knn_testing_cm <- visualize__confusion_matrix(Model.knn.testing.confusion,
                                                   "KNN Testing Set Confusion Matrix")

print(plot_knn_training_cm)
print(plot_knn_testing_cm)