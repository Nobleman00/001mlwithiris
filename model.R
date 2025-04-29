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

# Compare scatter plot of the 80 and 20 data subsets

# Column to indicate the subset
TrainingSet$Set <- "Training"
TestingSet$Set <- "Testing"

# Combine both
combined_sets <- rbind(TrainingSet, TestingSet)

# Jitter plot using species
ggplot(combined_sets, aes(x = Species, y = Set, color = Set)) + geom_jitter(width = 0.2, height = 0.2, alpha = 0.7) + labs(title = "Jitter Plot of Species by Set", x = "Species", y = "Set (Training or Testing)") + theme_minimal()






#####################################
# SVM model (polynomial kernel)

# Build Training Model
Model<-train(Species ~ ., data = TrainingSet,
             method = "svmPoly",
             na.action = na.omit,
             preProcess = c("scale", "center"),
             trControl = trainControl(method = "none"),
             tuneGrid = data.frame(degree=1,scale=1,C=1)
)

# Build CV model
Model.cv<-train(Species ~ ., data = TrainingSet,
                method = "svmPoly",
                na.action = na.omit,
                preProcess=c("scale", "center"),
                trControl = trainControl(method = "cv", number=10),
                tuneGrid = data.frame(degree=1,scale=1,C=1)
)


# Apply model for prediction
Model.training<-predict(Model, TrainingSet) # Apply model to make prediction on Training set
Model.testing<-predict(Model, TestingSet) # Apply model to make prediction on Testing set
Model.cv<-predict(Model.cv, TrainingSet) # Perform cross-validation

# Model performance(Displays confusion matrix and statistics)
Model.training.confusion<-confusionMatrix(Model.training ,TrainingSet$Species)
Model.testing.confusion<-confusionMatrix(Model.testing ,TestingSet$Species)
Model.cv.confusion<-confusionMatrix(Model.cv ,TrainingSet$Species)

print(Model.training.confusion)
print(Model.testing.confusion)
print(Model.cv.confusion)

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
training_metrics<-calculate_metrics(Model.training.confusion)
print("Training Metrics:")
print(training_metrics)


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
plot_training_cm<-visualize__confusion_matrix(Model.training.confusion,
                                              "Training Set Confusion Matrix")
plot_testing_cm<-visualize__confusion_matrix(Model.testing.confusion,
                                             "Testing Set Confusion Matrix")

print(plot_training_cm)
print(plot_testing_cm)

# Assignment: Use the same method and functions;
# for evaluation metrics, use the KNN model algorithm and any other model algorithm of choice (criteria - ability to be used on Classification task and the dataset).
# Create a new model_nameofchoicealgorithm.R file and build there for each model algorithm.
# Feature importance
Importance<-varImp(Model)
plot(Importance)
plot(Importance,col="red")