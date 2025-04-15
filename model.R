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

irs_upt<-cl_irs_col(irs)

# To achieve reproducible model; set the random seed number
set.seed(250)

# Performs stratified random split of the data set
TrainingIndex<-createDataPartition(iris$Species,p=0.8,list = F)
TrainingSet<-iris[TrainingIndex,] # Training Set
TestingSet<-iris[-TrainingIndex,] # Test Set

# Compare scatter plot of the 80 and 20 data subsets




#####################################
# SVM model (polynomial kernel)

# Build Training Model
Model<-train(Species ~ ., data = TrainingSet,
             method = "svmPoly",
             na.action = na.omit,
             preProcess=c("scale", "center"),
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

# Feature importance
Importance<-varImp(Model)
plot(Importance)
plot(Importance,col="red")