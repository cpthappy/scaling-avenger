library(Amelia)
library(dplyr)
library(ggplot2)
library(caret)
library(lubridate)

training_data <- read.csv("pml-training.csv", na.strings = c("NA", "", "#DIV/0!"))
training_data %>% filter(new_window=="no") -> training_data

nr_rows <- dim(training_data)[[1]]
na_count <- apply(X = training_data, 2, function(x) {sum(is.na(x))})

training_data <- tbl_df(training_data[, !(na_count == nr_rows)])
training_data %>% select(-new_window, -X) -> training_data
training_data %>% mutate(cvtd_timestamp= dmy_hm(training_data$cvtd_timestamp)) -> training_data

in_train <- createDataPartition(training_data$classe, p = 0.75, list=FALSE)

training <- training_data[in_train,]
cross_validation <- training_data[-in_train,]

fit_control <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 5,
  ## repeated ten times
  repeats = 5)

model_fit <- train(classe~.,
                   data=training,
                   method="rf",
                   trControl=fit_control,
                   allowParallel=TRUE )
