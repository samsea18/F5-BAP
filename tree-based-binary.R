
pacman::p_load(tidyverse, caret, corrplot,car, relaimpo, rpart, rpart.plot, ROCR, ggplot2)

setwd("C:/Users/zhiwen/Desktop/EBAC/Sem 2/05 Project")
df<-read.csv("restaurant_list.csv",na.strings = c('?'),fileEncoding="UTF-8-BOM")

seed = 123
str(df)

#find the % of NA for every column
colMeans(is.na(df))

# we exclude those variables with > 30% NAs
# special_diets, price_range, price_range_max, price_range_min - 16:19

df_1 = df %>% dplyr::select(-special_diets, -price_range, -price_range_max, -price_range_min)
str(df_1)

table(df['General.Location'])
table(df['Postal.District'])
table(df['cuisines_binned'])


# exclude rank as it is another of our target variable
# exclude restaurant_name as it is not meaningful
# exclude address, Postal.District, Region, postal code, area - represented by Postal.District
# exclude cuisines, binned into cuisines_binned

df_final = df_1 %>% dplyr::select(-overall_score, -restaurant_name, -address, -General.Location, -postal_code, -area, -cuisines, -Region)


#df_final = df_1 %>% dplyr::select(-rank, -restaurant_name, -address, -General.Location, -postal_code, -area, -cuisines, -Region, -cuisines_binned, -Postal.District)

#df_final = df_1 %>% dplyr::select(-rank, -restaurant_name, -address, -Postal.District, -General.Location, -postal_code, -area, -cuisines)

#df_final = df_1 %>% dplyr::select(-rank, -restaurant_name, -address, -Postal.District, -General.Location, -postal_code, -area, -cuisines, -cuisines_binned, -General.Location)



str(df_final)
colSums(is.na(df_final))
colMeans(is.na(df_final))

#remove rows with NA
df_final = df_final[complete.cases(df_final), ]
colMeans(is.na(df_final))
str(df_final)

#prepare binary classification column
df_final = df_final %>% arrange(rank)

head(df_final)

0.2*3541 = 708
#First 708 rows will be 1
df_final['top20percent'] <- NA
df_final$top20percent[1:708] <- 1
df_final$top20percent[708:3541] <- 0

head(df_final)
tail(df_final)


# factorize columns
fact_cols = c('top20percent', 'cuisines_binned', 'Postal.District')
df_final[,fact_cols] = lapply(df_final[,fact_cols], factor)
str(df_final)

# drop rank column
df_final$rank <- NULL
str(df_final)

train_portion = 0.8

#set initial seed for reproducibility
set.seed(123)
# create n% training, 1-n% testing
inds = createDataPartition(df_final$top20percent, p=train_portion, 
                           list=F,times=1)

# generate training data - n% of df_final
train_set = df_final[inds,]
nrow(train_set)/nrow(df_final)
str(train_set)

# generate testing data - 1-n% of df_final
test_set = df_final[-inds,]
nrow(test_set)/nrow(df_final)


#tune maximum tree depth
set.seed(seed)
rpartTune <- train(
                top20percent ~ ., data = train_set,
                method = "rpart2",
                tuneGrid = data.frame(maxdepth = seq(1, 10)),
                trControl = trainControl(
                  method = "cv", number = 10,
                  verboseIter = TRUE, 
            ))
plot(rpartTune)
#Max Tree Depth of 3 seems to be giving the lowest RMSE already
#Stick to 5 for classification > highest accuracy already


# Any improvement in reducing the complexity parameter below the default of 0.01
set.seed(seed)
rpartTune <- train(
  top20percent ~ ., data = train_set,
  method = "rpart",
  tuneGrid = data.frame(cp = seq(0.001, 0.01, 0.001)),
  trControl = trainControl(
    method = "cv", number = 10,
    verboseIter = TRUE, 
  ))

plot(rpartTune)
#cp of 0.003 is giving the highest accuracy already


#Train rpart model
set.seed(seed)
rpart_model <- rpart(top20percent ~ .,
                     data=train_set,
                     control=rpart.control(maxdepth=3,
                                           cp=0.003))

rpart.plot(rpart_model)
print(rpart_model)

rpart_model$variable.importance %>% 
  data.frame() %>%
  rownames_to_column(var = "Feature") %>%
  rename(Overall = '.') %>%
  ggplot(aes(x = fct_reorder(Feature, Overall), y = Overall)) +
  geom_pointrange(aes(ymin = 0, ymax = Overall), color = "cadetblue", size = .3) +
  theme_minimal() +
  coord_flip() +
  labs(x = "", y = "", title = "Variable Importance")

rpart_model$variable.importance

#for rpart classification
rpart_model_train_preds <- predict(rpart_model, train_set, type = "class")
#rpart_model_train_preds
rpart_model_test_preds <- predict(rpart_model, test_set, type = "class")
#rpart_model_test_preds

confusionMatrix(rpart_model_train_preds, train_set$top20percent)
confusionMatrix(rpart_model_test_preds, test_set$top20percent)

#Evaluation metrics for model
#postResample(pred = rpart_model_train_preds, obs = train_set$overall_score)
#postResample(pred = rpart_model_test_preds, obs = test_set$overall_score)

## Random Forest Model

# Tune number of variables to select each time
rf_tune <- train(
  top20percent ~ ., 
  data = train_set, 
  method = "rf",
  tuneGrid = expand.grid(mtry = 1:5),
  trControl = trainControl(
    method = "cv", number = 10,
    verboseIter = TRUE, 
  ))

plot(rf_tune)

#Very little increase in Accuracy post 3

plot(varImp(rf_tune), main="Variable Importance with Random Forest")

#use maxnodes of 5, taken from decision tree
set.seed(seed)
rf_model <- randomForest::randomForest(top20percent ~ .,
                                       data=train_set, 
                                       ntree=1000,
                                       mtry=3,
                                       importance=TRUE,
                                       maxnodes=5,
                                       votes=TRUE,
                                       na.action=randomForest::na.roughfix)

rf_model_train_preds <- predict(rf_model, train_set)
rf_model_train_preds

rf_model_test_preds <- predict(rf_model, test_set)
rf_model_test_preds

rf_model$votes[,2]

rf_model_test_preds_prob <-predict(rf_model, test_set, type = 'prob')
rf_model_test_preds_prob[,2]

randomForest::importance(rf_model)
rn <- round(randomForest::importance(rf_model), 2)
rn
rn[order(rn[,3], decreasing=TRUE),]

rn[,3] %>% 
  data.frame() %>%
  rownames_to_column(var = "Feature") %>%
  rename(Overall = '.') %>%
  ggplot(aes(x = fct_reorder(Feature, Overall), y = Overall)) +
  geom_pointrange(aes(ymin = 0, ymax = Overall), color = "cadetblue", size = .3) +
  theme_minimal() +
  coord_flip() +
  labs(x = "", y = "", title = "Variable Importance - MeanDecAccuracy")




# Calculate metrics for regression tree
#postResample(pred = rf_model_train_preds, obs = train_set$overall_score)
#postResample(pred = rf_model_test_preds, obs = test_set$overall_score)

# Generate the confusion matrix showing counts.
table(test_set$overall_score, rf_model_preds,
      useNA="ifany",
      dnn=c("Actual", "Predicted"))


confusionMatrix(rf_model_train_preds, train_set$top20percent)
confusionMatrix(rf_model_test_preds, test_set$top20percent)

# Generate the confusion matrix showing proportions.
pcme <- function(actual, cl)
{
  x <- table(actual, cl)
  nc <- nrow(x) # Number of classes.
  nv <- length(actual) - sum(is.na(actual) | is.na(cl)) # Number of values.
  tbl <- cbind(x/nv,
               Error=sapply(1:nc,
                            function(r) round(sum(x[r,-r])/sum(x[r,]), 2)))
  names(attr(tbl, "dimnames")) <- c("Actual", "Predicted")
  return(tbl)
}
per <- pcme(test_set$overall_score, rf_model_preds)
round(per, 2)

# Calculate the overall error percentage.

cat(100*round(1-sum(diag(per), na.rm=TRUE), 2))



#Adaboost
set.seed(seed)
ada_model <- ada::ada(top20percent ~ .,
                data=train_set,
                control=rpart::rpart.control(maxdepth=3))

ada_model_train_preds <- predict(ada_model, train_set)
ada_model_train_preds

ada_model_test_preds <- predict(ada_model, test_set)

confusionMatrix(ada_model_train_preds, train_set$top20percent)
confusionMatrix(ada_model_test_preds, test_set$top20percent)

ada::varplot(ada_model)



# Calculate the Area Under the Curve (AUC).


roc_rf = pROC::roc(test_set$top20percent, rf_model_test_preds_prob[,2])
plot(roc_rf, col="blue")


rpart_model_test_preds_prob <- predict(rpart_model, test_set, type='prob')

roc_rpart = pROC::roc(test_set$top20percent, rpart_model_test_preds_prob[,2])
plot(roc_rpart, col="orange", add = TRUE)


ada_model_test_preds_prob <- predict(ada_model, test_set, type='prob')
roc_ada = pROC::roc(test_set$top20percent, ada_model_test_preds_prob[,2])
plot(roc_ada, col="red", add = TRUE)
