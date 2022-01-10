
pacman::p_load(tidyverse, caret, corrplot,car, relaimpo, rpart, rpart.plot)

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

df_final = df_1 %>% dplyr::select(-rank, -restaurant_name, -address, -General.Location, -postal_code, -area, -cuisines, -Region, -cuisines_binned, -Postal.District)

#df_final = df_1 %>% dplyr::select(-rank, -restaurant_name, -address, -Postal.District, -General.Location, -postal_code, -area, -cuisines)

#df_final = df_1 %>% dplyr::select(-rank, -restaurant_name, -address, -Postal.District, -General.Location, -postal_code, -area, -cuisines, -cuisines_binned, -General.Location)



str(df_final)
colSums(is.na(df_final))
colMeans(is.na(df_final))

#remove rows with NA
df_final = df_final[complete.cases(df_final), ]
colMeans(is.na(df_final))
str(df_final)

# factorize columns

#fact_cols = c('cuisines_binned', 'General.Location', 'overall_score')


#fact_cols = c('overall_score', 'Region')


#fact_cols = c('cuisines_binned', 'Region')
#fact_cols = c('overall_score', 'cuisines_binned', 'Postal.District')

#fact_cols = c('overall_score')
#df_final[,fact_cols] = lapply(df_final[,fact_cols], factor)


#apply binning
binCats <- function(x){
  if(x==1.5 | x==2)
    return('1.5-2')
  else if(x==2.5 | x==3)
    return('2.5-3')
  else if(x==3.5 | x==4)
    return('3.5-4')
  else if(x==4.5 | x==5)
    return('4.5-5')
}

df_final['overall_score'] = apply(df_final['overall_score'], 1, binCats)


df_final['overall_score'] = lapply(df_final['overall_score'], factor)

str(df_final)


train_portion = 0.8

#set initial seed for reproducibility
set.seed(123)
# create n% training, 1-n% testing
inds = createDataPartition(df_final$overall_score, p=train_portion, 
                           list=F,times=1)

# generate training data - n% of df_final
train_set = df_final[inds,]
nrow(train_set)/nrow(df_final)
str(train_set)
colMeans(is.na(train_set))

# generate testing data - 1-n% of df_final
test_set = df_final[-inds,]
nrow(test_set)/nrow(df_final)


#tune maximum tree depth
set.seed(seed)
rpartTune <- train(
                overall_score ~ ., data = train_set,
                method = "rpart2",
                tuneGrid = data.frame(maxdepth = seq(1, 10)),
                trControl = trainControl(
                  method = "cv", number = 10,
                  verboseIter = TRUE, 
            ))
plot(rpartTune)
#Max Tree Depth of 5 seems to be giving the lowest RMSE already
#Stick to 5 for classification > highest accuracy already


# Any improvement in reducing the complexity parameter below the default of 0.01
set.seed(seed)
rpartTune <- train(
  overall_score ~ ., data = train_set,
  method = "rpart",
  tuneGrid = data.frame(cp = seq(0.001, 0.01, 0.001)),
  trControl = trainControl(
    method = "cv", number = 10,
    verboseIter = TRUE, 
  ))

plot(rpartTune)
#cp of 0.003 is giving the highest accuracy already


#Train model with depth 3 for better visualization
rpart_model <- rpart(overall_score ~ .,
                     data=train_set,
                     control=rpart.control(maxdepth=3,
                                           cp=0.003))

rpart.plot(rpart_model)
print(rpart_model)


#train model for tree 1 - Base
rpart_model <- rpart(overall_score ~ .,
               data=train_set,
               control=rpart.control(maxdepth=5,
                                     cp=0.003))



#prp(rpart_model)
rpart.plot(rpart_model)
print(rpart_model)
printcp(rpart_model)

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

confusionMatrix(rpart_model_train_preds, train_set$overall_score)
confusionMatrix(rpart_model_test_preds, test_set$overall_score)

#Evaluation metrics for model
#postResample(pred = rpart_model_train_preds, obs = train_set$overall_score)
#postResample(pred = rpart_model_test_preds, obs = test_set$overall_score)

## Random Forest Model

# Tune number of variables to select each time
rf_tune <- train(
  overall_score ~ ., 
  data = train_set, 
  method = "rf",
  tuneGrid = expand.grid(mtry = 1:5),
  trControl = trainControl(
    method = "cv", number = 10,
    verboseIter = TRUE, 
  ))

plot(rf_tune)

#Very little reduction in RMSE post 4

plot(varImp(rf_tune), main="Variable Importance with Random Forest")

#before setting maxnodes
set.seed(seed)
rf_model <- randomForest::randomForest(overall_score ~ .,
                                       data=train_set, 
                                       ntree=1000,
                                       mtry=3,
                                       importance=TRUE,
                                       na.action=randomForest::na.roughfix)

#use maxnodes of 8 for binned
set.seed(seed)
rf_model <- randomForest::randomForest(overall_score ~ .,
                                 data=train_set, 
                                 ntree=1000,
                                 mtry=2,
                                 importance=TRUE,
                                 maxnodes=8,
                                 na.action=randomForest::na.roughfix)


#use maxnodes of 12 for unbinned
set.seed(seed)
rf_model <- randomForest::randomForest(overall_score ~ .,
                                       data=train_set, 
                                       ntree=1000,
                                       mtry=2,
                                       importance=TRUE,
                                       maxnodes=12,
                                       na.action=randomForest::na.roughfix)

rf_model_train_preds <- predict(rf_model, train_set)
rf_model_train_preds

rf_model_test_preds <- predict(rf_model, test_set)
rf_model_test_preds

rn <- round(randomForest::importance(rf_model), 2)
rn
rn[order(rn[,3], decreasing=TRUE),]

rn[,1]


rn[,1] %>% 
  data.frame() %>%
  rownames_to_column(var = "Feature") %>%
  rename(Overall = '.') %>%
  ggplot(aes(x = fct_reorder(Feature, Overall), y = Overall)) +
  geom_pointrange(aes(ymin = 0, ymax = Overall), color = "cadetblue", size = .3) +
  theme_minimal() +
  coord_flip() +
  labs(x = "", y = "", title = "Variable Importance - %IncMSE")




# Calculate metrics for regression tree
#postResample(pred = rf_model_train_preds, obs = train_set$overall_score)
#postResample(pred = rf_model_test_preds, obs = test_set$overall_score)

# Generate the confusion matrix showing counts.
table(test_set$overall_score, rf_model_preds,
      useNA="ifany",
      dnn=c("Actual", "Predicted"))


train_set$overall_score
rf_model_train_preds

confusionMatrix(rf_model_train_preds, train_set$overall_score)
confusionMatrix(rf_model_test_preds, test_set$overall_score)

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






#Error
set.seed(seed)
ada <- ada::ada(overall_score ~ .,
                data=train_set,
                control=rpart::rpart.control(maxdepth=6,
                                             cp=0.010000,
                                             minsplit=20,
                                             xval=10),
                iter=50)