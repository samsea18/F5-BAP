
pacman::p_load(tidyverse, caret, corrplot,car, relaimpo, rpart, rpart.plot)

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

df_final = df_1 %>% dplyr::select(-rank, -restaurant_name, -address, -General.Location, -postal_code, -area, -cuisines)

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


fact_cols = c('Region', 'cuisines_binned', 'Postal.District')
df_final[,fact_cols] = lapply(df_final[,fact_cols], factor)

str(df_final)


train_portion = 0.8

#set initial seed for reproducibility
set.seed(123)
# create 70% training, 30% testing
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


#Scenario 1 = Base model, all variables apart from Region
train_set = dplyr::select(train_set,-Region)
test_set = dplyr::select(test_set,-Region)
str(train_set)
str(test_set)


#Scenario 2 = Use Region instead of Postal.District - Region has lesser categories than Postal.District
train_set = dplyr::select(train_set,-Postal.District)
test_set = dplyr::select(test_set,-Postal.District)
str(train_set)
str(test_set)


#Scenario 3 = Drop Postal.District & cusines_binned
train_set = dplyr::select(train_set,-Postal.District, -cuisines_binned, -Region)
test_set = dplyr::select(test_set,-Postal.District, -cuisines_binned, -Region)
str(train_set)
str(test_set)


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
#cp of 0.002 is giving the lowest RMSE already


#Train model with depth 3 for better visualization
rpart_model <- rpart(overall_score ~ .,
                     data=train_set,
                     control=rpart.control(maxdepth=3,
                                           cp=0.002))

rpart.plot(rpart_model)
print(rpart_model)


#train model for regression tree 1 - Base
rpart_model <- rpart(overall_score ~ .,
               data=train_set,
               control=rpart.control(maxdepth=5,
                                     cp=0.002))


#train model for regression tree 2
rpart_model <- rpart(overall_score ~ .,
                     data=train_set,
                     control=rpart.control(maxdepth=4,
                                           cp=0.003))

#rpart for classification tree
rpart_model <- rpart(overall_score ~ .,
                     data=train_set,
                     control=rpart.control(maxdepth=4,
                                           cp=0.004))


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
#rpart_model train preds
rpart_model_train_preds <- predict(rpart_model, train_set, type = "vector")
#rpart_model_train_preds

#rpart_model test preds
rpart_model_test_preds <- predict(rpart_model, test_set, type = "vector")

#for rpart classification
rpart_model_preds <- predict(rpart_model, test_set, type = "class")
rpart_model_preds
confusionMatrix(rpart_model_preds, test_set$overall_score)


rpart_rmse <- RMSE(
  pred = rpart_model_preds,
  obs = test_set$overall_score
)
rpart_rmse


rpart_mae <- MAE(
  pred = rpart_model_preds,
  obs = test_set$overall_score
)
rpart_mae

#Evaluation metrics for model
postResample(pred = rpart_model_train_preds, obs = train_set$overall_score)

postResample(pred = rpart_model_test_preds, obs = test_set$overall_score)

## Random Forest Model

# Tune number of variables to select each time
rf_tune <- train(
  overall_score ~ ., 
  data = train_set, 
  method = "rf",
  tuneGrid = expand.grid(mtry = 1:7),
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


set.seed(seed)
rf_model <- randomForest::randomForest(overall_score ~ .,
                                 data=train_set, 
                                 ntree=1000,
                                 mtry=3,
                                 importance=TRUE,
                                 maxnodes=22,
                                 na.action=randomForest::na.roughfix)


rf_model_train_preds <- predict(rf_model, train_set)
#rf_model_train_preds

rf_model_test_preds <- predict(rf_model, test_set)
#rf_model_test_preds

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
postResample(pred = rf_model_train_preds, obs = train_set$overall_score)
postResample(pred = rf_model_test_preds, obs = test_set$overall_score)

# Generate the confusion matrix showing counts.
table(test_set$overall_score, rf_model_preds,
      useNA="ifany",
      dnn=c("Actual", "Predicted"))


confusionMatrix(rf_model_preds, test_set$overall_score)

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