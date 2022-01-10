#Steps----
#Scatter plot
#Check Strength of Correlation and Test Significance
#Fit Linear Regression
#Check VIF > 10
#Goodness of fit
#Residuals normality check
setwd("~/NUS Y1S2/Practice Module Project")
load('MLR.RData')
options(scipen = 100) 

pacman::p_load(tidyverse, caret, corrplot,car, relaimpo)
df<-read.csv("restaurant_list.csv",na.strings = c('?'),fileEncoding="UTF-8-BOM")

str(df)

#find the % of NA for every column, this is equivalent to sum(is.na(df$atmosphere_ratings))/nrow(df)
colMeans(is.na(df))

#we exclude those variables with > 30% NAs
# special_diets, price_range, price_range_max, price_range_min - 16:19
#we also exclude rank (target) -1 
#exclude restaurant_name, address, General.Location, Region, cuisines, postal code, area - 2:6, 15, 20

df_final<-subset(df,select=-c(1,2:6,15:20))
colMeans(is.na(df_final))

str(df_final)

#transform Postal.District to factor
df_final$Postal.District=as.factor(df_final$Postal.District)
#df_final$postal_code=as.factor(df_final$postal_code)

str(df_final)

df_final<-df_final %>% drop_na()



#inspect distribution of target variable: overall_score
ggplot(df_final,aes(overall_score)) + geom_histogram() 

ggplot(df_final,aes(overall_score))+geom_boxplot(outlier.colour='black', outlier.shape=16, outlier.size=2)

#step 1----
#transform Y
ggplot(df_final,aes((overall_score)^3)) + geom_histogram()

ggplot(df_final,aes((overall_score)^3))+geom_boxplot(outlier.colour='black', outlier.shape=16, outlier.size=2)
#after applying power 3 to Y, the distribution is now symmetrical


#step 2----

corrplot(cor(df_final[, sapply(df_final, is.numeric)],
                       use="complete.obs"), method = "number", type='lower')
#food_ratings, service_ratings, value_ratings have strong correlation with Y (ideal)
#however service_ratings have strong correlation with food_ratings, and value_ratings also have strong correlation with food_ratings
#we will dive deeper into multi-collinearility using VIF check later

#step 3----
#as there are only 7 maximum Xs, let's add all into full_model and test for VIF
full_model=lm((overall_score)^3~.,data=df_final)
vif(full_model)
summary(full_model)

#set initial seed for reproducibility
set.seed(123)

# create 70% training, 30% testing > change to 80/20
inds = createDataPartition(df_final$overall_score, p=0.8, 
                           list=F,times=1)

# generate training data - 70% of df_final
train_set = df_final[inds,]
nrow(train_set)/nrow(df_final)

# generate testing data - 30% of df_final
test_set = df_final[-inds,]
nrow(test_set)/nrow(df_final)

#we can drop cuisines and postal district, as there are too many categories and will have minimum impact

full_model_revised=lm((overall_score)^3~. - cuisines_binned -Postal.District,data=train_set)

#step 4----
vif(full_model_revised)
summary(full_model_revised)
formula(full_model_revised)

#use stepwise regression for model selection

base_model=lm((overall_score)^3~1,data=train_set) #use train data instead of df final----

#full_model=lm((overall_score)^3~.,data=df_final)
step_model=step(base_model,
                scope=list(lower=base_model,upper=full_model_revised),
                trace=FALSE)

vif(step_model)

#step 5----
formula(step_model)
summary(step_model)
#adjusted R-squared 60.15%

#step 6----

plot(step_model)

# check multicollinearity
vif(step_model)

# relative importance of various predictors----

imp = calc.relimp(step_model, type = c("lmg"),rela = TRUE) 

(imp = data.frame(lmg = imp$lmg, 
                  vars = names(imp$lmg),
                  row.names =NULL))

imp %>%
  ggplot(aes(x = reorder(vars, -lmg), y = lmg)) + 
  geom_bar(stat='identity')
# *we learnt that service and food are the two most important factors*

predictTest = predict(step_model, newdata=test_set)

predictTest

# results of test data----
R2(predictTest, test_set$overall_score)
RMSE(predictTest, test_set$overall_score)
#model is 64.9% accurate

predictTrain = predict(step_model, newdata=train_set)

R2(predictTrain, train_set$overall_score)
RMSE(predictTrain, train_set$overall_score)


save.image(file='MLR.RData')