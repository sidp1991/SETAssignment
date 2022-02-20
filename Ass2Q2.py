# **Linear Regression** - Preprocessing, Training and Testing

# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing, model_selection, metrics

# importing the dataset in the dataframe
data = pd.read_csv("/content/sample_data/energydata_complete.csv")

data.head()

# Data Exploration

# printing the number of rows and columns in the dataset
print('The number of rows in dataset are: ' , data.shape[0])
print('The number of columns in dataset are: ' , data.shape[1])

# number of null values in all columns
data.isnull().sum().sort_values(ascending = True)

from sklearn.model_selection import train_test_split

# 75% of the data is used for the training of the models and the rest is used for testing
train, test = train_test_split(data,test_size=0.25,random_state=40)

# grouping the columns based on the type for clear column management 
col_time=["date"]
col_temp = ["T1","T2","T3","T4","T5","T6","T7","T8","T9"]
col_hum = ["RH_1","RH_2","RH_3","RH_4","RH_5","RH_6","RH_7","RH_8","RH_9"]
col_weather = ["T_out", "Tdewpoint","RH_out","Press_mm_hg","Windspeed","Visibility"]
col_light = ["lights"]
col_randoms = ["rv1", "rv2"]
col_target = ["Appliances"]

# seperating dependent and independent variables 
feature_vars  = train[ col_time + col_temp + col_hum + col_weather + col_light + col_randoms ]
target_vars = train[col_target]

# checking the distribution of values in lights column
feature_vars.lights.value_counts()

# due to lot of zero enteries this column is of not much use and will be ignored in rest of the model
_ = feature_vars.drop(['lights'], axis=1 , inplace= True) ;

feature_vars.head(2)

# Data Visualization

# histogram of all the features to understand the distribution
feature_vars.hist(bins = 20 , figsize= (12,16)) ;

# focussed displots for RH_6, RH_out, Visibility, Windspeed due to irregular distribution
f, ax = plt.subplots(2,2,figsize=(12,8))
vis1 = sns.distplot(feature_vars["RH_6"],bins=10, ax= ax[0][0])
vis2 = sns.distplot(feature_vars["RH_out"],bins=10, ax=ax[0][1])
vis3 = sns.distplot(feature_vars["Visibility"],bins=10, ax=ax[1][0])
vis4 = sns.distplot(feature_vars["Windspeed"],bins=10, ax=ax[1][1])

# distribution of values in the Appliances column
f = plt.figure(figsize=(12,5))
plt.xlabel('Appliance consumption in Wh')
plt.ylabel('Frequency')
sns.distplot(target_vars , bins=10 )

# spliting training dataset into independent and dependent varibales
train_X = train[feature_vars.columns]
train_y = train[target_vars.columns]
train_X.drop(['date'], axis=1, inplace=True)

# splitting testing dataset into independent and dependent variables
test_X = test[feature_vars.columns]
test_y = test[target_vars.columns]

# columns to be removed
train_X.drop(["rv1","rv2","Visibility","T6","T9"],axis=1 , inplace=True)
test_X.drop(["rv1","rv2","Visibility","T6","T9","date"], axis=1, inplace=True)

train_X.columns

test_X.columns

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

# creating the test and training set by including the Appliances column
train = train[list(train_X.columns.values) + col_target ]
test = test[list(test_X.columns.values) + col_target ]

# creating dummy test and training set to hold scaled values
sc_train = pd.DataFrame(columns=train.columns , index=train.index)
sc_train[sc_train.columns] = sc.fit_transform(train)
sc_test= pd.DataFrame(columns=test.columns , index=test.index)
sc_test[sc_test.columns] = sc.fit_transform(test)

sc_train.head()

sc_test.head()

# removing Appliances column from the traininig and testing set
train_X =  sc_train.drop(['Appliances'] , axis=1)
train_y = sc_train['Appliances']
test_X =  sc_test.drop(['Appliances'] , axis=1)
test_y = sc_test['Appliances']

train_X.head()

train_y.head()

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

models = [
           ['RandomForest ',RandomForestRegressor()],
           ['ExtraTreeRegressor :',ExtraTreesRegressor()]
]

# running all the proposed models and updating the information in a list model_data
import time
from math import sqrt
from sklearn.metrics import mean_squared_error

model_data = []
for name,curr_model in models :
    curr_model_data = {}
    curr_model.random_state = 78
    curr_model_data["Name"] = name
    start = time.time()
    curr_model.fit(train_X,train_y)
    end = time.time()
    curr_model_data["Train_Time"] = end - start
    curr_model_data["Train_R2_Score"] = metrics.r2_score(train_y,curr_model.predict(train_X))
    curr_model_data["Test_R2_Score"] = metrics.r2_score(test_y,curr_model.predict(test_X))
    curr_model_data["Test_RMSE_Score"] = sqrt(mean_squared_error(test_y,curr_model.predict(test_X)))
    model_data.append(curr_model_data)

model_data

df = pd.DataFrame(model_data)
df

df.plot(x="Name", y=['Test_R2_Score' , 'Train_R2_Score' , 'Test_RMSE_Score'] , kind="bar", title = 'R2 Score Results' , figsize= (10,8)) ;