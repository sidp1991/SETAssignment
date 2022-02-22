# -*- coding: utf-8 -*-
"""airQuality.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HGiM6hJRqx5FEr1xx1fV-tXOvSjToiqQ

importing libraries
"""

# Commented out IPython magic to ensure Python compatibility.
# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")
import pandas.util.testing as tm

"""Loading the Dataset"""

#reading the dataset file using pandas
from google.colab import files
uploaded = files.upload()

import io
Data = pd.read_excel(io.BytesIO(uploaded['AirQualityUCI.xlsx']))

Data.head()
#give 1st 5 row command name attributes and variables

Data.tail()
#give last 5 rows of dataset

Data.shape
#shows number of rows and column in dataset

Data.describe()
#Generate descriptive statistics

Data.info()
#information about dataset

"""Data Pre Processing"""

#Checking and counting for missing data points for each column
Data.isnull().sum()

"""Deleting missing values"""

#Remove missing values
DataCleaned = Data.dropna()
DataCleaned.isnull().sum()

#Finding out the mode value for color column
ModeValueForColor = Data['CO(GT)'].mode()[0]
print('mode value for COGT column is: ', ModeValueForColor)

#Now we can see all columns have zero missing values
print(Data.isnull().sum())
print(Data.info())

Data['CO(GT)'].value_counts()
#returns object containing counts of unique values

Data.columns
#find columns name air quality is working as data frame

"""Data Pre-processing"""

Data.dtypes
#data types of each attribute

# FINDING -200 USING SIMPLE FOR LOOPS WHICH CAN ALSO BE DONE WITH VALUE_COUNTS AND THEN REPLACED
l=[]
for i in range(len(Data.columns)):
  f=Data.columns[i]
  count=0
  for j in range(len(Data[f])):
    if Data[f][j]==-200:
      count+=1
  l.append((f,count))
print("Values from each column that reads to be replaced with avg \n ",l)

num=Data._get_numeric_data()
num[num<0]=0
Data

Data['CO(GT)'].value_counts()
# i.e all -200 are replaced with 0

"""## Exploration"""

#Outlier in Dataset is just -200

#creating dataset
np.random.seed(10)
data=Data['CO(GT)']
print(data)
fig=plt.figure(figsize=(10,7))

#creating plot
plt.boxplot(data)

#show plot
plt.show()

pip install pandas-visual-analysis

from pandas_visual_analysis import VisualAnalysis
VisualAnalysis(Data)

from google.colab import output
output.enable_custom_widget_manager()

from google.colab import output
output.disable_custom_widget_manager()

#Correlation with other variables
Data.corr()

corrmat=Data.corr()
top_corr_feature=corrmat.index
plt.figure(figsize=(30,20))
#to plot heat map
g=sns.heatmap(Data[top_corr_feature].corr(),annot=True,cmap='viridis')

sns.pairplot(Data)

Data.plot(kind='scatter',x='C6H6(GT)',y='PT08.S5(O3)')
plt.show()

Data["T"].plot.hist(bins=50)

"""## Conclusion

Highest positive correaltion can be seen among T,RH.C6H6 with .97,.92 etc....

## Prediction
"""

#features
feature=Data
feature=feature.drop('Date',axis=1)
feature=feature.drop('Time',axis=1)
feature=feature.drop('C6H6(GT)',axis=1)

feature.head()

#labels
label=Data['C6H6(GT)']

label.head()

#test and train split
X_train, X_test,y_train,y_test = train_test_split(feature,label,test_size=.3)

print(X_train.shape,y_train.shape)

print(X_test.shape,y_test.shape)

lr=LinearRegression()
lr.fit(X_train,y_train)

lr.score(X_test,y_test)

y_pred=lr.predict(X_test)
y_pred

#The coefficients
print("Coefficients: \n",lr.coef_)

# Commented out IPython magic to ensure Python compatibility.
#The mean squared error
print("Mean squared of determination: %.2f"
#   %r2_score(y_test,y_pred))

# the r squared value
print('R squared value: %.2f'%r2_score(y_test,y_pred))