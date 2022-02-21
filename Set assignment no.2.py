Loading...
Dataset3.ipynb
Dataset3.ipynb_
Dataset 3: Energydata_Complete Import the Libraries

[ ]
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
Importing The dataset

[ ]
dataset = pd.read_csv('energydata_complete.csv')
[ ]
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 2].values
Splitting the dataset into training set and test set

[ ]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 1/3 , random_state = 0)
fitting linear regression to training set

[ ]
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
visualising the training set results

[ ]
#plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'Yellow')
plt.title('Appliances vs Lights(Training set)')
plt.xlabel('Appliances')
plt.ylabel('Lights')
plt.show()

[ ]
#plt.scatter(X_test, y_test, color = 'blue')
plt.plot(X_train, regressor.predict(X_train), color = 'red')
plt.title('Appliances vs Lights (Test set)')
plt.xlabel('Appliances')
plt.ylabel('Lights')
plt.show()

