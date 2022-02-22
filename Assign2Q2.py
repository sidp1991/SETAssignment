import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from google.colab import files
uploaded = files.upload()
# https://archive.ics.uci.edu/ml/datasets/Air+Quality dataset
df = pd.read_csv('AirQualityUCI.csv')
df.head()
x=df.iloc[:,2].values
y=df.iloc[:,3].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3,
random_state = 0)
x_test=x_test.reshape(-1,1)
x_train=x_train.reshape(-1,1)

lin_reg=linear_model.LinearRegression()
lin_reg.fit(x_train,y_train)
lin_reg_pred=lin_reg.predict(x_test)
print("Coefficients:\n",lin_reg.coef_)
print("Intercept:\n",lin_reg.intercept_)
print("Mean squared error: %.2f"

% mean_squared_error(y_test, lin_reg_pred))


# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, lin_reg_pred))
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_test, lin_reg_pred, color = 'blue')
plt.title('Temperature vs Humidity(Test set)')
plt.xlabel('Temperature')
plt.ylabel('Relative Humidity')
plt.show()