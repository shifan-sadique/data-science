from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score

data=pd.read_csv("mallcust.csv")
x=data[['Annual Income (k$)']]
y=data[['Spending Score (1-100)']]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)

from sklearn import linear_model
lr=linear_model.LinearRegression()

lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)

mean_squared_error(y_test,y_pred)

r2_score(y_test,y_pred)

import matplotlib.pyplot as plt
plt.scatter(x_test,y_test)
plt.plot(x_test,y_pred)

