import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error,r2_score

df=datasets.load_diabetes()
df['feature_names']

x,y=datasets.load_diabetes(return_X_y=True)
x.shape

x=x[:,:1]
x.shape

x_train=x[:-20]
x_test=x[-20:]

y_train=y[:-20]
y_test=y[-20:]

regr=linear_model.LinearRegression()
regr.fit(x_train,y_train)

y_pred=regr.predict(x_test)

print(regr.coef_)
print(mean_squared_error(y_test,y_pred))
print(r2_score(y_test,y_pred))

plt.scatter(x_test,y_test)
plt.plot(x_test,y_pred)

x,y=datasets.load_diabetes(return_X_y=True)
x=x[:,[0,2]]

x_train=x[:-20]
x_test=x[-20:]

y_train=y[:-20]
y_test=y[-20:]

regr=linear_model.LinearRegression()
regr.fit(x_train,y_train)

ypred=regr.predict(x_test)


print(regr.coef_)
print(regr.intercept_)
print(mean_squared_error(y_test,y_pred))
print(r2_score(y_test,y_pred))
