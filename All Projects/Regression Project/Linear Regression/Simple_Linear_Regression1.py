

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

db = pd.read_csv(r'C:\Users\khush\Downloads\Salary_Data (1).csv')


x = db.iloc[:, :-1]
y = db.iloc[:, -1]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred=regressor.predict(x_test)

plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary Vs Experience (Test Set)')
plt.xlabel('Year of experiece ')
plt.ylabel('Salary')
plt.show()

m_slope =regressor.coef_
print(m_slope)

c_intercwept = regressor.intercept_
print(c_intercwept)

y_12 = m_slope*12+c_intercwept
print(y_12)


y_20= m_slope*20+c_intercwept
print(y_20)


bias_score= regressor.score(x_train, y_train)
print(bias_score)

variance_score= regressor.score(x_test, y_test)
print(variance_score)


db.mean()

db['Salary'].mean()

db.median()
db["Salary"].median()


db.mode()
db['Salary'].mode()

db.var()
db['Salary'].var()


db.std()

from scipy.stats import variation

variation(db.values)

variation(db['Salary'])

#Correlation
db.corr()

db['Salary'].corr(db['YearsExperience'])

#Skewness
db.skew()
db['Salary'].skew()


#Standard Error
db.sem()

#Z-Score

import scipy.stats as stats

db.apply(stats.zscore)

stats.zscore(db['Salary'])



#SSR
y_mean = np.mean(y)
SSR=np.sum((y_pred-y_mean)**2)
print(SSR)

#SSE
y=y[0:6]
SSE= np.sum((y-y_pred)**2)
print(SSE)

#SST
mean_total=np.mean(db.values)
SST=np.sum((db.values-mean_total)**2)
print(SST)


#R2
r_square= 1-SSR/SST
print(r_square)


bias=regressor.score(x_train, y_train)
print(bias)

variance=regressor.score(x_test, y_test)
print(variance)

import pickle
filename='Linear_regression_model.plkl'
with open(filename,'wb')as file:
    pickle.dump(regressor, file)
print("Model has been pickled ")    

import os
os.getcwd()



