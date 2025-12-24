

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset= pd.read_csv(r"C:\Users\khush\Desktop\NIT pr\New Batch\Regression\Non Linear Regression\emp_sal (1).csv")

dataset

x= dataset.iloc[:, 1:2].values
y= dataset.iloc[:,2]

from sklearn.linear_model import LinearRegression

lim_reg=LinearRegression()
lim_reg.fit(x, y)


plt.scatter(x, y, color='red')
plt.plot(x,lim_reg.predict(x), color='blue')
plt.title('Linear regression Moel')
plt.xlabel('position')
plt.ylabel('Salary')

plt.show()


lin_model_pred= lim_reg.predict

# Polynomial Regression Model --->

from sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures(degree=5)
X_pol=poly_reg.fit_transform(x)

poly_reg.fit(X_pol,y)

lin_reg_2=LinearRegression()
lin_reg_2.fit(X_pol,y)


plt.scatter(x, y, color='red')
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)) , color='blue')
plt.title('Linear regression Moel')
plt.xlabel('position')
plt.ylabel('Salary')

plt.show()


poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6]]))
poly_model_pred                                    
                                    


#Suppport Vector Regressor (SVR) Model --->



from sklearn.svm import SVR


svr_reg = SVR()

svr_model = svr_reg.fit(x,y)

svr_pred= svr_reg.predict([[6]])
print(svr_pred)





# KNN  Model  == K Nearest Neighbors --->


from sklearn.neighbors import KNeighborsRegressor

knn_reg= KNeighborsRegressor(n_neighbors=3)
knn_reg.fit(x, y) 

knn_pred= knn_reg.predict([[6]])

print(knn_pred)                                   


# Decission Tree -->

from sklearn.tree import DecisionTreeRegressor

dt_reg = DecisionTreeRegressor()

dt_reg.fit(x, y)

dt_pred= dt_reg.predict([[6]])

print(dt_pred)



#Random Forest Model --->

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(random_state=0)
rf_reg.fit(x, y)
rf_pred = rf_reg.predict([[6]])

print(rf_pred)


                                    