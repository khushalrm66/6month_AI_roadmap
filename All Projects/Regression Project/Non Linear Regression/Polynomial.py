import numpy as np
import matplotlib.pyplot as plt
import  pandas as pd 


dt=pd.read_csv(r"C:\Users\khush\Desktop\NIT pr\New Batch\Regression\Non Linear Regression\emp_sal (1).csv")


x= dt.iloc[:, 1:2].values
y= dt.iloc[:,2]
#------------------------------------------------------------------
#Linear Model Prediction for Example 
from sklearn.linear_model import LinearRegression
lin_reg= LinearRegression()
lin_reg.fit(x,y) 

#Linear Graph 
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title('Linear Regression graph')
plt.xlabel('Position level')
plt.ylabel('salary')
plt.show()


lin_model_pred = lin_reg.predict([[6.5]]) #Linear Regression Model Prediction 
print(lin_model_pred)

#--------------------------------------------------------------------

# Polynomial Model

from sklearn.preprocessing import PolynomialFeatures

poly_reg= PolynomialFeatures(degree=5)  #HyperParameter Tunning the Degree are 5

X_poly=poly_reg.fit_transform(x)

poly_reg.fit(X_poly,y)

lin_reg2 = LinearRegression() # Linear Model with degree 2
lin_reg2.fit(X_poly, y)


#---------------------------

plt.scatter(x,y,color='red')
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)), color='blue')

plt.title('Truth or bluff(Polynomial regression)')
plt.xlabel('Position level')
plt.ylabel('salary')
plt.show()


#-----------------------------

lin_model_pred =lin_reg.predict([[6.5]])
lin_model_pred

poly_model_pred=lin_reg2.predict(poly_reg.fit_transform([[6.5]])) #linear Model Prediction with Degree 2 
poly_model_pred


#--------------------------------

# SVR 

