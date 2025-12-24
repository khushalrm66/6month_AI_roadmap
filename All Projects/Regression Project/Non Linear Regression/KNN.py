import numpy as np
import matplotlib.pyplot as plt
import  pandas as pd 


dt=pd.read_csv(r"C:\Users\khush\Desktop\NIT pr\New Batch\Regression\Non Linear Regression\emp_sal (1).csv")


x= dt.iloc[:, 1:2].values
y= dt.iloc[:,2]


#----------------------------
# KNN Regression
#----------------------
from sklearn.neighbors import KNeighborsRegressor
knn_reg= KNeighborsRegressor(n_neighbors=4 , weights='distance',p=2)
knn_reg.fit(x, y)


y_pred_knn= knn_reg.predict([[6.5]])
print(y_pred_knn)
