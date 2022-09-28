import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn import linear_model
df=load_boston()
x=df.data
y=df.target
data=pd.DataFrame(x,columns=df.feature_names)
data['sales price']=y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42) 
reg=linear_model.LinearRegression()
reg.fit(x_train,y_train)
prediction=reg.predict(x_test)
print("actual price= "+str(y_test[0]))
print("predicted price= "+str(prediction))
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,prediction)
rmse=np.sqrt(mse)
print(rmse)

