import pandas as pd
import numpy as np
from sklearn import datasets,linear_model
from sklearn.model_selection import train_test_split
df=datasets.load_iris()
x=df.data
y=df.target
data=pd.DataFrame(x,columns=df.feature_names)
data['flower']=y
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.2)
reg=linear_model.LinearRegression()
reg.fit(x_train,y_train)
prediction=reg.predict(x_test)
print("actual price= "+str(y_test[0]))
print("predicted price= "+ str(prediction[0]))
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,prediction)
print(mse)
