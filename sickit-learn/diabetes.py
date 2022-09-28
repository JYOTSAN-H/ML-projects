from sklearn import datasets,linear_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
df=datasets.load_diabetes()

#dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module']
x=df.data
y=df.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)
reg=linear_model.LinearRegression()
reg.fit(x_train,y_train)
prediction=reg.predict(x_test)
print("actual value= "+str(y_test[6]))
print("predicted price " + str(prediction[6]))
mse = mean_squared_error(y_test,prediction)
print(mse)

bam=pd.DataFrame(y_test,prediction)
print(bam)

