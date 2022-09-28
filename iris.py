from sklearn.datasets import load_iris
import pandas as pd
df=load_iris()
x=df.data
y=df.target
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()
reg.fit(x_train,y_train)
prediction=reg.predict(x_test)
#bam=pd.DataFrame(y_test,prediction)
#print(bam[5:15])
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,prediction)
print(mse)