import pandas as pd
from sklearn import linear_model
df=pd.read_csv('hiring.csv')
print(df)
reg=linear_model.LinearRegression()
reg.fit(df[['exp','score','interview']],df.salary)
print(reg.predict([[12,10,10]]))