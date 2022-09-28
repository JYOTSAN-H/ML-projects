import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
df=pd.read_csv('part.csv')
#print(df)
reg=linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)
#print(reg.coef_)
#print(reg.intercept_)
print(reg.predict([[3000,3,40]]))