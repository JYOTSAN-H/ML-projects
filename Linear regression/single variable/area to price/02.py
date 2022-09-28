
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
df=pd.read_csv('Book1.csv')
plt.xlabel('area')
plt.ylabel('price')
plt.plot(df[['area']],df[['price']],marker="+")
plt.show()

x=df[['area']]
y=df[['price']]
reg=linear_model.LinearRegression()
reg.fit(x,y)
area=input('how much area would you like to train?')
print(reg.predict([[area]]))
print(reg.coef_)