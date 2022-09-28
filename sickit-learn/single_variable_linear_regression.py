import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
df=pd.read_csv('canada_per_capita_income.csv')
print(df)
y_train=df[['per capita']]
x_train=df[['year']]
plt.xlabel('per capita income')
plt.ylabel('year')
plt.scatter(x_train,y_train,marker='+',color='blue')

reg=linear_model.LinearRegression()
reg.fit(x_train,y_train)
print(reg.predict([[2020]]))
plt.show()


