import pandas as pd
from sklearn import linear_model
df = pd.read_csv('one.csv')
reg=linear_model.LinearRegression()
reg.fit(df[['face','height','age','rich','humor']],df.Girl)
face=input('what is your face score /10:')
height=input('whaty is your height?:')
age=input('what is your age?:')
rich=input('how rich are you/100?:')
humor=input('what is your funny rate /100:')
print(reg.predict([[face,height,age,rich,humor]]))