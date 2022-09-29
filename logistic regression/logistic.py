
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('logistic.csv')
plt.scatter(df.age,df.bought_insurance,marker='+',color='red')
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df[['age']],df.bought_insurance,test_size=0.1)
from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()
reg.fit(x_train,y_train)
predict=reg.predict(x_test)
bam=pd.DataFrame(predict,y_test7)
print(bam)
