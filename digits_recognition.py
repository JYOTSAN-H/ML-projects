from sklearn.datasets import load_digits
import pandas as pd
import matplotlib.pyplot as plt
df=load_digits()
print(df.data[34])
# this is the simple preview for which row image  you can see through matplotlib using plt.matshow i use it as comment you can use to experiment also.
#plt.gray()
#plt.matshow(df.images[34])
#plt.show()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df.data,df.target,test_size=0.2)
from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()
reg.fit(x_train,y_train)
predict=reg.predict(df.data[5:9])
bam=pd.DataFrame(df.target[5:9],predict)
print(bam[0:4])
