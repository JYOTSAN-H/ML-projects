

import pandas as pd
df=pd.read_csv('company.csv')
x=df.drop('salary_more_then_100k',axis='columns')
y=df['salary_more_then_100k']

#data preprocessing
from sklearn.preprocessing import LabelEncoder

L_company=LabelEncoder()
L_job=LabelEncoder()
L_degree=LabelEncoder()

x['L_company']=L_company.fit_transform(df['company'])
x['L_job']=L_job.fit_transform(df['job'])
x['L_degree']=L_degree.fit_transform(df['degree'])


xs=x.drop(['company','job','degree'],axis='columns')


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(xs,y,test_size=0.1)


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
#model.fit(x_train,y_train)
print(x_train[1:15])
#prediction=model.predict(x_test)

#bam=pd.DataFrame(y_test,prediction)
#print(bam)