import pandas as pd
df=pd.read_csv('titanic.csv')
#print(df.head(10))
f_del=['PassengerId', 'SibSp', 'Ticket', 'Cabin', 'Embarked','Parch','Pclass','Name']
df=df.drop(f_del,axis='columns')



from sklearn.preprocessing import LabelEncoder
en=LabelEncoder()
df['SEX']=en.fit_transform(df['Sex'])
df=df.drop('Sex',axis='columns')
#print(df)
x=df.drop('SEX',axis='columns')
x=df.fillna(df['Age'].mean())
y=df['Survived']

#print(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.tree import DecisionTreeClassifier
reg=DecisionTreeClassifier()
reg.fit(x_train,y_train)

predict=reg.predict(x_test)
#print(y_test)

print(reg.score(x_test,y_test))



