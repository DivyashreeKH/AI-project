# AI-project
project to find type of sentence
import pandas as pd
import numpy as np
data=pd.read_csv("C:/Users/khdiv/OneDrive/Desktop/Ai/Labelled1Data.csv")
#print(data)
data['first_column'] = data['first_column']
data['first_column'] = data.first_column.str.split().str.get(0)
data['b'] = data['first_column'].str[1:]
print(data.b)
def impute_b(cols):
       b=cols[0]
       if b=="How":
           return 1
        elif b=="when":
           return 2
       elif b=="what":
            return 3
        else:
            return 4
data['b']=data['b'].apply(str)
data['second_column'] = data['second_column']
data['second_column'] = data.second_column.str.split().str.get(0)
data['x'] = data['second_column'].str[:-1]
a=pd.get_dummies(data['x'],drop_first=True)
print(a)
print(data)
data.drop(['first_column','second_column'],axis=1,inplace=True)
data=pd.concat([data,a],axis=1)
data.drop(['x'],axis=1,inplace=True)
print(data)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(data.iloc[:,0:1],data.iloc[:,1:],test_size=0.25,random_state=1)
print(X_train)
print(Y_train)
print(X_test)
print(Y_test)
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,Y_train)
X_test=logmodel.predict(Y_test)
print(X_test)
