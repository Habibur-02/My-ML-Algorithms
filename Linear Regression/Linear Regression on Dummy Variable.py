import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
%matplotlib inline
import math
df=pd.read_csv('/kaggle/input/allvariable/homepricesdummyvariable.csv')
print(df)
dummy=pd.get_dummies(df.town)
dummy
dummy=dummy.drop(["west windsor"],axis='columns')
dummy
df=df.drop(['town'],axis='columns')
df
df=pd.concat([df,dummy],axis='columns')
df
y=df.price
x=df.drop(['price'],axis='columns')
print(x)
y
model=linear_model.LinearRegression()
model.fit(x,y)

model.predict([[2800,0,1]])
model.score(x,y)

