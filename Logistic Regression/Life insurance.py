import pandas as pd
import numpy as np
import math
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
df=pd.read_csv('/kaggle/input/logisticregresion/insurance_data.csv')
df
plt.scatter(df.age,df.bought_insurance,marker='+',color='g')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['age']],df.bought_insurance,train_size=0.8)
X_test
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
model.predict([[54]])
