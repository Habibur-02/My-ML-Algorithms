import pandas as pd  
import numpy as np
df=pd.read_csv("C:\\Users\\User\\Documents\\Machine Learning CSV File\\spam.csv")
df

df['spam']=df['Category'].apply(lambda x: 1 if x=='spam' else 0)
df
df=df.drop(['Category'],axis='columns')
df
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
clf=Pipeline(

[
('vectorizer' , CountVectorizer()),
('nb' , MultinomialNB())
]
)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df.Message,df.spam,test_size=0.2)
len(X_test)
email=[

'Hei assif, how are you?',
'Upto 20% bonus for you'

]
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
clf.predict(email)
