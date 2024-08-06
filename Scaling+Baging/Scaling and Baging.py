import pandas as pd  
import numpy as np

from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,RandomizedSearchCV,KFold
from sklearn.tree import DecisionTreeClassifier
df=pd.read_csv("C:\\Users\\User\\Documents\\Machine Learning CSV File\\heart.csv")
df
df.HeartDisease.value_counts()

df.isnull().sum()

df=pd.get_dummies(df)
df
target=df.HeartDisease
df.drop(['HeartDisease'],axis='columns',inplace=True)
df

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df,target,test_size=0.2)
len(X_test)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
X_scale=StandardScaler()
scale=X_scale.fit_transform(df)
scale[:3]
from sklearn.ensemble import BaggingClassifier
bag_model=BaggingClassifier(
    # base_estimator=SVC(),
    n_estimators=100,
    max_samples=0.8,
    oob_score=True,
    random_state=0

)

bag_model.fit(X_train,y_train)
bag_model.score(X_test,y_test)

svc_model=SVC(kernel='rbf',C=30,gamma='scale')
svc_model.fit(X_train,y_train)
svc_model.score(X_test,y_test)
cross0=cross_val_score(DecisionTreeClassifier(),X_train,y_train,cv=5)
cross0

cross1=cross_val_score(BaggingClassifier(),X_train,y_train,cv=5)
cross1.mean()

cross2=cross_val_score(SVC(),X_train,y_train,cv=5)
cross2
