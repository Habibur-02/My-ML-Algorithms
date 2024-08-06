import pandas as pd     
import numpy as np
from sklearn.model_selection import cross_val_score,train_test_split,GridSearchCV,RandomizedSearchCV,KFold
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
df=load_iris()
print(dir(df))
from sklearn.model_selection import GridSearchCV
clf=GridSearchCV(SVC(),{

'C':[1,10,15,20],
'kernel':['rbf','linear'],
'gamma':['scale','auto']

},cv=5,return_train_score=False)


clf.fit(df.data,df.target)
clf.cv_results_


df0=pd.DataFrame(clf.cv_results_)
df0=df0[['params','mean_test_score']]
df0


clf.best_score_
clf.best_params_



from sklearn.model_selection import RandomizedSearchCV
Rlf=RandomizedSearchCV(SVC(),{

'C':[1,10,15,20],
'kernel':['rbf','linear'],
'gamma':['scale','auto']

},cv=5,return_train_score=False,n_iter=2)


Rlf.fit(df.data,df.target)
Rlf.cv_results_
rdf=pd.DataFrame(Rlf.cv_results_)
rdf[['params','mean_test_score']]

Rlf.best_score_

model_params={

'svm':{
'model':SVC(gamma='auto'),
'params':{

'C':[1,10,15],
'kernel':['rbf','linear']

}
},


'random_forest':{
'model':RandomForestClassifier(),
'params':{
'n_estimators':[1,5,10]

}
},

'logistic_regression':{
'model':LogisticRegression(solver='liblinear',multi_class='auto'),
'params':{

'C':[1,10,15]
# 'kernel':['rbf','linear']

}

}

}


score=[]
for model_name,mp in model_params.items():
    clf=GridSearchCV(mp['model'],mp['params'],cv=5,return_train_score=False)
    clf.fit(df.data,df.target)
    score.append({

     'model' : model_name,
     'best_score' : clf.best_score_,
     'best_params': clf.best_params_



    })

    df1=pd.DataFrame(score)


#  score
df1




score=[]
for model_name,mp in model_params.items():
    clf=RandomizedSearchCV(mp['model'],mp['params'],cv=5,return_train_score=False)
    clf.fit(df.data,df.target)
    score.append({

     'model' : model_name,
     'best_score' : clf.best_score_,
     'best_params': clf.best_params_



    })

    df1=pd.DataFrame(score)

    
#  score
df1


