import pandas as pd     
import numpy as np
from sklearn.datasets import load_wine
df=load_wine()
dir(df)
df0=pd.DataFrame(df.data,columns=df.feature_names)
df0
df.target_names

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB


model_params={

  'svm':{
      
      'model':SVC(gamma='auto'),

      'params':{
          
           'C':[1,10,15,20],
           'kernel':['rbf','linear']
      }
     
      



  },

  'decison_tree':{
      
      'model':DecisionTreeClassifier(),

      'params':{
          
           'criterion':['gini','entropy'],
           'max_depth':[2,4,6,8]
      }
  },
  

'random_forest':{
'model':RandomForestClassifier(),
'params':{
'n_estimators':[1,5,10]

}
}





}

# core=[]
# for model_name,mp in model_params.items():
#     clf=RandomizedSearchCV(mp['model'],mp['params'],cv=5,return_train_score=False)
#     clf.fit(df.data,df.target)
#     score.append({

#      'model' : model_name,
#      'best_score' : clf.best_scorcv=5e_,
#      'best_params': clf.best_params_



#     })

#     df1=pd.DataFrame(score)

    
# #  score
# df1


score=[]
for model_name,mp in model_params.items():
    clf=GridSearchCV(mp['model'],mp['params'],cv=5,return_train_score=False)
    clf.fit(df.data,df.target)
    score.append(
        
    {'model':model_name,
    'best_score': clf.best_score_,
    'best_params': clf.best_params_}


    )
  
df2=pd.DataFrame(score)
df2
