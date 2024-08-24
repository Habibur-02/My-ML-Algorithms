import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split , GridSearchCV , KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
from nltk.stem import PorterStemmer
from sklearn import metrics

df=pd.read_csv('/kaggle/input/spam-mails-dataset/spam_ham_dataset.csv')
df

df=df.drop_duplicates(keep='first')
df = df.drop_duplicates(keep='first')
df
# df=df.drop(['Unnamed: 0','label'],axis='columns')
# df

x=df['text'].values
y=df['label_num'].values
# x.shape()

def lower(word):
    return word.lower()

vectorizer=CountVectorizer(ngram_range=(1,2),min_df=0.006,preprocessor=lower)
# vectorizer = CountVectorizer()
x=vectorizer.fit_transform(x)
x

plt.hist(df['label'],rwidth=1)

from collections import Counter

from imblearn.under_sampling import NearMiss
from collections import Counter
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)

print('Original dataset shape', Counter(y))

# fit predictor and target
x,y = ros.fit_resample(x, y)

print('Modified dataset shape', Counter(y))





print(Counter(y))

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
x_train


model=MultinomialNB()
model.fit(x_train,y_train)
model.score(x_test,y_test)
# p='Hi asif, how are you?'

# p=input()
# # c=p.values
# model.predict(vectorizer.transform([p]).toarray())

# y_pred_NB = model.predict(X_test)
# NB_Acc=clf.score(X_test, y_test)
print('Accuracy score= {:.4f}'.format(model.score(x_test, y_test)))


model1=SVC(kernel='linear',C=1)
model1.fit(x_train,y_train)
model1.score(x_test,y_test)


from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
iris = datasets.load_iris()

# model_params = {
#     'svm': {
#         'model': svm.SVC(gamma='auto'),
#         'params' : {
#             'C': [1,10,20],
#             'kernel': ['rbf','linear']
#         }  
#     },
#     'random_forest': {
#         'model': RandomForestClassifier(),
#         'params' : {
#             'n_estimators': [1,5,10]
#         }
#     },
#     'logistic_regression' : {
#         'model': LogisticRegression(solver='liblinear',multi_class='auto'),
#         'params': {
#             'C': [1,5,10]
#         }
#     }
# }

# scores = []

# for model_name, mp in model_params.items():
#     clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
#     clf.fit(x_train,y_train)
#     scores.append({
#         'model': model_name,
#         'best_score': clf.best_score_,
#         'best_params': clf.best_params_
#     })
    
# df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
# df

params  = {"C":[0.2,0.5] , "kernel" : ['linear', 'sigmoid'] }
cval = KFold(n_splits = 2)
model =  SVC();
TunedModel = GridSearchCV(model,params,cv= cval)
TunedModel.fit(x_train,y_train)

GridSearchCV(cv=KFold(n_splits=2, random_state=None, shuffle=False),
             estimator=SVC(),
             param_grid={'C': [0.2, 0.5], 'kernel': ['linear', 'sigmoid']})

accuracy = metrics.accuracy_score(y_test, TunedModel.predict(x_test))
accuracy_percentage = 100 * accuracy
accuracy_percentage
