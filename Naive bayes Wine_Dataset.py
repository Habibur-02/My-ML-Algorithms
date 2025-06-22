from sklearn.datasets import load_wine
wine=load_wine()
df=pd.DataFrame(wine.data,columns=wine.feature_names)
df

wine.feature_names
df
wine.target
dir(wine)
df['target']=wine.target
df['target_names']=df['target'].apply(lambda x:wine.target_names[x])
df
# df['target']=df['target'].map({'0':'asif','2':'janina'})
# df
def function(x): #just for understand
    if x==0:
        return 'asif'
    elif x==1:
        return 'sd'
    else:
        return 'dj'
df['target']=df['target'].apply(lambda x:function(x))
df




from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(wine.data,wine.target,test_size=0.2)

len(X_test)

Gmodel=GaussianNB()
Gmodel.fit(X_train,y_train)
Gmodel.score(X_test,y_test)

Mmodel=MultinomialNB()
Mmodel.fit(X_train,y_train)
Mmodel.score(X_test,y_test)

x=[Gmodel.score(X_test,y_test),Mmodel.score(X_test,y_test)]
x
