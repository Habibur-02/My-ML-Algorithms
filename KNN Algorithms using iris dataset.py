import pandas as pd 
from sklearn.datasets import load_iris
iris=load_iris()
dir(iris)
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.2)
len(X_train)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)
knn.score(X_test,y_test)

# from sklearn.matrics import confusion_matrix
from sklearn.metrics import confusion_matrix
y_pred=knn.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
cm
# pip install seaborn
import matplotlib.pyplot as plt 
%matplotlib inline
import seaborn as sn 
plt.figure(figsize=(7,5))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
