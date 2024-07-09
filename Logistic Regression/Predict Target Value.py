from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
%matplotlib inline
import matplotlib.pyplot as plt
digits = load_digits()
len(digits)
plt.gray()
# for i in range(9):
#   plt.matshow(digits.images[i])
print(dir(digits))
digits.data[1]
digits.target[1]
reg=LogisticRegression()
X_train,X_test,Y_train,Y_test=train_test_split(digits.data,digits.target, test_size=0.2)
reg.fit(X_train,Y_train)
reg.predict([digits.data[12]])
reg.predict(digits.data[8:12])

plt.matshow(digits.images[988]) #just a random value show
