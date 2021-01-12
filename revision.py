import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()

X = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.4)


from sklearn.linear_model import LinearRegression
ln = LinearRegression()
ln.fit(X_train,y_train)
ln.score(X_test,y_test)

y_pred = ln.predict(X_test)
print(y_pred)
plt.scatter(y_test,y_pred)
plt.plot(X_train,y_train)
plt.show()



from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
knn.fit(X_test,y_test)

knn.score(X_train,y_train)
knn.score(X_test,y_test)


from sklearn.tree import DecisionTreeClassifier
dlt = DecisionTreeClassifier(max_depth=1)
dlt.fit(X_train,y_train)
dlt.fit(X_test,y_test)

dlt.score(X_train,y_train)
dlt.score(X_test,y_test)
'''

from sklearn.linear_model import LogisticRegression
lg_reg = LogisticRegression()
lg_reg.fit(X_train,y_train)

y_pred = lg_reg.predict(X_test)
print(y_pred)
lg_reg.score(X_train,y_train)
lg_reg.score(X_test,y_test)

plt.scatter(y_test,y_pred)
plt.plot(y_train,y_train)
plt.show()
'''



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

from sklearn.metrics import precision_score,recall_score,f1_score
ps = precision_score(y_test,y_pred)
rs  = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
print(f1)
print(rs)
print(ps)





