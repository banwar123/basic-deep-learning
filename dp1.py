import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('D:/dataset/churn_Modelling.csv')

X = dataset.iloc[:,3:13]
y = dataset.iloc[:,13]

geography = pd.get_dummies(X["Geography"],drop_first =True)
gender =  pd.get_dummies(X['Gender'],drop_first = True)

X = pd.concat([X,geography,gender],axis =1)

X = X.drop(['Geography','Gender'],axis =1)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout

classifier = Sequential()
#first layer
#classifier.add(Dense(output_dim = 6,init= 'he_uniform',activation = 'relu',input_dim=11))
classifier.add(Dense(activation="relu", input_dim=11, units=10, kernel_initializer="he_normal"))
classifier.add(Dropout(0.3))
#second layer
classifier.add(Dense(activation="relu", input_dim=11, units=20, kernel_initializer="he_normal"))
classifier.add(Dropout(0.4))
#hidden layer

#classifier.add(Dense(output_dim = 6,init = 'he_uniform',activation = 'relu'))
classifier.add(Dense(activation="relu", units=15, kernel_initializer="he_normal"))
classifier.add(Dropout(0.2))
#output layer

#classifier.add(Dense(output_dim = 1,init = 'glorot_uniform',activation ='sigmoid'))
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="glorot_uniform"))

classifier.compile(optimizer = 'Adamax',loss = 'binary_crossentropy',metrics = ['accuracy'])

model_history = classifier.fit(X_train,y_train,validation_split = 0.33,batch_size = 10,nb_epoch =100)

print(model_history.history.keys())

plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc = 'upper left')
plt.show()


plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc = 'upper left')
plt.show()


y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

from sklearn.metrics import confusion_matrix ,accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)
score = accuracy_score(y_pred,y_test)
print(score)








