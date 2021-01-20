import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

file = open('D:/sofware/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt')
lines = file.readlines()

processedList = []

for i, line in enumerate(lines):
    try:
        line = line.split(',')
        last = line[5].split(';')[0]
        last = last.strip()
        if last == '':
            break;
        temp = [line[0], line[1], line[2], line[3], line[4], last]
        processedList.append(temp)
    except:
        #print('Error at line number: ', i)
        pass
        
columns = ['user', 'activity', 'time', 'x', 'y', 'z']
data = pd.DataFrame(data = processedList, columns = columns)


df = data.drop(['user', 'time'], axis = 1).copy()
df.head()

df['activity'].value_counts()
Walking = df[df['activity']=='Walking'].head(3555).copy()
Jogging = df[df['activity']=='Jogging'].head(3555).copy()
Upstairs = df[df['activity']=='Upstairs'].head(3555).copy()
Downstairs = df[df['activity']=='Downstairs'].head(3555).copy()
Sitting = df[df['activity']=='Sitting'].head(3555).copy()
Standing = df[df['activity']=='Standing'].copy()

balanced_data = pd.DataFrame()
balanced_data = balanced_data.append([Walking, Jogging, Upstairs, Downstairs, Sitting, Standing])
balanced_data.shape

balanced_data['activity'].value_counts()

balanced_data.head()

label = LabelEncoder()
balanced_data['label'] = label.fit_transform(balanced_data['activity'])
balanced_data.head()

X = balanced_data[['x', 'y', 'z']]
y = balanced_data['label']


scaler = StandardScaler()
X = scaler.fit_transform(X)

scaled_X = pd.DataFrame(data = X, columns = ['x', 'y', 'z'])
scaled_X['label'] = y.values

scaled_X.head()

import scipy.stats as stats

Fs = 20
frame_size = Fs*4 # 80
hop_size = Fs*2 # 40


def get_frames(df, frame_size, hop_size):

    N_FEATURES = 3

    frames = []
    labels = []
    for i in range(0, len(df) - frame_size, hop_size):
        x = df['x'].values[i: i + frame_size]
        y = df['y'].values[i: i + frame_size]
        z = df['z'].values[i: i + frame_size]
        
        # Retrieve the most often used label in this segment
        label = stats.mode(df['label'][i: i + frame_size])[0][0]
        frames.append([x, y, z])
        labels.append(label)

    # Bring the segments into a better shape
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)

    return frames, labels

X, y = get_frames(scaled_X, frame_size, hop_size)

X.shape, y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2,random_state = 0,stratify= y)

X_train.shape, X_test.shape
X_train[0].shape, X_test[0].shape


X_train = X_train.reshape(425, 80, 3, 1)
X_test = X_test.reshape(107, 80, 3, 1)

X_train[0].shape, X_test[0].shape

'''
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
'''
model = Sequential()
model.add(Conv2D(16, (2, 2), activation = 'relu', input_shape = X_train[0].shape))
model.add(Dropout(0.2))

model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(6, activation='softmax'))

model.compile(optimizer=Adam(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model_history = model.fit(X_train, y_train, epochs = 100,batch_size = 10, validation_data= (X_test, y_test), verbose=1)
scores = model.evaluate(X_test, y_test, verbose=0)
print(model_history.history.keys())
print(scores)

plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
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


y_pred = model.predict(X_test)
y_pred = (y_pred>0.5)
#list[y_pred[0]]


from sklearn.metrics import confusion_matrix ,accuracy_score

cm = confusion_matrix(y_test,y_pred)
plot_confusion_matrix(conf_mat=cm, class_names=label.classes_, show_normed=True, figsize=(7,7))

print(cm)
score = accuracy_score(y_pred,y_test)
print(score)




