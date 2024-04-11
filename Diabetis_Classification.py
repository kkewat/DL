import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential

model = Sequential()

data = np.loadtxt('pima-indians-diabetes.csv',delimiter=',')

X = data[:,0:8]
y = data[:,8]

model.add(Dense(units=12,activation='relu',input_dim=8))
model.add(Dense(units=8,activation ='relu'))
model.add(Dense(units=1,activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X,y,epochs=100,batch_size=10)

_,accuracy=model.evaluate(X,y)

prediction = model.predict(X)

for i in range(5):
    print(X[i],"\t\t\t",prediction[i],"\t",y[i])
