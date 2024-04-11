import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Dense(units=2,activation='relu',input_dim=2))
model.add(Dense(units=2,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())
print("Model Weights :\n",model.get_weights())

X = np.array([[1.,1.],[1.,0.],[0.,1.],[0.,0.]])
y = np.array([0.,1.,1.,0.])

print("X :\n",X)
print("Y :\n",y)

model.fit(X,y,epochs=1000,batch_size=4)

print(model.get_weights())

print(model.predict(X,batch_size=4))
