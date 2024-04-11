import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X,y = make_blobs(n_samples=100,centers=2,n_features=2,random_state=1)

scaler.fit(X)
X = scaler.transform(X)

model = Sequential()
model.add(Dense(units=4,input_dim=2,activation='relu'))
model.add(Dense(units=4,activation='relu'))
model.add(Dense(units=2,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

model.fit(X,y, epochs=100)

test_X,test_y = make_blob(n_samples=5,centers=2,n_features=2,random_state=20)
prediction = model.predict(test_X)

for i in range(5):
    print(test_X[i],"\t",prediction[i],"\t",test_y[i])
