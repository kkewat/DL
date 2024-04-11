import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler

X,Y = make_regression(n_samples=100,n_features=2,noise=0.2,random_state=1)

X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_scaler.fit(X)
y_scaler.fit(Y.reshape(100,1))

X = X_scaler.transform(X)
Y = y_scaler.transform(Y.reshape(100,1))

model = Sequential()
model.add(Dense(units=4,input_dim=2,activation='relu'))
model.add(Dense(units=2,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))

model.compile(loss='mse',optimizer='adam')

print(model.summary())

model.fit(X,Y,epochs=100,verbose=0)

x_test,y_test = make_regression(n_samples=5,n_features=2,noise=0.2,random_state=18)
x_test = X_scaler.transform(x_test)
#y_test = y_scaler.transform(x_test.reshape(100,1))

prediction = model.predict(x_test)

for i in range(5):
    print(x_test[i],"\t",prediction[i])
