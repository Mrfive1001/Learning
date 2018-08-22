from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
import pickle

with open(r'mydata.pkl', 'rb') as f:
    x_train, y_train, x_test = pickle.load(f)

model = Sequential()

model.add(Dense(input_dim=x_train.shape[1], output_dim=500))
model.add(Activation('sigmoid'))

model.add(Dense(output_dim=500))
model.add(Activation('sigmoid'))

model.add(Dense(output_dim=1))
model.add(Activation('softmax'))

model.compile(loss='mse', optimizer=SGD(lr=0.1), metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=20, epochs=300)
y_pre = model.predict(x_test)

with open('ypre.pkl', 'wb') as f:
    pickle.dump(y_pre, f)
