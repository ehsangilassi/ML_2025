from __future__ import print_function
from pandas import read_csv
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from sklearn.preprocessing import Normalizer
from keras import callbacks
from keras.callbacks import CSVLogger
from utility import DATA_PATH, BASE_DIR

train_data = read_csv(DATA_PATH + 'train_data_numeric.csv', header=0)
test_data = read_csv(DATA_PATH + 'test_data_numeric.csv', header=0)


X = train_data.iloc[:, 1:42]
Y = train_data.iloc[:, 0]
C = test_data.iloc[:, 0]
T = test_data.iloc[:, 1:42]

scaler = Normalizer().fit(X)
trainX = scaler.transform(X)

scaler = Normalizer().fit(T)
testT = scaler.transform(T)

y_train = np.array(Y)
y_test = np.array(C)

# reshape input to be [samples, time steps, features]
X_train = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
X_test = np.reshape(testT, (testT.shape[0], 1, testT.shape[1]))

batch_size = 32
print(X_train.shape)

# 1. define the network
model = Sequential()
model.add(LSTM(4, input_shape=(1, 41)))
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('sigmoid'))
print(model.get_config())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpointer = callbacks.ModelCheckpoint(filepath=BASE_DIR + "/checkpoint-{epoch:02d}.keras", verbose=1,
                                         save_best_only=True, monitor='val_acc', mode='max')
csv_logger = CSVLogger(BASE_DIR + '/lstm_train_analysis.csv', separator=',', append=False)

model.fit(X_train, y_train, batch_size=batch_size, epochs=10, validation_data=(X_test, y_test),
          callbacks=[checkpointer, csv_logger])
# model.save("results/lstm1layer_model.keras")

loss, accuracy = model.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))
# y_pred = model.predict_classes(X_test)
# np.savetxt('lstm1predicted.txt', y_pred, fmt='%01d')
