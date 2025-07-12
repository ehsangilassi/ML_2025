from __future__ import print_function
from pandas import read_csv
from sklearn.preprocessing import Normalizer
from keras.models import Sequential
from keras.layers import Convolution1D, Dense, Dropout, Flatten, MaxPooling1D
from numpy import array, reshape
from keras import callbacks
from keras.callbacks import CSVLogger

from utility import DATA_PATH, BASE_DIR

traindata = read_csv(DATA_PATH + 'train_data_numeric.csv', header=0)
testdata = read_csv(DATA_PATH + 'test_data_numeric.csv', header=0)

X = traindata.iloc[:, 1:42]
Y = traindata.iloc[:, 0]
T = testdata.iloc[:, 1:42]
C = testdata.iloc[:, 0]

scaler = Normalizer().fit(X)
trainX = scaler.transform(X)

scaler = Normalizer().fit(T)
testT = scaler.transform(T)

y_train = array(Y)
y_test = array(C)

X_train = reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
X_test = reshape(testT, (testT.shape[0], testT.shape[1], 1))

""" CNN Model """
cnn = Sequential()
cnn.add(Convolution1D(64, 3, activation="relu", input_shape=(41, 1)))
cnn.add(MaxPooling1D(pool_size=2))
cnn.add(Flatten())
cnn.add(Dense(128, activation="relu"))
cnn.add(Dropout(0.3))
cnn.add(Dense(1, activation="sigmoid"))
print(cnn.summary())

cnn.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

checkpointer = callbacks.ModelCheckpoint(filepath=BASE_DIR + "/checkpoint-{epoch:02d}.keras", verbose=1,
                                         save_best_only=True, monitor='val_acc', mode='max')
csv_logger = CSVLogger(BASE_DIR + '/cnn_train_analysis_01.csv', separator=',', append=False)
cnn.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[checkpointer, csv_logger])

loss, accuracy = cnn.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))
