from __future__ import print_function
import numpy as np
from pandas import read_csv
from sklearn.preprocessing import Normalizer
from keras.models import Sequential
from keras.layers import Convolution1D, Dense, Dropout, Flatten, MaxPooling1D
import numpy as np
import h5py
from keras import callbacks
from keras.layers import LSTM, GRU, SimpleRNN
from keras.callbacks import CSVLogger
import os
from utility import DATA_PATH, BASE_DIR

np.random.seed(42)


traindata = read_csv(DATA_PATH + 'train_data_numeric.csv', header=0)
testdata = read_csv(DATA_PATH + 'test_data_numeric.csv', header=0)
# In[3]:


X = traindata.iloc[:, 1:42]
Y = traindata.iloc[:, 0]
C = testdata.iloc[:, 0]
T = testdata.iloc[:, 1:42]

scaler = Normalizer().fit(X)
trainX = scaler.transform(X)

scaler = Normalizer().fit(T)
testT = scaler.transform(T)

y_train = np.array(Y)
y_test = np.array(C)

# In[4]:


X_train = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
X_test = np.reshape(testT, (testT.shape[0], testT.shape[1], 1))

lstm_output_size = 70

cnn = Sequential()
cnn.add(Convolution1D(64, 3, activation="relu", input_shape=(41, 1)))
cnn.add(MaxPooling1D(pool_size=(2)))
cnn.add(LSTM(lstm_output_size))
cnn.add(Dropout(0.1))
cnn.add(Dense(1, activation="sigmoid"))


cnn.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

# train
checkpointer = callbacks.ModelCheckpoint(filepath=BASE_DIR + "/checkpoint-{epoch:02d}.hdf5", verbose=1,
                                         save_best_only=True, monitor='val_acc', mode='max')
csv_logger = CSVLogger(BASE_DIR + '/cnn_train_analysis_CLSTMNET.csv', separator=',', append=False)
cnn.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[checkpointer, csv_logger])
# cnn.save("results/cnn1results/cnn_model.hdf5")

loss, accuracy = cnn.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))
