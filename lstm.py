import gc
import datetime
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import keras
# from keras.models import Sequential
# from keras.layers import Activation, Dense
# from keras.layers import LSTM
# from keras.layers import Dropout



print("Loading data...")
print('Print complete data head')
complete_data = pd.read_csv('export-EtherPrice.csv')
print(complete_data.head())
print('print only dates head')
complete_date = pd.read_csv('export-EtherPrice.csv').iloc[:,0]
print(complete_date.head())
print("Data loaded!")

# Step 2. Creating Training and Test Data

#Setting The training set ratio
training_ratio = 80

#Calculating the test set ratio
test_ratio = 100-training_ratio

#Rounding the training set length to avoid fractions
training_len = round(len(complete_data)*(training_ratio/100))

#Setting the Test set length
test_len = round(len(complete_data)-training_len)

#Splitting the data based on the calculated lengths
dataset_train = complete_data.tail(training_len)
dataset_test = complete_data.head(test_len)

#Printing the shapes of training and test sets

print("Shape Of Training Set :", dataset_train.shape)
print("Shape Of Test Set :", dataset_test.shape)

print(dataset_test.tail(10))
print(dataset_train.head(10))

# 3.2 Scaling and Sequencing¶

# A method to preprocess the data in to sequences and to return x and y 

#Initializing the MinMaxScaler object
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))

def bit_pre_process(raw_data , seq_len, column = 1):
  
  #Select the feature/column 
  data = raw_data.iloc[:, column].values
  data = data.reshape(-1, 1)
  
  #Feature Scaling
  data = sc.fit_transform(data)
  
  #Making sequences
  
  X = []
  y = []

  for i in range(seq_len, len(data)):
      X.append(data[i-seq_len:i, 0])
      y.append(data[i, 0])
  X, y = np.array(X), np.array(y)

  # Reshaping
  X = np.reshape(X, (X.shape[0], X.shape[1], 1))
  
  return X, y

#Setting the sequence length (Try different values)
sequence_length = 60

#Choosing the idex of the Close column
comumn_index= 1

#Preprocessing the training set
X_train, y_train = bit_pre_process(dataset_train , sequence_length, comumn_index)

print(X_train.shape)
print(y_train.shape)

#Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Initialising the RNN
regressor = Sequential()

#Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

#Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#Adding the output layer
regressor.add(Dense(units = 1))

#Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mse', 'mae', 'mape', 'cosine'])

#Fitting the RNN to the Training set and training the RNN
regressor.fit(X_train, y_train, epochs = 30, batch_size = 90)

print(dataset_test.head())
print(dataset_test.shape)
x_test, y_true = bit_pre_process(dataset_test , sequence_length, comumn_index)
print(x_test.shape)
y_true.shape
print(y_true.shape)
predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
real_stock_price = sc.inverse_transform(y_true.reshape(-1, 1))

def plot_predictions(real_price, predicted_price, title, x_label, y_label):
  plt.plot(real_price, color = 'green', label = 'Real Stock Price')
  plt.plot(predicted_price, color = 'red', label = 'Predicted Stock Price')
  plt.title(title)
  plt.xlabel('Time')
  plt.ylabel('Ethereum Stock Price')
  plt.legend()
  plt.show()
  
#   Plotting real_stock_pric vs predicted_stock_price
plot_predictions(real_stock_price, predicted_stock_price, "Ethereum Closing Price Prediction", "Time", "Closing Price")
