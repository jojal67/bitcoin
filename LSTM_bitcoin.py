# -*- coding: utf-8 -*-
import sys
import os
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout 
from keras.models import load_model
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib
from scipy.ndimage import gaussian_filter
import time
import requests
import json
import pandas as pd
#https://api.coinranking.com/v1/public/coin/:coin_id/history/:timeframe
#https://docs.coinranking.com/
"""
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import cufflinks as cf
init_notebook_mode(connected=True)
"""




def hist_price_dl(coin_id=1335,timeframe = "5y",currency = "USD"):
	'''It accepts coin_id, timeframe, and currency parameters to clean the historic coin data taken from COINRANKING.COM
	It returns a Pandas Series with daily mean values of the selected coin in which the date is set as the index'''
	r = requests.get("https://api.coinranking.com/v1/public/coin/"+str(coin_id)+"/history/"+timeframe+"?base="+currency)
	coin = json.loads(r.text)['data']['history'] #Reading in json and cleaning the irrelevant parts
	print("toto",np.shape(coin))
	df = pd.DataFrame(coin)
	df['price'] = pd.to_numeric(df['price'])
	df['timestamp'] = pd.to_datetime(df['timestamp'],unit='ms').dt.date
	df=df.groupby('timestamp').mean()['price']
	#df.to_pickle("prices_bitcoin")
	return df#df.groupby('timestamp').mean()['price']




def price_matrix_creator(data, seq_len=30):
    '''
    It converts the series into a nested list where every item of the list contains historic prices of 30 days
    '''
    price_matrix = []
    for index in range(len(data)-seq_len+1):
        price_matrix.append(data[index:index+seq_len])
    return np.array(price_matrix)

def normalize_windows(window_data):
    '''
    It normalizes each value to reflect the percentage changes from starting point
    '''
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return np.array(normalised_data)

def train_test_split_(price_matrix, train_size=0.9, shuffle=False, return_row=True):
    '''
    It makes a custom train test split where the last part is kept as the training set.
    '''
    price_matrix = np.array(price_matrix)
    #print(price_matrix.shape)
    row = int(round(train_size * len(price_matrix)))
    X_train, y_train = price_matrix[:row,:-1], price_matrix[:row,-1]
    X_test, y_test = price_matrix[row:,:-1], price_matrix[row:,-1]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    if return_row:
        return row, X_train, y_train, X_test, y_test
    else:
        X_train, y_train, X_test, y_test


def deserializer(preds, data, train_size=0.9, train_phase=False):
    '''
    Arguments:
    preds : Predictions to be converted back to their original values
    data : It takes the data into account because the normalization was made based on the full historic data
    train_size : Only applicable when used in train_phase
    train_phase : When a train-test split is made, this should be set to True so that a cut point (row) is calculated based on the train_size argument, otherwise cut point is set to 0
    
    Returns:
    A list of deserialized prediction values, original true values, and date values for plotting
    '''
    price_matrix = np.array(price_matrix_creator(ser))
    if train_phase:
        row = int(round(train_size * len(price_matrix)))
    else:
        row=0
    date = ser.index[row+29:]
    date = np.reshape(date, (date.shape[0]))
    X_test = price_matrix[row:,:-1]
    y_test = price_matrix[row:,-1]
    preds_original = []
    preds = np.reshape(preds, (preds.shape[0]))
    for index in range(0, len(preds)):
        pred = (preds[index]+1)* X_test[index][0]
        preds_original.append(pred)
    preds_original = np.array(preds_original)
    if train_phase:
        return [date, y_test, preds_original]
    else:
        import datetime
        return [date+datetime.timedelta(days=1),y_test]



def train_model(X_train,y_train):
	# LSTM Model parameters, I chose
	batch_size = 2            # Batch size (you may try different values)
	epochs = 15               # Epoch (you may try different values)
	seq_len = 30              # 30 sequence data (Representing the last 30 days)
	loss='mean_squared_error' # Since the metric is MSE/RMSE
	optimizer = 'rmsprop'     # Recommended optimizer for RNN
	activation = 'linear'     # Linear activation
	input_shape=(None,1)      # Input dimension
	output_dim = 30           # Output dimension


	model = Sequential()
	model.add(LSTM(units=output_dim, return_sequences=True, input_shape=input_shape))
	model.add(Dense(units=32,activation=activation))
	model.add(LSTM(units=output_dim, return_sequences=False))
	model.add(Dense(units=1,activation=activation))
	model.compile(optimizer=optimizer,loss=loss)


	start_time = time.time()
	model.fit(x=X_train,
		  y=y_train,
		  batch_size=batch_size,
		  epochs=epochs,
		  validation_split=0.05)
	end_time = time.time()
	processing_time = end_time - start_time
	model.save('coin_predictor.h5')









############################################################################################################
#
#
#                                                Main
#
#
#
############################################################################################################








"""
worksave="/Users/chardin/projet_nico/"
times=np.fromfile(worksave+"times")[0:1000]
prices=np.fromfile(worksave+"prices")[0:1000]
#prices=gaussian_filter(prices, sigma=2)
# plot the whole data set
plt.figure(1)
plt.plot(prices)
print(prices.shape)
price_matrix = price_matrix_creator(prices) # Creating a matrix using the dataframe
print(price_matrix.shape)
price_matrix = normalize_windows(price_matrix) # Normalizing its values to fit to RNN
print(price_matrix.shape)
row, X_train, y_train, X_test, y_test = train_test_split_(price_matrix) # Applying train-test splitting, also returning the splitting-point
train_model(X_train,y_train)
"""



worksave="/Users/chardin/projet_nico/"
times=np.fromfile(worksave+"times")
prices=np.fromfile(worksave+"prices")
plt.figure(1)
plt.plot(prices)


courbe_test=prices[5000:6000]
plt.figure(2)
plt.plot(courbe_test)


model = load_model('coin_predictor.h5')
predicted_curve=[]
X_test=courbe_test[0:30]
for i in range(len(courbe_test)):
	preds = model.predict(X_test, batch_size=2)
	predicted_curve.append()
	X_test[0:len(X_test)-1]=X_test[1:len(X_test)]
	X_test[len(X_test)-1]=preds






plt.show()



