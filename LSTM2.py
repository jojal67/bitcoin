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



def make_bar(toolbar_width):
	# setup toolbar
	sys.stdout.write("[%s]" % (" " * toolbar_width))
	sys.stdout.flush()
	sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

def update_bar(toolbar_width,i):
	sys.stdout.write("-")
	sys.stdout.flush()
	if i==toolbar_width-1:
		sys.stdout.write("\n")



def transform_train_data(Npreviousstep,sizetrainingset,prices):
	scaler = MinMaxScaler(feature_range = (0, 1))
	prices_scaled = scaler.fit_transform(prices.reshape(-1,1))
	features_set = []  
	labels = []  
	for i in range(Npreviousstep, sizetrainingset+Npreviousstep):  

		
		a=np.random.randint(Npreviousstep,len(prices))
		features_set.append(prices_scaled[a-Npreviousstep:a,0])
		labels.append(prices_scaled[a,0])
		
		
		#features_set.append(prices_scaled[i-Npreviousstep:i,0])
		#labels.append(prices_scaled[i,0])
		
	features_set, labels = np.array(features_set), np.array(labels)  
	print(np.shape(features_set))
	print(np.shape(labels))
	features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))  
	print(np.shape(features_set))
	return scaler,features_set,labels





def transform_test_data(Npreviousstep,sizetrainset,sizetestset,ideptrainingset,prices):
	scaler = MinMaxScaler(feature_range = (0, 1))
	prices_scaled = scaler.fit_transform(prices.reshape(-1,1))
	features_set = []  
	labels = []   
	ndebuttraining=ideptrainingset#Npreviousstep+sizetrainset+ideptrainingset # 4 heures apres
	for i in range(ndebuttraining+Npreviousstep, ndebuttraining+sizetestset+Npreviousstep):  
	    features_set.append(prices_scaled[i-Npreviousstep:i,0])
	    labels.append(prices_scaled[i,0])
	features_set, labels = np.array(features_set), np.array(labels)  
	print(np.shape(features_set))
	print(np.shape(labels))
	features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))  
	print(np.shape(features_set))
	return scaler,features_set,labels





def fit_lstm(features_set, labels, x_test_set, y_test_set, nbepoch):
	model = Sequential()  
	model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))
	model.add(Dropout(0.5)) 
	model.add(LSTM(units=50, return_sequences=True))  
	model.add(Dropout(0.5))
	model.add(LSTM(units=50, return_sequences=True))  
	model.add(Dropout(0.5))
	model.add(LSTM(units=50))  
	model.add(Dropout(0.5))
	model.add(Dense(units = 1))  
	model.compile(optimizer = 'adam', loss = 'mean_squared_error') 
	model.fit(features_set, labels, epochs = nbepoch, batch_size = 32, shuffle=True,validation_data=(x_test_set, y_test_set)) 

	# saving the model
	save_dir = "/Users/chardin/projet_nico/test_1/"
	model_name = 'lstm_model.h5'
	model_path = os.path.join(save_dir, model_name)
	model.save(model_path)
	print('Saved trained model at %s ' % model_path)




def make_predictions(x_test_set,Npreviousstep,scaler,Npredictions):
	save_dir = "/Users/chardin/projet_nico/test_1/"
	model_name = 'lstm_model.h5'
	model_path = os.path.join(save_dir, model_name)
	lstm_model = load_model(model_path)


	print(np.shape(x_test_set))
	predictionslist=[]
	make_bar(Npredictions)
	for i in range(Npredictions):
		predictions = lstm_model.predict(x_test_set)
		predictionsinverted = scaler.inverse_transform(predictions) 

		#print("step = ", i," last inputs curve = ",scaler.inverse_transform(x_test_set[0,Npreviousstep-1,0].reshape(-1, 1))," prediction = ", predictionsinverted[0][0])
		predictionslist.append(predictionsinverted[0][0])
		x_test_set[0,0:Npreviousstep-1,0]=x_test_set[0,1:Npreviousstep,0]
		x_test_set[0,Npreviousstep-1,0]=predictions[0][0]
		update_bar(Npredictions,i)
	predictionslist=np.array(predictionslist)
	return predictionslist





def make_figure(times,Npreviousstep,sizetrainingset,ideptrainingset,sizetestset,prices,predictionslist):


	worksave="/Users/chardin/projet_nico/"
	pricesnonfiltered=np.fromfile(worksave+"prices")

	Ndebuttraining=0
	Nfintraining=Npreviousstep+sizetrainingset

	Ndebuttest=Npreviousstep+ideptrainingset
	Nfintest=Npreviousstep+ideptrainingset+sizetestset


	plt.figure(1)
	#plt.plot((times[Ndebuttraining:Nfintraining]-times[0])/3600,pricesnonfiltered[Ndebuttraining:Nfintraining],c="b",ls=":",label="non filtered")
	plt.plot((times[Ndebuttraining:Nfintraining]-times[0])/3600,prices[Ndebuttraining:Nfintraining],c="b",label="training set")
	plt.plot((times[Ndebuttest:Nfintest]-times[0])/3600,prices[Ndebuttest:Nfintest],c="r",label="test set")	

	plt.plot((times[Ndebuttest:Nfintest]-times[0])/3600,predictionslist,c="g",label="predictions")
	#plt.fill_between((times[50:50+Npreviousstep]-times[0])/3600, y1=3100, y2=3700,color="r",alpha=0.5)
	plt.legend()





def process_all():
	# load the data set
	worksave="/Users/chardin/projet_nico/"
	times=np.fromfile(worksave+"times")
	prices=np.fromfile(worksave+"prices")
	prices=gaussian_filter(prices, sigma=2)


	#times=np.linspace(0,2000,5000)
	#prices=np.sin(times)
	
	resolution=len(prices)

	################################
	# prepare training set
	################################
	Npreviousstep=100   # 12 = 1h
	sizetrainingset=500
	scaler,features_set,labels=transform_train_data(Npreviousstep,sizetrainingset,prices)
	#########################################
	# create a testing set to validate training 
	#########################################
	ideptrainingset=100 
	sizetestset=20
	scaler,x_test_set,y_test_set=transform_test_data(Npreviousstep,sizetrainingset,sizetestset,ideptrainingset,prices)
	################################
	# train the nnetwork
	################################
	nbepoch=100
	#fit_lstm(features_set, labels, x_test_set, y_test_set, nbepoch)
	#########################################
	# prepare single test set for prediction
	#########################################
	ideptrainingset=600 # candels we skip from the training set to construct the test set
	sizetestset=1
	scaler,x_test_set,y_test_set=transform_test_data(Npreviousstep,sizetrainingset,sizetestset,ideptrainingset,prices)
	################################
	# making predictions
	################################
	Npredictions=50#int(resolution/2) # Number of predictions to make 
	predictionslist=make_predictions(x_test_set,Npreviousstep,scaler,Npredictions)
	make_figure(times,Npreviousstep,sizetrainingset,ideptrainingset,Npredictions,prices,predictionslist)









################################################################
#
#                      	     Main
#
###############################################################

	

process_all()

#progression_bar()


plt.show()



