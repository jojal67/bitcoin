# -*- coding: utf-8 -*-
import sys
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout 
from keras.models import load_model, model_from_json
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import math


def train(nseg):
	Xtrain=np.load("x_train.npy")
	Ytrain=np.load("y_train.npy")
	Xtest=np.load("x_test.npy")
	Ytest=np.load("y_test.npy")

	print(Xtrain.shape)

	Xtrain=Xtrain.reshape((Xtrain.shape[0],Xtrain.shape[1],1))
	Xtest=Xtest.reshape((Xtest.shape[0],Xtest.shape[1],1))


	print(Xtrain.shape)


	# create and fit the LSTM network
	model = Sequential()
	model.add(LSTM(4, input_shape=(nseg, 1)))
	#model.add(LSTM(10))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam',metrics=["accuracy"])
	model.fit(Xtrain, Ytrain, epochs=5, batch_size=1, verbose=1,validation_data=(Xtest,Ytest))


	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("trained_network.h5")




def prediction_some_point():
	Xtrain=np.load("x_train.npy")
	Ytrain=np.load("y_train.npy")
	Xtest=np.load("x_test.npy")
	Ytest=np.load("y_test.npy")

	Xtrain=Xtrain.reshape((Xtrain.shape[0],Xtrain.shape[1],1))
	Xtest=Xtest.reshape((Xtest.shape[0],Xtest.shape[1],1))


	# load json and create model
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	trained_network = model_from_json(loaded_model_json)
	# load weights into new model
	trained_network.load_weights("trained_network.h5")
	print("Loaded model from disk")


	# making predictions
	trainPredict = trained_network.predict(Xtrain)
	testPredict = trained_network.predict(Xtest)

	print(len(testPredict))

	test=[]
	Predictiontest=[]
	cpt_good=0
	n=100
	for i in range(n):
		if testPredict[i][0]>0.5:
			Predictiontest.append(1)
		if testPredict[i][0]<=0.5 and testPredict[i][0]>-0.5:
			Predictiontest.append(0)
		if testPredict[i][0]<=-0.5:
			Predictiontest.append(-1)

		if Predictiontest[i]==Ytest[i]:
			cpt_good=cpt_good+1
		
		test.append(Ytest[i])
	Predictiontest=np.array(Predictiontest)


	print("There are ",cpt_good*100/n," % of good predictions")


	plt.figure(1)
	plt.plot(test,c="r",label="real")
	plt.plot(Predictiontest,c="g",label="prediction")
	plt.legend()




def prediction_all_curve(nseg):


	# load json and create model
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	trained_network = model_from_json(loaded_model_json)
	# load weights into new model
	trained_network.load_weights("trained_network.h5")
	print("Loaded model from disk")


	prices=np.load("prices_test_all_curve.npy")
	action_list=np.load("action_test_all_curve.npy")


	list_pred=[]
	n=1500
	for i in range(n):
		vec=action_list[i:i+nseg]
		vec=vec.reshape((1,nseg,1))
		action_tpun=trained_network.predict(vec)[0][0]
		if action_tpun>0.5:
			list_pred.append(1)
		if action_tpun<=0.5 and action_tpun>-0.5:
			list_pred.append(0)
		if action_tpun<=-0.5:
			list_pred.append(-1)
	list_pred=np.array(list_pred)

	print(len(prices[nseg:n+nseg]),len(list_pred))
	

	cptb=0
	cpts=0
	plt.figure(2)
	for i in range(len(prices[nseg:n+nseg])):
		if list_pred[i]==-1:
			if cptb==0:
				plt.axvline(i,color="y",label="buy")
			if cptb>0:
				plt.axvline(i,color="y")
			cptb=cptb+1
		if list_pred[i]==1:
			if cpts==0:
				plt.axvline(i,color="c",label="sell")
			if cpts>0:
				plt.axvline(i,color="c")
			cpts=cpts+1
	plt.plot(prices[nseg:n+nseg])

	plt.legend()






def check_bad_prediction():

	Xtest=np.load("x_test.npy")
	Ytest=np.load("y_test.npy")
	Xtest=Xtest.reshape((Xtest.shape[0],Xtest.shape[1],1))


	# load json and create model
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	trained_network = model_from_json(loaded_model_json)
	# load weights into new model
	trained_network.load_weights("trained_network.h5")
	print("Loaded model from disk")


	prices=np.load("prices_test_all_curve.npy")
	action_list=np.load("action_test_all_curve.npy")


	list_pred=[]
	n=1500
	for i in range(n):
		vec=action_list[i:i+nseg]
		vec=vec.reshape((1,nseg,1))
		action_tpun=trained_network.predict(vec)[0][0]
		if action_tpun>0.5:
			list_pred.append(1)
		if action_tpun<=0.5 and action_tpun>-0.5:
			list_pred.append(0)
		if action_tpun<=-0.5:
			list_pred.append(-1)
	list_pred=np.array(list_pred)
	

	cptbad=0
	cptgood=0
	plt.figure(3)
	for i in range(len(list_pred)):
		if action_list[i+nseg]!=list_pred[i]:
			if cptbad==0:
				plt.axvline(i,color="r",label="bad")
			if cptbad>0:
				plt.axvline(i,color="r")
			cptbad=cptbad+1
		if action_list[i+nseg]==list_pred[i]:
			if cptgood==0:
				plt.axvline(i,color="g",label="good")
			if cptgood>0:
				plt.axvline(i,color="g")
			cptgood=cptgood+1

	plt.plot(prices[nseg:n+nseg])
	print("There are ",cptbad," bad predictions over ",len(list_pred)," points so ",cptbad*100/len(list_pred)," % of bad predictions")

	plt.legend()





def simulation_gain(nseg):
	
	real_price=np.load("real_prices_test_all_curve.npy")
	prices=np.load("prices_test_all_curve.npy")
	action_list=np.load("action_test_all_curve.npy")



	# load json and create model
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	trained_network = model_from_json(loaded_model_json)
	# load weights into new model
	trained_network.load_weights("trained_network.h5")
	print("Loaded model from disk")


	########################################
	#
	#   making the whole curve prediction
	#
	########################################


	list_pred=[]
	n=4000
	for i in range(n):
		vec=action_list[i:i+nseg]
		vec=vec.reshape((1,nseg,1))
		action_tpun=trained_network.predict(vec)[0][0]
		if action_tpun>0.5:
			list_pred.append(1)
		if action_tpun<=0.5 and action_tpun>-0.5:
			list_pred.append(0)
		if action_tpun<=-0.5:
			list_pred.append(-1)
	list_pred=np.array(list_pred)

	###########################
	#
	#   making the simulation
	#
	###########################
	print("################################################################################################################# ")
	print("#")
	print("#")
	print("We do the simulation over ",n," points or over ",n*5," minutes or ",n*5/60," hours, or over ",n*5/60/24," days") 
	print("#")
	print("#")
	print("################################################################################################################# ")
	print(" ")
	print(" ")
	print("################################################################################################################# ")
	print("#")
	print("#")
	print("Simu sans fees")
	print("#")
	print("#")
	print("################################################################################################################# ")
	print(" ")
	print(" ")
	budget_depart=100 # euros
	nbitcoin_local=0
	transaction_price=0.26/100
	euros_sans_fees=[]
	bitcoin_sans_fees=[]
	achatOK=0
	venteOK=1
	nbtrade=0
	for i in range(n):
		if list_pred[i]==-1 and achatOK==0:
			# We buy
			real_price_local=real_price[i+nseg]
			nbitcoin_local=budget_depart/real_price_local 
			budget_depart=0
			achatOK=1
			venteOK=0
			nbtrade=nbtrade+1
			#print("We buy and we have ",nbitcoin_local," bitcoins")
			bitcoin_sans_fees.append(nbitcoin_local)
		if list_pred[i]==1 and venteOK==0:
			# We sell
			real_price_local=real_price[i+nseg]
			budget_depart=nbitcoin_local*real_price_local 
			nbitcoin_local=0
			venteOK=1
			achatOK=0
			#print("We sell and we have ",budget_depart," Euros")
			nbtrade=nbtrade+1
			euros_sans_fees.append(budget_depart)
	print("We did ",nbtrade," trades in total")



	print("################################################################################################################# ")
	print("#")
	print("#")
	print("Simu avec fees")
	print("#")
	print("#")
	print("################################################################################################################# ")
	print(" ")
	print(" ")
	Mise=100
	budget_depart=Mise # euros
	nbitcoin_local=0
	transaction_price=0.26/100
	euros_avec_fees=[]
	bitcoin_avec_fees=[]
	achatOK=0
	venteOK=1
	nbtrade=0
	for i in range(n):
		if list_pred[i]==-1 and achatOK==0:
			# We buy
			real_price_local=real_price[i+nseg]
			nbitcoin_local_before=budget_depart/real_price_local 
			nbitcoin_local=nbitcoin_local_before - transaction_price*nbitcoin_local_before
			budget_depart=0
			achatOK=1
			venteOK=0
			nbtrade=nbtrade+1
			#print("We buy and we have ",nbitcoin_local_before," bitcoins before transaction fees and ",nbitcoin_local," after")
			bitcoin_avec_fees.append(nbitcoin_local)
		if list_pred[i]==1 and venteOK==0:
			# We sell
			real_price_local=real_price[i+nseg]
			budget_depart_before=nbitcoin_local*real_price_local 
			budget_depart = budget_depart_before - transaction_price*budget_depart_before
			nbitcoin_local=0
			venteOK=1
			achatOK=0
			#print("We sell and we have ",budget_depart_before," Euros before transaction fees and ",budget_depart," after")
			nbtrade=nbtrade+1
			euros_avec_fees.append(budget_depart)
	print("We did ",nbtrade," trades in total")



	plt.figure(4)

	ax=plt.subplot(2,1,1)
	ax.plot(euros_sans_fees,c="b",ls=":",label="sans fees")
	ax.plot(euros_avec_fees,c="b",ls="-",label="avec fees")
	ax.axhline(Mise,color="r")
	ax.axhline(1.2*Mise,color="r",ls=":")
	ax.get_xaxis().set_visible(False)
	ax.set_ylabel("Euros")

	ax1=plt.subplot(2,1,2)
	ax1.plot(bitcoin_sans_fees,c="r",ls=":")
	ax1.plot(bitcoin_avec_fees,c="r",ls="-")
	ax1.set_ylabel("# Bitcoins")
	ax1.set_xlabel("# trade")

	plt.subplots_adjust(wspace=0, hspace=0)
	ax.legend()

#####################################################################
#
#
#				Main
#
#
#####################################################################


nseg=10

#train(nseg)


#prediction_some_point()


#prediction_all_curve()


#check_bad_prediction()


simulation_gain(nseg)


plt.show()

