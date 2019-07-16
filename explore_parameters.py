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
from scipy.ndimage import gaussian_filter



def annotate(minp,maxp,prices,sig,thresold_profit):	

	pricess=(prices-minp)/(maxp-minp)
	#print("minp = ",minp," maxp =  ",maxp)
	smoothed_prices=gaussian_filter(pricess, sigma=sig)
	
	first_deriv=np.gradient(smoothed_prices)
	second_deriv=np.gradient(first_deriv)
	asign = np.sign(first_deriv)
	signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
	indices_min_max=np.where(signchange==1)[0]

	cptmin=0
	cptmax=0
	action_list=np.zeros(len(smoothed_prices))
	for i in indices_min_max:
		if second_deriv[i]>0:
			# on est sur un minimum, il faut acheter
			action_list[i]=-1
			cptmin=cptmin+1
		if second_deriv[i]<0:
			# on est sur un maximum, il faut vendre
			action_list[i]=1
			cptmax=cptmax+1

	pricebuy=0
	pricesell=0
	achatOK=0
	venteOK=1
	cpt=1
	indices_buy=[]
	indices_sell=[]
	for i in range(len(action_list)):
		if action_list[i]==-1 and achatOK==0:
			pricebuy=prices[i]
			indicebuy=i
			achatOK=1
			venteOK=0
			cpt=0
		if action_list[i]==1 and venteOK==0:
			pricesell=prices[i]
			indicesell=i
			achatOK=0
			venteOK=1

		if achatOK==0 and venteOK==1 and cpt==0:
			benefit=pricesell-pricebuy
			percent_benefit=benefit*100/pricebuy
			#print("price buy = ",pricebuy," price sell = ",pricesell," benefit = ",benefit," percent benefit = ",percent_benefit)
			#print(" benefit = ",benefit," percent benefit = ",percent_benefit)
			cpt=1

			if percent_benefit>thresold_profit:
				indices_buy.append(indicebuy)
				indices_sell.append(indicesell)
	
	action_list_benefit=np.zeros(len(action_list))
	action_list_benefit[indices_buy]=-1
	action_list_benefit[indices_sell]=1

	#print("So naturally there are ",len(indices_buy), "pixels or ",len(indices_buy)*100/len(prices)," % in the min (buy) category for the benefit only")
	#print("So naturally there are ",len(indices_sell), "pixels or ",len(indices_sell)*100/len(prices)," % in the max (sell) category for the benefit only")
	#print("So naturally there are ",len(prices)-len(indices_buy)-len(indices_sell),"pixels  or ",(len(prices)-len(indices_buy)-len(indices_sell))*100/len(prices)," % in the third (do nothing) category for the benefit only")


	ncell_needed_in_each_category=int(len(prices)/3)
	nbuymissing=ncell_needed_in_each_category-len(indices_buy)
	nsellmissing=ncell_needed_in_each_category-len(indices_sell)
	#print("There are ",nbuymissing,"cells missing in the buy category")
	#print("There are ",nsellmissing,"cells missing in the sell category")

	ncell_to_add_per_side_buy=5#int(  nbuymissing/len(indices_buy)/2    )
	ncell_to_add_per_side_sell=5#int(  nsellmissing/len(indices_sell)/2    )
	#print(ncell_to_add_per_side_buy," cells need to be added around each side of a buy label")
	#print(ncell_to_add_per_side_sell," cells need to be added around each side of a sell label")

	indice_buy=np.where(action_list_benefit==-1)[0]
	indice_sell=np.where(action_list_benefit==1)[0]
	
	for i in range(len(indice_buy)):
		action_list_benefit[indice_buy[i]-ncell_to_add_per_side_buy-1:indice_buy[i]+ncell_to_add_per_side_buy+1]=-1

	for i in range(len(indice_sell)):
		action_list_benefit[indice_sell[i]-ncell_to_add_per_side_sell-1:indice_sell[i]+ncell_to_add_per_side_sell+1]=1

	# Checking if balance is OK

	indice_buy=np.where(action_list_benefit==-1)[0]
	indice_sell=np.where(action_list_benefit==1)[0]
	indice_wait=np.where(action_list_benefit==0)[0]

	#print("After balancing, there are ",len(indice_buy), " pixels or ",len(indice_buy)*100/len(prices)," % of buy labels")
	#print("After balancing, there are ",len(indice_sell), " pixels or ",len(indice_sell)*100/len(prices)," % of sell labels")
	#print("After balancing, there are ",len(indice_wait), " pixels or ",len(indice_wait)*100/len(prices)," % of wait labels")



	return action_list_benefit






def generate_ICs(ideptrain,ifintrain,ideptest,ifintest,ntraining,ntesting,sizeseg,sig,thresold_profit):
	###########################################################
	# Points are given every 5 minutes for the bitcoin prices 
	###########################################################
	print(" ")
	print(" ")
	print(" ")
	print(" ")

	print("###########################################################################")
	print(" ")
	print(" ")
	print("The length of the segment given to the CNN will be ",sizeseg," cells")	
	print("It corresponds to the last ",sizeseg*5," minutes of the bitcoin prices and the CNN will predict if you should buy (-1) or sell (1) or do nothing (0) in the next five minutes")	
	print(" ")
	print(" ")
	print("###########################################################################")
	print(" ")
	print(" ")
	print(" ")
	print(" ")


	prices=np.fromfile("prices")
	print("The original list is composed of ",len(prices)," cells")
	prices=prices[ideptrain:ifintest]
	print ("Taking ",len(prices)," points on the curve, so a total of ",len(prices)*5," Minutes or ",len(prices)*5/60," hours or ",len(prices)*5/60/24," Days")
	minp=np.min(prices)
	maxp=np.max(prices)
	pricess=(prices-minp)/(maxp-minp)
	smoothed_prices=gaussian_filter(pricess, sigma=sig)



	action_list_benefit=annotate(minp,maxp,prices,sig,thresold_profit)
	
	indice_buy=np.where(action_list_benefit==-1)[0]
	indice_sell=np.where(action_list_benefit==1)[0]
	indice_wait=np.where(action_list_benefit==0)[0]

	# Fancy plot
	minbuy=[]
	maxbuy=[]
	minbuy.append(indice_buy[0])	
	for i in range(1,len(indice_buy)):
		if indice_buy[i]!=indice_buy[i-1]+1:
			minbuy.append(indice_buy[i])
			maxbuy.append(indice_buy[i-1])
	maxbuy.append(indice_buy[len(indice_buy)-1])


	minsell=[]
	maxsell=[]
	minsell.append(indice_sell[0])	
	for i in range(1,len(indice_sell)):
		if indice_sell[i]!=indice_sell[i-1]+1:
			minsell.append(indice_sell[i])
			maxsell.append(indice_sell[i-1])
	maxsell.append(indice_sell[len(indice_sell)-1])


	plt.figure(1)
	
	for i in range(len(minbuy)):
		plt.axvspan(minbuy[i], maxbuy[i], facecolor='r', alpha=0.5)
	for i in range(len(minsell)):
		plt.axvspan(minsell[i], maxsell[i], facecolor='g', alpha=0.5)
	plt.plot(smoothed_prices,c="k",ls="-",label="smoothed")



	###########################################################
	# 		Generating training set
	###########################################################

	x_training_set=np.zeros((ntraining,sizeseg))
	y_training_set=np.zeros(ntraining)
	for i in range(ntraining):
		dep=np.random.randint(ideptrain,ifintrain-sizeseg)
		fin=dep+sizeseg
		#x_training_set[i]=action_list[dep:fin]
		#y_training_set[i]=action_list[fin]
		x_training_set[i]=action_list_benefit[dep:fin]
		y_training_set[i]=action_list_benefit[fin]


	###########################################################
	# 		Generating testing set
	###########################################################

	x_testing_set=np.zeros((ntesting,sizeseg))
	y_testing_set=np.zeros(ntesting)
	for i in range(ntesting):
		dep=np.random.randint(ideptest,ifintest-sizeseg)
		fin=dep+sizeseg
		#x_testing_set[i]=action_list[dep:fin]
		#y_testing_set[i]=action_list[fin]
		x_testing_set[i]=action_list_benefit[dep:fin]
		y_testing_set[i]=action_list_benefit[fin]


	np.save("minp.npy",minp)
	np.save("maxp.npy",maxp)
	np.save("x_train.npy",x_training_set)
	np.save("y_train.npy",y_training_set)
	np.save("x_test.npy",x_testing_set)
	np.save("y_test.npy",y_testing_set)


	np.save("real_prices_test_all_curve.npy",prices[ideptest:ifintest])
	np.save("prices_test_all_curve.npy",smoothed_prices[ideptest:ifintest])
	#np.save("action_test_all_curve.npy",action_list[ideptest:ifintest])
	np.save("action_test_all_curve.npy",action_list_benefit[ideptest:ifintest])









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







	



def simulation_gain_realistic(nseg,transaction_price,sig,thresold_profit):
	
	prices=np.fromfile("prices")[10000:16000]
	minp=np.load("minp.npy")
	maxp=np.load("maxp.npy")



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

	def check_action(action_list):
		cptfirstaction=0
		for i in range(len(action_list)):
			if action_list[i]!=0 and cptfirstaction==0:
				print(" i = ",i," premiere action = ",action_list[i])
				cptfirstaction=1
		action_list=action_list[::-1]
		cptfirstaction=0
		for i in range(len(action_list)):
			if action_list[i]!=0 and cptfirstaction==0:
				print(" i = ",i," derniere action = ",action_list[i])
				cptfirstaction=1
		
		
		



	list_pred=[]
	idep=1000
	n=2000

	for i in range(idep,idep+n):
		action_list=annotate(minp,maxp,prices[0:i+nseg],sig,thresold_profit)
		check_action(action_list)
		vec=action_list[i:i+nseg]
		vec=vec.reshape((1,nseg,1))
		action_tpun=trained_network.predict(vec)[0][0]
		if action_tpun>0.5:
			decision=1
		if action_tpun<=0.5 and action_tpun>-0.5:
			decision=0
		if action_tpun<=-0.5:
			decision=-1
		list_pred.append(decision)
		if decision!=0:
			print("prediction number ",i)
			print(vec," ",decision)
	list_pred=np.array(list_pred)

	

	cptb=0
	cpts=0
	plt.figure(2)
	for i in range(len(prices[idep+nseg:idep+n+nseg])):
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
	plt.plot(prices[idep+nseg:idep+n+nseg])

	plt.legend()



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
	euros_sans_fees=[]
	bitcoin_sans_fees=[]
	bitcoin_sans_fees.append(0)
	euros_sans_fees.append(budget_depart)
	achatOK=0
	venteOK=1
	nbtrade=0
	for i in range(n):
		if list_pred[i]==-1 and achatOK==0:
			# We buy
			real_price_local=prices[idep+i+nseg]
			nbitcoin_local=budget_depart/real_price_local 
			budget_depart=0
			achatOK=1
			venteOK=0
			nbtrade=nbtrade+1
			print("indice = ",i," We buy at ",real_price_local ," euros and we have ",nbitcoin_local," bitcoins")
			bitcoin_sans_fees.append(nbitcoin_local)
		if list_pred[i]==1 and venteOK==0:
			# We sell
			real_price_local=prices[i+nseg]
			budget_depart=nbitcoin_local*real_price_local 
			nbitcoin_local=0
			venteOK=1
			achatOK=0
			print("indice = ",i," We sell at ",real_price_local ," euros and we have ",budget_depart," Euros")
			nbtrade=nbtrade+1
			euros_sans_fees.append(budget_depart)
	print("We did ",nbtrade," trades in total")
	percent_profit_sans=((euros_sans_fees[len(euros_sans_fees)-1]/euros_sans_fees[0])-1)*100
	print("With no fees, we make ",percent_profit_sans," % of profit")


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
	euros_avec_fees=[]
	bitcoin_avec_fees=[]
	bitcoin_avec_fees.append(0)
	euros_avec_fees.append(budget_depart)
	achatOK=0
	venteOK=1
	nbtrade=0
	for i in range(n):
		if list_pred[i]==-1 and achatOK==0:
			# We buy
			real_price_local=prices[i+nseg]
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
			real_price_local=prices[i+nseg]
			budget_depart_before=nbitcoin_local*real_price_local 
			budget_depart = budget_depart_before - transaction_price*budget_depart_before
			nbitcoin_local=0
			venteOK=1
			achatOK=0
			#print("We sell and we have ",budget_depart_before," Euros before transaction fees and ",budget_depart," after")
			nbtrade=nbtrade+1
			euros_avec_fees.append(budget_depart)

	print("We did ",nbtrade," trades in total")
	percent_profit_avec=((euros_avec_fees[len(euros_avec_fees)-1]/euros_avec_fees[0])-1)*100
	print("With fees, we make ",percent_profit_avec," % of profit")





	plt.figure(3)
	ax=plt.subplot(1,1,1)
	ax.plot(euros_sans_fees,c="b",ls=":",label="sans fees")
	ax.plot(euros_avec_fees,c="b",ls="-",label="avec fees")
	ax.axhline(Mise,color="r")
	ax.text(1,1.005*Mise,"Mise de départ",color="r")
	ax.axhline(1.2*Mise,color="r",ls=":")
	ax.text(1,1.205*Mise,"20 % Profit",color="r")
	ax.axhline(1.1*Mise,color="r",ls=":")
	ax.text(1,1.105*Mise,"10 % Profit",color="r")
	
	

	ax.set_ylim(0,1.3*Mise)
	ax.set_ylabel("Euros")
	ax.set_xlabel("# trade")

	plt.subplots_adjust(wspace=0, hspace=0)
	ax.legend()


	
	




############################################################################################
#
#
#                                   Main
#
#
############################################################################################

ideptrain=0
ifintrain=10000#200

ideptest=11000#0
ifintest=16000#200

ntraining=1000
ntesting=100

sizeseg=10


sig=3
thresold_profit=0.3
transaction_price=0.26/100



#generate_ICs(ideptrain,ifintrain,ideptest,ifintest,ntraining,ntesting,sizeseg,sig,thresold_profit)


#train(sizeseg)


simulation_gain_realistic(sizeseg,transaction_price,sig,thresold_profit)









plt.show()


