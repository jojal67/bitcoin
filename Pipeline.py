# -*- coding: utf-8 -*-
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout 
from keras.models import load_model, model_from_json
from keras.utils import to_categorical
from keras.utils import np_utils


def log_diff_prices(price):
	diff=[]
	diff.append(0)
	for i in range(1,len(price)):
		diff.append(np.log10(price[i]/price[i-1]))
	diff=np.array(diff)
	return diff		



def check_action(action_list):
	cptfirstaction=0
	for i in range(len(action_list)):
		if action_list[i]!=0 and cptfirstaction==0:
			cptfirstaction=1
			if action_list[i]==-1:
				first_action=-1 # buy
				#print("first action : buy ")
			else:
				first_action=1 # sell
				#print("first action : sell ")

	action_list=action_list[::-1]
	cptfirstaction=0
	for i in range(len(action_list)):
		if action_list[i]!=0 and cptfirstaction==0:
			cptfirstaction=1
			if action_list[i]==-1:
				last_action=-1 # buy
				#print("last action : buy ")
			else:
				last_action=1 # sell
				#print("last action : sell ")
	
	return first_action,last_action





def transform_curve(prices,threshold_benefit,ncell):
	
	#print ("Taking ",len(prices)," points on the curve,  so a total of ",len(prices)*5," Minutes or ",len(prices)*5/60," hours or ",len(prices)*5/60/24," Days")
	minp=np.min(prices)
	maxp=np.max(prices)
	pricess=(prices-minp)/(maxp-minp)
	smoothed_prices=gaussian_filter(pricess, sigma=3)
	

	first_deriv=np.gradient(smoothed_prices)
	minderiv=np.min(first_deriv)
	maxderiv=np.max(first_deriv)
	first_deriv_normalized=(first_deriv-minderiv)/(maxderiv-minderiv)
	

	second_deriv=np.gradient(first_deriv)
	minderiv=np.min(second_deriv)
	maxderiv=np.max(second_deriv)
	second_deriv_normalized=(second_deriv-minderiv)/(maxderiv-minderiv)

	asign = np.sign(first_deriv)
	signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
	indices_min_max=np.where(signchange==1)[0]
	#print("There are ",len(indices_min_max)," max and min")
	
	##########################################
	# Annotate a first time for min and max
	##########################################

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
	
	check_action(action_list)

	
	
	##########################################
	# Annotate a second time to make benefit
	##########################################
	
	action_list_original=np.copy(action_list)
		
	list_indice_sell=[]	
	listindiceselloriginal=np.where(action_list_original==1)[0]
	cpt=0
	while(cpt<len(listindiceselloriginal)-1):
		if prices[listindiceselloriginal[cpt+1]]>=prices[listindiceselloriginal[cpt]]:		
			while(prices[listindiceselloriginal[cpt+1]]>=prices[listindiceselloriginal[cpt]] and cpt+1<len(listindiceselloriginal)-1):
				cpt=cpt+1
			indicesell=listindiceselloriginal[cpt]
			list_indice_sell.append(indicesell)
			cpt=cpt+1
		else:
			cpt=cpt+1


	list_indice_buy=[]
	listindicebuyoriginal=np.where(action_list_original==-1)[0]
	indicebuy=0
	list_indice_buy.append(indicebuy)
	cptachat=0
	cpt=0
	while(cpt<len(listindicebuyoriginal)-1 and cptachat<len(list_indice_sell)-1):
		if listindicebuyoriginal[cpt]>list_indice_sell[cptachat]:
			if prices[listindicebuyoriginal[cpt+1]]<=prices[listindicebuyoriginal[cpt]]:
				while(prices[listindicebuyoriginal[cpt+1]]<=prices[listindicebuyoriginal[cpt]] and cpt+1<len(listindicebuyoriginal)-1):
					cpt=cpt+1
				indicebuy=listindicebuyoriginal[cpt]
				list_indice_buy.append(indicebuy)
				cptachat=cptachat+1	
				cpt=cpt+1
			else:
				indicebuy=listindicebuyoriginal[cpt]
				list_indice_buy.append(indicebuy)
				cptachat=cptachat+1	
				cpt=cpt+1
		else:
			cpt=cpt+1	
			
	action_list=np.zeros(len(smoothed_prices))
	action_list[list_indice_sell]=1
	action_list[list_indice_buy]=-1

	
	
	####################################################
	# Annotate a third time to suppress action repeated
	####################################################
	
	first_action,derniere_action=check_action(action_list)
	cpt=0
	last_action=first_action
	for i in range(len(action_list)):
		if action_list[i]!=0:
			action_local=action_list[i]
			if cpt>0 and action_local==last_action:
				action_list[i]=0
			last_action=action_local
			cpt=cpt+1
	
	
	###################################################################################################
	# Annotate a fourth time to supress couple of buy sell action below the percent_profit value
	###################################################################################################

	"""
	indice_buy=np.where(action_list==-1)[0]
	indice_sell=np.where(action_list==1)[0]
	print(len(indice_buy),len(indice_sell))
	for i in range(len(indice_buy)):
		benefit=prices[indice_sell[i]]-prices[indice_buy[i]]
		percent_benefit=benefit*100/prices[indice_buy[i]]	
		if percent_benefit<threshold_benefit:
			action_list[indice_buy[i]]=0
			action_list[indice_sell[i]]=0
		

	indice_buy=np.where(action_list==-1)[0]
	"""
	#####################################################################################################################
	# Annotate a fourth time to add ncell around the buy sell actions to unbalance data to train a future neural network  
	#####################################################################################################################
	
	indice_buy=np.where(action_list==-1)[0]
	indice_sell=np.where(action_list==1)[0]
	for i in range(len(indice_buy)):
		action_list[indice_buy[i]-ncell:indice_buy[i]+ncell]=-1
	for i in range(len(indice_sell)):
		action_list[indice_sell[i]-ncell:indice_sell[i]+ncell]=1
	
	
	return action_list











def simu_before_training(threshold_benefit,ncell,Mise,transaction_price):
	#prices=np.fromfile("prices")
	prices=np.load("prices_four_years_5_min.npy")[200:20000]
	list_pred=transform_curve(prices,threshold_benefit,ncell)

	minp=np.min(prices)
	maxp=np.max(prices)
	pricess=(prices-minp)/(maxp-minp)
	pricess=gaussian_filter(pricess, sigma=3)

	budget_depart=Mise # euros
	nbitcoin_local=0
	euros_avec_fees=[]
	bitcoin_avec_fees=[]
	euros_avec_fees.append(Mise)
	achatOK=0
	venteOK=1
	nbtrade=0
	for i in range(len(prices)):
		if list_pred[i]==-1 and achatOK==0:
			# We buy
			real_price_local=prices[i]
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
			real_price_local=prices[i]
			budget_depart_before=nbitcoin_local*real_price_local 
			budget_depart = budget_depart_before - transaction_price*budget_depart_before
			nbitcoin_local=0
			venteOK=1
			achatOK=0
			#print("We sell and we have ",budget_depart_before," Euros before transaction fees and ",budget_depart," after")
			nbtrade=nbtrade+1
			euros_avec_fees.append(budget_depart)
	profit=euros_avec_fees[len(euros_avec_fees)-1]-Mise
	percent_profit_avec=((euros_avec_fees[len(euros_avec_fees)-1]/Mise)-1)*100
	print("We did ",nbtrade," trades in total over ",len(prices)*5," minutes or ",len(prices)*5/60," hours or ",len(prices)*5/60/24," Days")
	print("La mise de depart est de ",Mise," euros et on a fait ",profit," euros de benefices, soit ",percent_profit_avec," pourcent e benefice sur notre mise de depart")
	


	plt.figure(0)
	cptmin=0
	cptmax=0
	for i in range(len(list_pred)):
		if list_pred[i]==-1:
			plt.axvline(i,color="y")
			cptmin=1
		if list_pred[i]==1:
			plt.axvline(i,color="c")
			cptmax=1
	plt.plot(pricess,c="b",ls=":",label="smoothed")
	plt.legend()
	


	plt.figure(1)
	ax=plt.subplot(2,1,1)
	ax.plot(euros_avec_fees,c="b",ls="-",label="avec fees")
	ax.axhline(Mise,color="r")
	ax.text(1,1.005*Mise,"Mise de départ",color="r")
	ax.axhline(1.1*Mise,color="r",ls=":")
	ax.text(1,1.105*Mise,"10 % Profit",color="r")
	ax.axhline(1.2*Mise,color="r",ls=":")
	ax.text(1,1.205*Mise,"20 % Profit",color="r")
	ax.axhline(1.3*Mise,color="r",ls=":")
	ax.text(1,1.305*Mise,"30 % Profit",color="r")
	ax.axhline(1.4*Mise,color="r",ls=":")
	ax.text(1,1.405*Mise,"40 % Profit",color="r")
	ax.get_xaxis().set_visible(False)
	ax.set_ylabel("Euros")
	ax1=plt.subplot(2,1,2)
	ax1.plot(bitcoin_avec_fees,c="r",ls="-")
	ax1.set_ylabel("# Bitcoins")
	ax1.set_xlabel("# trade")
	plt.subplots_adjust(wspace=0, hspace=0)
	ax.legend()


	##########################
	#   Some statistics
	##########################
	action_list=list_pred

	indice_buy=np.where(action_list==-1)[0]
	indice_sell=np.where(action_list==1)[0]
	ncell_between_two_buy=[]
	for i in range(1,len(indice_buy)):
		ncell_between_two_buy.append(indice_buy[i]-indice_buy[i-1])
	ncell_between_two_buy=np.array(ncell_between_two_buy)
	ncell_between_two_sell=[]
	for i in range(1,len(indice_sell)):
		ncell_between_two_sell.append(indice_sell[i]-indice_sell[i-1])
	ncell_between_two_sell=np.array(ncell_between_two_sell)
	ncell_between_one_buy_and_one_sell=[]
	for i in range(len(indice_sell)):
		ncell_between_one_buy_and_one_sell.append(indice_sell[i]-indice_buy[i])
	ncell_between_one_buy_and_one_sell=np.array(ncell_between_one_buy_and_one_sell)

	ncell_mean_bt_two_buy=np.average(ncell_between_two_buy)
	ncell_mean_bt_two_sell=np.average(ncell_between_two_sell)
	ncell_mean_bt_buy_sell=np.average(ncell_between_one_buy_and_one_sell)

	print("Duree moyenne entre deux achats = ",ncell_mean_bt_two_buy," cells or ",ncell_mean_bt_two_buy*5/60," heures")
	print("Duree moyenne entre deux ventes = ",ncell_mean_bt_two_sell," cells or ",ncell_mean_bt_two_sell*5/60," heures")
	print("Duree moyenne entre un achat et une vente = ",ncell_mean_bt_buy_sell," cells or ",ncell_mean_bt_buy_sell*5/60," heures")










def generate_ICs(ideptrain,ifintrain,ideptest,ifintest,ntraining,ntesting,sizeseg,thresold_profit,ncell):

	#prices=np.fromfile("prices")
	prices=np.load("prices_four_years_5_min.npy")
	action_list_benefit=transform_curve(prices,threshold_benefit,ncell)
	minp=np.min(prices)
	maxp=np.max(prices)
	pricess=(prices-minp)/(maxp-minp)
	pricess=gaussian_filter(pricess, sigma=3)
	pricess=log_diff_prices(pricess)

	volume=np.load("volume_four_years_5_min.npy")
	minv=np.min(volume)
	maxv=np.max(volume)
	volume=(volume-minv)/(maxv-minv)
	volume=gaussian_filter(volume, sigma=3)
	#volume=log_diff_prices(volume)

	###########################################################
	# 		Generating training set
	###########################################################
	x_training_set=np.zeros((ntraining,sizeseg))
	volume_training_set=np.zeros((ntraining,sizeseg))
	y_training_set=np.zeros(ntraining)
	action_train=action_list_benefit[ideptrain:ifintrain]
	pricestrain=pricess[ideptrain:ifintrain]
	volumetrain=volume[ideptrain:ifintrain]
	cpt=0
	###################################
	# get a third with buy at the end
	###################################
	indices_buy=np.where(action_train==-1)[0]
	print("len indices_buy train = ",len(indices_buy))
	cptbuy=0
	ran=0
	while cptbuy<ntraining/2:
		#ran=np.random.randint(0,len(indices_buy)-1)
		if indices_buy[ran]>sizeseg:
			vec=pricestrain[indices_buy[ran]-sizeseg:indices_buy[ran]]
			x_training_set[cpt]=vec
			vec=volumetrain[indices_buy[ran]-sizeseg:indices_buy[ran]]
			volume_training_set[cpt]=vec
			y_training_set[cpt]=0#action_train[indices_buy[ran]]
			cpt=cpt+1
			cptbuy=cptbuy+1
		ran=ran+1
	###################################
	# get a third with sell at the end
	###################################
	indices_sell=np.where(action_train==1)[0]
	cptsell=0
	ran=0
	while cptsell<ntraining/2 and cpt<ntraining:
		#ran=np.random.randint(0,len(indices_sell)-1)
		if indices_sell[ran]>sizeseg:
			vec=pricestrain[indices_sell[ran]-sizeseg:indices_sell[ran]]		
			x_training_set[cpt]=vec
			vec=volumetrain[indices_sell[ran]-sizeseg:indices_sell[ran]]
			volume_training_set[cpt]=vec
			y_training_set[cpt]=1#action_train[indices_sell[ran]]
			cpt=cpt+1
			cptsell=cptsell+1
		ran=ran+1
	"""
	###################################
	# get a third with wait at the end
	###################################
	indices_wait=np.where(action_train==0)[0]
	cptwait=0
	while cptwait<int(ntraining/3) and cpt<ntraining:
		ran=np.random.randint(0,len(indices_wait)-1)
		if indices_wait[ran]>sizeseg:
			vec=pricestrain[indices_wait[ran]-sizeseg:indices_wait[ran]]		
			x_training_set[cpt]=vec
			y_training_set[cpt]=action_train[indices_wait[ran]]
			cpt=cpt+1
			cptwait=cptwait+1
	"""

	###########################################################
	# 		Generating testing set
	###########################################################
	x_testing_set=np.zeros((ntesting,sizeseg))
	volume_testing_set=np.zeros((ntraining,sizeseg))
	y_testing_set=np.zeros(ntesting)
	action_test=action_list_benefit[ifintrain:ifintest]
	pricestest=pricess[ifintrain:ifintest]
	volumetest=volume[ifintrain:ifintest]
	cpt=0
	###################################
	# get a third with buy at the end
	###################################
	indices_buy=np.where(action_test==-1)[0]
	print("len indices_buy test = ",len(indices_buy))
	cptbuy=0
	ran=0
	while cptbuy<ntesting/2:
		#ran=np.random.randint(0,len(indices_buy)-1)
		if indices_buy[ran]>sizeseg:
			vec=pricestest[indices_buy[ran]-sizeseg:indices_buy[ran]]		
			x_testing_set[cpt]=vec
			vec=volumetest[indices_buy[ran]-sizeseg:indices_buy[ran]]
			volume_testing_set[cpt]=vec
			y_testing_set[cpt]=0#action_test[indices_buy[ran]]
			cpt=cpt+1
			cptbuy=cptbuy+1
		ran=ran+1
	###################################
	# get a third with sell at the end
	###################################
	indices_sell=np.where(action_test==1)[0]
	cptsell=0
	ran=0
	while cptsell<ntesting/2 and cpt<ntesting:
		#ran=np.random.randint(0,len(indices_sell)-1)
		if indices_sell[ran]>sizeseg:
			vec=pricestest[indices_sell[ran]-sizeseg:indices_sell[ran]]		
			x_testing_set[cpt]=vec
			vec=volumetest[indices_sell[ran]-sizeseg:indices_sell[ran]]
			volume_testing_set[cpt]=vec
			y_testing_set[cpt]=1#action_test[indices_sell[ran]]
			cpt=cpt+1
			cptsell=cptsell+1
		ran=ran+1
	"""
	###################################
	# get a third with wait at the end
	###################################
	indices_wait=np.where(action_test==0)[0]
	cptwait=0
	while cptwait<int(ntesting/3) and cpt<ntesting:
		ran=np.random.randint(0,len(indices_wait)-1)
		if indices_wait[ran]>sizeseg:
			vec=pricestest[indices_wait[ran]-sizeseg:indices_wait[ran]]		
			x_testing_set[cpt]=vec
			y_testing_set[cpt]=action_test[indices_wait[ran]]
			cpt=cpt+1
			cptwait=cptwait+1
	"""

	#########################################
	# shuffle both the training and test set
	#########################################
	
	p=np.arange(len(y_training_set))
	np.random.shuffle(p)
	x_training_set=x_training_set[p]
	volume_training_set=volume_training_set[p]
	y_training_set=y_training_set[p]

	p=np.arange(len(y_testing_set))
	np.random.shuffle(p)
	x_testing_set=x_testing_set[p]
	volume_testing_set=volume_testing_set[p]
	y_testing_set=y_testing_set[p]
	

	np.save("minp.npy",minp)
	np.save("maxp.npy",maxp)
	np.save("x_train.npy",x_training_set)
	np.save("volume_train.npy",volume_training_set)
	np.save("y_train.npy",y_training_set)
	np.save("x_test.npy",x_testing_set)
	np.save("volume_test.npy",volume_testing_set)
	np.save("y_test.npy",y_testing_set)

	
	plt.figure(1)
	for i in range(10):
		if y_training_set[i]==1:
			plt.plot(x_training_set[i])
	plt.figure(11)
	for i in range(10):
		if y_training_set[i]==1:
			plt.plot(volume_training_set[i])

	







def train(nseg,nepoch):
	Xtrain=np.load("x_train.npy")
	Vtrain=np.load("volume_train.npy")
	Ytrain=np.load("y_train.npy")
	Xtest=np.load("x_test.npy")
	Vtest=np.load("volume_test.npy")
	Ytest=np.load("y_test.npy")
	print(Ytrain[0:10])
	Ytrain = to_categorical(Ytrain,num_classes=2)
	Ytest = to_categorical(Ytest,num_classes=2)
	Xtrain=Xtrain.reshape((Xtrain.shape[0],Xtrain.shape[1],1))
	Xtest=Xtest.reshape((Xtest.shape[0],Xtest.shape[1],1))
	Vtrain=Vtrain.reshape((Vtrain.shape[0],Vtrain.shape[1],1))
	Vtest=Vtest.reshape((Vtest.shape[0],Vtest.shape[1],1))
	print(Ytrain[0:10])

	print(Xtrain.shape)
	print(Ytrain.shape)

	

	model = Sequential()
	model.add(LSTM(100, input_shape=(Xtrain.shape[1], 1)))
	model.add(Dropout(0.2))
	model.add(Dense(100, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(Ytrain.shape[1], activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(Xtrain, Ytrain, epochs=nepoch, batch_size=128, verbose=1,validation_data=(Xtest,Ytest))


	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("trained_network.h5")







def simulation_after_training(threshold_benefit,ncell,nseg,Mise):
	#prices=np.fromfile("prices")[12000:]
	prices=np.load("prices_four_years_5_min.npy")[340000:]
	minp=np.load("minp.npy")
	maxp=np.load("maxp.npy")
	minp=np.min(prices)
	maxp=np.max(prices)
	prices2=(prices-minp)/(maxp-minp)
	prices3=gaussian_filter(prices2, sigma=3)
	prices4=log_diff_prices(prices3)

	# load json and create model
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	trained_network = model_from_json(loaded_model_json)
	# load weights into new model
	trained_network.load_weights("trained_network.h5")
	print("Loaded model from disk")

	
	list_pred=[]
	nprediction=1000
	
	last_decision=0
	cpt_vente=0
	cpt_achat=0
	for i in range(nprediction):
		price_list=prices4[i:i+nseg]
		price_list=price_list.reshape((1,nseg,1))
		action_tpun=trained_network.predict(price_list)[0]
		indices_highest_proba=np.where(action_tpun==np.max(action_tpun))[0]


		"""
		############################################################
		# Strategies ou on laisse les predictions
		############################################################
		if indices_highest_proba==0:
			if action_tpun[indices_highest_proba]>0.99:
				decision=-1
			else :
				decision=0
				
		if indices_highest_proba==1:
			if action_tpun[indices_highest_proba]>0.99:
				decision=1
			else :
				decision=0
		list_pred.append(decision)
		if i%100==0:
			print("prediction number ",i)
			print(action_tpun,decision)

		"""

		
		############################################################
		# Strategies ou on achetes apres n fois la meme prediction
		############################################################
		npred=10
		if indices_highest_proba==0:
			if action_tpun[indices_highest_proba]>0.99:
				last_decision=-1
				cpt_vente=cpt_vente+1
				cpt_achat=0
				if last_decision==-1 and cpt_vente==npred: # on compte n prediction succesive de vente avant de vendre
					decision=-1
				else : 
					decision=0
			else :
				decision=0
				last_decision=0
				cpt_achat=0
				cpt_vente=0
		if indices_highest_proba==1:
			if action_tpun[indices_highest_proba]>0.99:
				last_decision=1
				cpt_achat=cpt_achat+1
				cpt_vente=0
				if last_decision==1 and cpt_achat==npred: # on compte n prediction succesive d achat avant d acheter
					decision=1
				else : 
					decision=0
			else :
				decision=0
				last_decision=0
				cpt_achat=0
				cpt_vente=0
		list_pred.append(decision)
		if i%100==0:
			print("prediction number ",i)
			print(action_tpun,decision)
		


	list_pred=np.array(list_pred)
	action_list_benefit=transform_curve(prices[nseg:nseg+nprediction],threshold_benefit,ncell)	

	plt.figure(3)
	for i in range(len(list_pred)):
		if list_pred[i]==-1:
			plt.axvline(i,color="y")
			cptmin=1
		if list_pred[i]==1:
			plt.axvline(i,color="c")
			cptmax=1
		if action_list_benefit[i]==-1:
			plt.axvline(i,color="g",ls=":",linewidth=5)
		if action_list_benefit[i]==1:
			plt.axvline(i,color="b",ls=":",linewidth=5)
	plt.plot(prices3[nseg:nseg+nprediction],c="r")
	

	######################
	# calcul du benefice
	######################
	prices=prices[nseg:nseg+nprediction]
	budget_depart=Mise # euros
	nbitcoin_local=0
	euros_avec_fees=[]
	bitcoin_avec_fees=[]
	euros_avec_fees.append(Mise)
	achatOK=0
	venteOK=1
	nbtrade=0
	for i in range(len(prices)):
		if list_pred[i]==-1 and achatOK==0:
			# We buy
			real_price_local=prices[i]
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
			real_price_local=prices[i]
			budget_depart_before=nbitcoin_local*real_price_local 
			budget_depart = budget_depart_before - transaction_price*budget_depart_before
			nbitcoin_local=0
			venteOK=1
			achatOK=0
			#print("We sell and we have ",budget_depart_before," Euros before transaction fees and ",budget_depart," after")
			nbtrade=nbtrade+1
			euros_avec_fees.append(budget_depart)
	profit=euros_avec_fees[len(euros_avec_fees)-1]-Mise
	percent_profit_avec=((euros_avec_fees[len(euros_avec_fees)-1]/Mise)-1)*100
	print("We did ",nbtrade," trades in total over ",len(prices)*5," minutes or ",len(prices)*5/60," hours or ",len(prices)*5/60/24," Days")
	print("La mise de depart est de ",Mise," euros et on a fait ",profit," euros de benefices, soit ",percent_profit_avec," pourcent e benefice sur notre mise de depart")
	

	plt.figure(4)
	ax=plt.subplot(2,1,1)
	ax.plot(euros_avec_fees,c="b",ls="-",label="avec fees")
	ax.axhline(Mise,color="r")
	ax.text(1,1.005*Mise,"Mise de départ",color="r")
	ax.axhline(1.1*Mise,color="r",ls=":")
	ax.text(1,1.105*Mise,"10 % Profit",color="r")
	ax.axhline(1.2*Mise,color="r",ls=":")
	ax.text(1,1.205*Mise,"20 % Profit",color="r")
	ax.axhline(1.3*Mise,color="r",ls=":")
	ax.text(1,1.305*Mise,"30 % Profit",color="r")
	ax.axhline(1.4*Mise,color="r",ls=":")
	ax.text(1,1.405*Mise,"40 % Profit",color="r")
	ax.get_xaxis().set_visible(False)
	ax.set_ylabel("Euros")
	ax1=plt.subplot(2,1,2)
	ax1.plot(bitcoin_avec_fees,c="r",ls="-")
	ax1.set_ylabel("# Bitcoins")
	ax1.set_xlabel("# trade")
	plt.subplots_adjust(wspace=0, hspace=0)
	ax.legend()






############################################################################################
#
#
#                                   Main
#
#
############################################################################################

# On a 385278 points sur la courbes pour 4 ans toutes les 5 minutes


threshold_benefit=0.8
ncell=0
Mise=100
transaction_price=0.26/100
#simu_before_training(threshold_benefit,ncell,Mise,transaction_price)



ideptrain=200
ifintrain=300000#200

ideptest=300000#0
ifintest=340000#200

ntraining=2000
ntesting=400

sizeseg=200
#generate_ICs(ideptrain,ifintrain,ideptest,ifintest,ntraining,ntesting,sizeseg,threshold_benefit,ncell)



nepoch=100
train(sizeseg,nepoch)




#simulation_after_training(threshold_benefit,ncell,sizeseg,Mise)






plt.show()
