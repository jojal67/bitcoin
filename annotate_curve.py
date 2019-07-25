# -*- coding: utf-8 -*-
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter



def check_action(action_list):
	cptfirstaction=0
	for i in range(len(action_list)):
		if action_list[i]!=0 and cptfirstaction==0:
			cptfirstaction=1
			if action_list[i]==-1:
				first_action=-1 # buy
				print("first action : buy ")
			else:
				first_action=1 # sell
				print("first action : sell ")

	action_list=action_list[::-1]
	cptfirstaction=0
	for i in range(len(action_list)):
		if action_list[i]!=0 and cptfirstaction==0:
			cptfirstaction=1
			if action_list[i]==-1:
				last_action=-1 # buy
				print("last action : buy ")
			else:
				last_action=1 # sell
				print("last action : sell ")
	
	return first_action,last_action








def transform_curve(threshold_benefit,ncell):
	a=np.random.randint(0,15000)
	prices=np.fromfile("prices")[a:a+200]
	print ("Taking ",len(prices)," points on the curve,  so a total of ",len(prices)*5," Minutes or ",len(prices)*5/60," hours or ",len(prices)*5/60/24," Days")
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
	print("There are ",len(indices_min_max)," max and min")
	
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
	
	print("There are ",cptmin," min and ",cptmax," max")	
	

	check_action(action_list)


	plt.figure(1)
	ax=plt.subplot(2,3,1)
	ax.plot(smoothed_prices,c="b",ls=":",label="smoothed")
	for i in range(len(action_list)):
		if action_list[i]==-1:
			ax.axvline(i,color="y")
		if action_list[i]==1:
			ax.axvline(i,color="c")
	ax.get_xaxis().set_visible(False)	


	
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

	print("indice sell = ",list_indice_sell)
	print("indice buy = ",list_indice_buy)		
			
	action_list=np.zeros(len(smoothed_prices))
	action_list[list_indice_sell]=1
	action_list[list_indice_buy]=-1

	ax=plt.subplot(2,3,2)
	ax.plot(smoothed_prices,c="b",ls=":",label="smoothed")
	for i in range(len(action_list)):
		if action_list[i]==-1:
			ax.axvline(i,color="y")
		if action_list[i]==1:
			ax.axvline(i,color="c")
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	####################################################
	# Annotate a third time to suppress action repeated
	####################################################
	
	first_action,derniere_action=check_action(action_list)

	
	cpt=0
	next_action=0
	last_action=first_action
	for i in range(len(action_list)):
		if action_list[i]!=0:

			#if next_action==1:
				#action_list[i]=0
				#next_action=0


			action_local=action_list[i]
			if cpt>0 and action_local==last_action:
				action_list[i]=0
				next_action=1
			last_action=action_local
			cpt=cpt+1
		 
			
	ax=plt.subplot(2,3,3)
	ax.plot(smoothed_prices,c="b",ls=":",label="smoothed")
	for i in range(len(action_list)):
		if action_list[i]==-1:
			ax.axvline(i,color="y")
		if action_list[i]==1:
			ax.axvline(i,color="c")
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	
	###################################################################################################
	# Annotate a fourth time to supress couple of buy sell action below the percent_profit value
	###################################################################################################

	"""
	indice_buy=np.where(action_list==-1)[0]
	indice_sell=np.where(action_list==1)[0]
	for i in range(len(indice_buy)):
		benefit=prices[indice_sell[i]]-prices[indice_buy[i]]
		percent_benefit=benefit*100/prices[indice_buy[i]]	
		if percent_benefit<threshold_benefit:
			action_list[indice_buy[i]]=0
			action_list[indice_sell[i]]=0
	"""	

	indice_buy=np.where(action_list==-1)[0]
	print("Finally for ",len(prices)*5," Minutes or ",len(prices)*5/60," hours or ",len(prices)*5/60/24," Days, we keep ",len(indice_buy)," couple of buy and sell actions")


	ax=plt.subplot(2,3,4)
	ax.plot(smoothed_prices,c="b",ls=":",label="smoothed")
	for i in range(len(action_list)):
		if action_list[i]==-1:
			ax.axvline(i,color="y")
		if action_list[i]==1:
			ax.axvline(i,color="c")




	#####################################################################################################################
	# Annotate a fourth time to add ncell around the buy sell actions to unbalance data to train a future neural network  
	#####################################################################################################################

	indice_buy=np.where(action_list==-1)[0]
	indice_sell=np.where(action_list==1)[0]
	for i in range(len(indice_buy)):
		action_list[indice_buy[i]-ncell:indice_buy[i]+ncell]=-1
	for i in range(len(indice_sell)):
		action_list[indice_sell[i]-ncell:indice_sell[i]+ncell]=1



	ax=plt.subplot(2,3,5)
	ax.plot(smoothed_prices,c="b",ls=":",label="smoothed")
	for i in range(len(action_list)):
		if action_list[i]==-1:
			ax.axvline(i,color="y")
		if action_list[i]==1:
			ax.axvline(i,color="c")
	ax.get_yaxis().set_visible(False)


	plt.subplots_adjust(hspace=0,wspace=0)


	
	plt.figure(2)
	cptmin=0
	cptmax=0
	for i in range(len(action_list)):
		if action_list[i]==-1:
			plt.axvline(i,color="y")
			cptmin=1
		if action_list[i]==1:
			plt.axvline(i,color="c")
			cptmax=1
	plt.plot(smoothed_prices,c="b",ls=":",label="smoothed")
	plt.plot(first_deriv,c="r",label="f'")
	plt.plot(second_deriv,c="g",label="f''")

	plt.legend()


	##########################
	#   Some statistics
	##########################

	indice_buy=np.where(action_list==-1)[0]
	indice_sell=np.where(action_list==1)[0]
	ncell_between_two_buy=[]
	for i in range(1,len(indice_buy)):
		ncell_between_two_buy.append(indice_buy[i]-indice_buy[i-1])
	ncell_between_two_buy=np.array(ncell_between_two_buy)
	print("Liste duree entre deux achats = ",ncell_between_two_buy)
	ncell_between_two_sell=[]
	for i in range(1,len(indice_sell)):
		ncell_between_two_sell.append(indice_sell[i]-indice_sell[i-1])
	ncell_between_two_sell=np.array(ncell_between_two_sell)
	print("Liste duree entre deux ventes = ",ncell_between_two_sell)
	ncell_between_one_buy_and_one_sell=[]
	for i in range(len(indice_sell)):
		ncell_between_one_buy_and_one_sell.append(indice_sell[i]-indice_buy[i])
	ncell_between_one_buy_and_one_sell=np.array(ncell_between_one_buy_and_one_sell)
	print("Liste duree entre un achat et une vente = ",ncell_between_one_buy_and_one_sell)


	ncell_mean_bt_two_buy=np.average(ncell_between_two_buy)
	ncell_mean_bt_two_sell=np.average(ncell_between_two_sell)
	ncell_mean_bt_buy_sell=np.average(ncell_between_one_buy_and_one_sell)

	print("Duree moyenne entre deux achats = ",ncell_mean_bt_two_buy," cells or ",ncell_mean_bt_two_buy*5/60," heures")
	print("Duree moyenne entre deux ventes = ",ncell_mean_bt_two_sell," cells or ",ncell_mean_bt_two_sell*5/60," heures")
	print("Duree moyenne entre un achat et une vente = ",ncell_mean_bt_buy_sell," cells or ",ncell_mean_bt_buy_sell*5/60," heures")
	


	
	

	return action_list








############################################################################################
#
#
#                                   Main
#
#
############################################################################################



threshold_benefit=1.
ncell=0
transform_curve(threshold_benefit,ncell)


plt.show()
