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
			else:
				first_action=1 # sell

	action_list=action_list[::-1]
	cptfirstaction=0
	for i in range(len(action_list)):
		if action_list[i]!=0 and cptfirstaction==0:
			cptfirstaction=1
			if action_list[i]==-1:
				last_action=-1 # buy
			else:
				last_action=1 # sell

	return first_action,last_action


def transform_curve():
	prices=np.fromfile("prices")[0:2000]#[500:800]
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

	
	####################################################
	# Annotate a third time to suppress action repeated
	####################################################

	first_action,derniere_action=check_action(action_list)
	cpt=0
	next_action=0
	last_action=first_action
	for i in range(len(action_list)):
		if action_list[i]!=0:

			if next_action==1:
				action_list[i]=0
				next_action=0

			action_local=action_list[i]
			if cpt>0 and action_local==last_action:
				action_list[i]=0
				next_action=1
			last_action=action_local
			cpt=cpt+1
		 
			
	
	
	###################################################################################################
	# Annotate a fourth time to supress not couple of buy sell action below the percent_profit value
	###################################################################################################



	plt.figure(1)
	plt.plot(smoothed_prices,c="b",ls=":",label="smoothed")
	plt.plot(first_deriv,c="r",label="f'")
	plt.plot(second_deriv,c="g",label="f''")
	#plt.plot(signchange,c="y",label="min and max")
	#plt.plot(action_list,c="c",label="min and max")
	plt.axhline(0)
	cptmin=0
	cptmax=0
	for i in range(len(action_list)):
		if action_list[i]==-1:
			plt.axvline(i,color="y")
			cptmin=1
		if action_list[i]==1:
			plt.axvline(i,color="c")
			cptmax=1

	plt.legend()








############################################################################################
#
#
#                                   Main
#
#
############################################################################################




transform_curve()


plt.show()
