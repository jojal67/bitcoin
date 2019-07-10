# -*- coding: utf-8 -*-
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter






def transform_curve():
	###########################################################
	# Points are given every 5 minutes for the bitcoin prices 
	###########################################################
	prices=np.fromfile("prices")[0:200]
	print ("Taking ",len(prices)," points on the curve")
	minp=np.min(prices)
	maxp=np.max(prices)
	prices=(prices-minp)/(maxp-minp)
	smoothed_prices=gaussian_filter(prices, sigma=2)
	

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
	#print("indices min max = ",indices_min_max)
	
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
	


	plt.figure(1)
	#plt.plot(prices,c="b",ls="--",label="original")
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


		

	
	




def generate_ICs(ideptrain,ifintrain,ideptest,ifintest,ntraining,ntesting,sizeseg):
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
	factor=-minp/(maxp-minp)
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
	
	
	print("So naturally there are ",cptmin, "pixels or ",cptmin*100/len(prices)," % in the min (buy) category")
	print("So naturally there are ",cptmax, "pixels or ",cptmax*100/len(prices)," % in the max (sell) category")
	print("So naturally there are ",len(prices)-cptmin-cptmax,"pixels  or ",(len(prices)-cptmin-cptmax)*100/len(prices)," % in the third (do nothing) category")


	###########################################################
	# 			Balancing data
	#
	#	Having unbalanced data is a very common problem. 
	#	Most machine learning classification algorithms 
	#	are sensitive to unbalanced data. 
	#	An unbalanced dataset will bias the prediction model 
	#	towards the more common class.
	###########################################################

	ncell_needed_in_each_category=int(len(prices)/3)
	print("To unbalance data we need to have ",ncell_needed_in_each_category,"cells in each category")
	nbuymissing=ncell_needed_in_each_category-cptmin
	nsellmissing=ncell_needed_in_each_category-cptmax
	print("There are ",nbuymissing,"cells missing in the buy category")
	print("There are ",nsellmissing,"cells missing in the sell category")

	ncell_to_add_per_side_buy=int(  nbuymissing/cptmin/2    )
	ncell_to_add_per_side_sell=int(  nsellmissing/cptmax/2    )
	print(ncell_to_add_per_side_buy," cells need to be added around each side of a buy label")
	print(ncell_to_add_per_side_sell," cells need to be added around each side of a sell label")

	indice_buy=np.where(action_list==-1)[0]
	indice_sell=np.where(action_list==1)[0]
	
	for i in range(len(indice_buy)):
		action_list[indice_buy[i]-ncell_to_add_per_side_buy-1:indice_buy[i]+ncell_to_add_per_side_buy+1]=-1

	for i in range(len(indice_sell)):
		action_list[indice_sell[i]-ncell_to_add_per_side_sell-1:indice_sell[i]+ncell_to_add_per_side_sell+1]=1
	

	# Checking if balance is OK

	indice_buy=np.where(action_list==-1)[0]
	indice_sell=np.where(action_list==1)[0]
	indice_wait=np.where(action_list==0)[0]

	print("After balancing, there are ",len(indice_buy), " pixels or ",len(indice_buy)*100/len(prices)," % of buy labels")
	print("After balancing, there are ",len(indice_sell), " pixels or ",len(indice_sell)*100/len(prices)," % of sell labels")
	print("After balancing, there are ",len(indice_wait), " pixels or ",len(indice_wait)*100/len(prices)," % of wait labels")



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


	plt.figure(2)
	plt.plot(smoothed_prices,c="b",ls=":",label="smoothed")
	for i in range(len(minbuy)):
		plt.axvspan(minbuy[i], maxbuy[i], facecolor='r', alpha=0.5)
	for i in range(len(minsell)):
		plt.axvspan(minsell[i], maxsell[i], facecolor='g', alpha=0.5)



	
	
	###########################################################
	# 		Generating training set
	###########################################################

	x_training_set=np.zeros((ntraining,sizeseg))
	y_training_set=np.zeros(ntraining)
	for i in range(ntraining):
		dep=np.random.randint(ideptrain,ifintrain-sizeseg)
		fin=dep+sizeseg
		x_training_set[i]=action_list[dep:fin]
		y_training_set[i]=action_list[fin]


	###########################################################
	# 		Generating testing set
	###########################################################

	x_testing_set=np.zeros((ntesting,sizeseg))
	y_testing_set=np.zeros(ntesting)
	for i in range(ntesting):
		dep=np.random.randint(ideptest,ifintest-sizeseg)
		fin=dep+sizeseg
		x_testing_set[i]=action_list[dep:fin]
		y_testing_set[i]=action_list[fin]

	

	#print(y_training_set)

	np.save("factor_normalization.py",factor)
	np.save("x_train.npy",x_training_set)
	np.save("y_train.npy",y_training_set)
	np.save("x_test.npy",x_testing_set)
	np.save("y_test.npy",y_testing_set)


	np.save("real_prices_test_all_curve.npy",prices[ideptest:ifintest])
	np.save("prices_test_all_curve.npy",smoothed_prices[ideptest:ifintest])
	np.save("action_test_all_curve.npy",action_list[ideptest:ifintest])




############################################################################################
#
#
#                                   Main
#
#
############################################################################################

#transform_curve()



ideptrain=0
ifintrain=10000

ideptest=11000
ifintest=16000

ntraining=1000
ntesting=10000

sizeseg=10


generate_ICs(ideptrain,ifintrain,ideptest,ifintest,ntraining,ntesting,sizeseg)


plt.show()
