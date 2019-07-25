# -*- coding: utf-8 -*-
import sys
import os
import matplotlib.pyplot as plt
import numpy as np




def convert():

	prices_1=np.load("prices_four_years_1_min.npy")
	prices_5=prices_1[::5]
	print(len(prices_5))

	plt.figure(1)
	plt.plot(prices_1,c="b")
	plt.figure(2)
	plt.plot(prices_5,c="r",ls=":")

	np.save("prices_four_years_5_min.npy",prices_5)





def convert_volume():

	prices_1=np.load("volume_1_min.npy")
	prices_5=prices_1[::5]
	print(len(prices_5))

	plt.figure(1)
	plt.plot(prices_1,c="b")
	plt.figure(2)
	plt.plot(prices_5,c="r",ls=":")

	np.save("volume_four_years_5_min.npy",prices_5)


def compare_prices_volume():

	prices=np.load("prices_four_years_5_min.npy")
	volume=np.load("volume_four_years_5_min.npy")

	minp=np.min(prices)
	maxp=np.max(prices)
	prices=(prices-minp)/(maxp-minp)

	minp=np.min(volume)
	maxp=np.max(volume)
	volume=(volume-minp)/(maxp-minp)


	dep=100
	fin=10000
	plt.figure(1)
	plt.plot(prices[dep:fin],c="b")
	plt.plot(volume[dep:fin],c="r")



#convert()

#convert_volume()

compare_prices_volume()

plt.show()
