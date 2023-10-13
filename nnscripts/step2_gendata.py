"""
	Prepare training data for neural network
	INPUT:
		Rich3d batch simulation results	[Ks, alpha, n, wcs, wcr, wci, htop, dz, dt, flag]
	OUTPUT:
		Neural network input, X : [Ks alpha, n, wcs, wcr, wci, htop, dz, dt] <- standardized
		Neural network output, Y : success or failure (1 or 0)
		Standardization coefficients : [mean, std]
"""


import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import numpy as np
import os
import math
from sklearn import datasets, decomposition, preprocessing


fname = ['batchpc-h500','batchmp-h500']
label = ['PCH','MPH']

'''
	Read all simulation results
'''
data = {}
x = {}
y = {}
t = {}
xpca = {}
for ll in range(len(label)):
	data[label[ll]] = np.genfromtxt(fname[ll], delimiter=',')
	x[label[ll]] = []
	y[label[ll]] = []
	t[label[ll]] = []
	for ii in range(np.shape(data[label[ll]])[0]):
		if data[label[ll]][ii,2] > 0.0:
			x[label[ll]].append(data[label[ll]][ii,:-2])
			y[label[ll]].append(data[label[ll]][ii,-1])
			t[label[ll]].append(data[label[ll]][ii,-2])
	x[label[ll]] = np.array(x[label[ll]])
	y[label[ll]] = np.array(y[label[ll]])
	t[label[ll]] = np.array(t[label[ll]])

'''
	Standardization
'''

for ll in range(len(label)):
	transcoef = []
	#	Processing x
	for jj in range(np.shape(x[label[ll]])[1]):
		column = np.squeeze(x[label[ll]][:,jj])
		for kk in range(len(column)):
			if column[kk] == 0:
				column[kk] += 1e-3
			column[kk] = math.log(column[kk])
		column = column.reshape(-1, 1)
		transcoef.append([np.nanmean(column), np.nanstd(column)])
		column = preprocessing.StandardScaler().fit_transform(column)
		x[label[ll]][:,jj] = column.reshape(np.shape(column)[0])
	#	Processing t
	column = t[label[ll]]
	for kk in range(len(column)):
		column[kk] = math.log(column[kk])
	column = column.reshape(-1, 1)
	transcoef.append([np.nanmean(column), np.nanstd(column)])
	column = preprocessing.StandardScaler().fit_transform(column)
	t[label[ll]] = column.reshape(np.shape(column)[0])
	np.savetxt('transcoef-'+label[ll], np.array(transcoef), delimiter=',')



'''
	Split data by P1
'''
for ll in range(len(label)):
	input1 = []
	output1 = []
	input2 = []
	output2 = []
	for ii in range(len(t[label[ll]])):
		dx = x[label[ll]]
		dy = y[label[ll]]
		dt = t[label[ll]]
		tmp = []
		for jj in range(np.shape(dx)[1]):
			tmp.append(dx[ii,jj])
		tmp.append(dt[ii])
		input1.append(tmp)
		if dy[ii] == 3:
			output1.append(1)
		else:
			output1.append(0)
	np.savetxt(label[ll]+'-X',np.array(input1),delimiter=',',fmt='%.5f')
	np.savetxt(label[ll]+'-Y',np.array(output1),delimiter=',',fmt='%d')
