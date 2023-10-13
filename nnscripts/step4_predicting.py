"""
	Test the trained network on new datasets
	INPUT:
		New dataset (vgextrap)
		Standardization coefficients
		Trained neural network
	OUTPUT:
		The optimal set of input parameters (predicted by NN)

	The optimal inputs can be used for forward simulation to see if it succeeds
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn import datasets, decomposition, preprocessing
import nnclassifier as clsf
import math

labels = ['PC','MP']
vg_limit = True
#	Upper limit of dt_max (2000 for PC/MP; 5000 for PCH/MPH)
dt0 = 5000.0
#	Number of training samples in the trained nn model
nsample = 10000
#	Number of VG groups
nvg = 100
# #	Initial water content
# wci_multiplier = 0.2
# #	Initial boundary condition
# htop = 0.01
# #	Grid resolution
# nz = 80
# dz = 1.0/nz
#	Available dt_max tested
tvec = []
torig = []
dt = 0.01
while dt < dt0:
	torig.append(dt)
	tvec.append(math.log(dt))
	dt = dt * 1.5
nt = len(tvec)
print(torig)
tvec = np.array(tvec)
torig = np.array(torig)
tvec = tvec.reshape(-1, 1)
#	Standardization of dt_max
transcoef = {}
dtmax = {}
for ll in range(len(labels)):
	transcoef[labels[ll]] = np.genfromtxt('transcoef-'+labels[ll],delimiter=',')
	dtmax[labels[ll]] = (tvec - transcoef[labels[ll]][-1,0]) / transcoef[labels[ll]][-1,1]
	dtmax[labels[ll]].reshape(np.shape(tvec)[0])


'''
	Load the new dataset + standardization
'''
vg = np.genfromtxt('vgextrap',delimiter=',')
np.random.shuffle(vg)
if vg_limit:
	data = []
	for ii in range(np.shape(vg)[0]):
		if vg[ii,1] < 20 and vg[ii,2] > 1.4 and vg[ii,3] < 0.6 and vg[ii,0] > 5e-8:
			data.append(vg[ii,:])
	data = np.array(data)
else:
	data = vg
print(np.shape(data))

x = {}
wci_column = []
dz_column = []
htop_column = []
all_column = np.zeros((np.shape(data)[0], np.shape(transcoef[labels[0]])[0]-1))
for jj in range(np.shape(transcoef[labels[0]])[0]-1):
	column = []
	if jj < 5:
		for ii in range(np.shape(data)[0]):
			if data[ii,jj] == 0:
				data[ii,jj] += 1e-3
			column.append(math.log(data[ii,jj]))
	elif jj == 5:
		for ii in range(np.shape(data)[0]):
			wci_multiplier = random.random() * 0.8
			wci = data[ii,4] + wci_multiplier * (data[ii,3] - data[ii,4])
			column.append(math.log(wci))
			wci_column.append(wci)
	elif jj == 6:
		for ii in range(np.shape(data)[0]):
			htop = random.random() * 0.1
			column.append(math.log(htop))
			htop_column.append(htop)
	elif jj == 7:
		for ii in range(np.shape(data)[0]):
			nz = np.round(random.random() * 60 + 40)
			dz = 1.0 / nz
			column.append(math.log(dz))
			dz_column.append(dz)
	column = np.array(column)
	column = column.reshape(-1, 1)
	all_column[:,jj] = np.squeeze(column)

#	Standardization
for ll in range(len(labels)):
	x[labels[ll]] = []
	for jj in range(np.shape(transcoef[labels[ll]])[0]-1):
		column = all_column[:,jj]
		column_std = (column - transcoef[labels[ll]][jj,0]) / transcoef[labels[ll]][jj,1]
		x[labels[ll]].append(column_std.reshape(np.shape(column_std)[0]))
	x[labels[ll]] = np.transpose(np.array(x[labels[ll]]))


'''
	Generate NN inputs (randomly choose nvg groups of data)
'''
idx = []
for ii in range(nvg):
	idx.append(int(np.round(random.random() * np.shape(x[labels[0]])[0])))
datax = {}
xorig = np.zeros((nvg*nt, np.shape(transcoef[labels[0]])[0]))
for ll in range(len(labels)):
	kk = 0
	datax[labels[ll]] = np.zeros((nvg*nt, np.shape(transcoef[labels[0]])[0]))
	for ii in range(nvg):
		for tt in range(nt):
			#	transformed input
			datax[labels[ll]][kk,:-1] = x[labels[ll]][idx[ii],:]
			datax[labels[ll]][kk,-1] = dtmax[labels[ll]][tt]
			#	original input
			if ll == 0:
				xorig[kk,:5] = data[idx[ii],:5]
				if np.shape(xorig)[1] > 6:
					xorig[kk,5] = wci_column[idx[ii]]
					xorig[kk,6] = htop_column[idx[ii]]
					xorig[kk,7] = dz_column[idx[ii]]
				xorig[kk,-1] = torig[tt]
			kk += 1

# np.savetxt('nninput-test',datax,delimiter=',')
# np.savetxt('nninput-orig',xorig,delimiter=',')

'''
	Predicting with NN
'''
for ll in range(len(labels)):
	model = torch.load(labels[ll]+'-N'+str(nsample)+'-Model.pt')
	model.eval()
	x_input = datax[labels[ll]]
	print('Shape of NN inputs: ',np.shape(x_input))
	x_extra = torch.from_numpy(x_input).type(torch.float)
	y_extra = model.predict(x_extra)
	y_extra = y_extra.detach().numpy()
	out = np.zeros((np.shape(x_input)[0],np.shape(x_input)[1]+1))
	for ii in range(len(y_extra)):
		out[ii,:-1] = x_extra[ii,:]
		out[ii,-1] = y_extra[ii]
	np.savetxt(labels[ll]+'-Out',out,delimiter=',',fmt='%.5f')

'''
	Post Processing
'''
for ll in range(len(labels)):
	out = np.genfromtxt(labels[ll]+'-Out', delimiter=',')
	ii = 0
	optimal = []
	bins = []
	print(np.shape(out)[0])
	while ii < np.shape(out)[0]-1:
		bins.append(out[ii,-1])
		if out[ii,0] != out[ii+1,0] and out[ii,1] != out[ii+1,1]:
			nb = len(bins)
			nsucc = 0
			#	total number of success cases in bins
			for jj in range(nb):
				if bins[jj] == 1:
					nsucc += 1
			#	Case 1 : All Failure
			if nsucc == 0:
				tmp = []
				for kk in range(np.shape(xorig)[1]-1):
					tmp.append(xorig[ii,kk])
				tmp.append(xorig[ii+1,-1])
				tmp.append(1)
				optimal.append(tmp)
				# optimal.append([xorig[ii,0],xorig[ii,1],xorig[ii,2],xorig[ii,3],xorig[ii,4],xorig[ii+1,5],1])
			#   Case 2 : All Success
			elif nsucc == nb:
				tmp = []
				for kk in range(np.shape(xorig)[1]):
					tmp.append(xorig[ii,kk])
				tmp.append(2)
				optimal.append(tmp)
				# optimal.append([xorig[ii,0],xorig[ii,1],xorig[ii,2],xorig[ii,3],xorig[ii,4],xorig[ii,5],2])
			else:
				#	Case 3 : Have success in the middle (wrong NN prediction)
				if bins[0] == 0:
					if bins[-1] == 1:
						tmp = []
						for kk in range(np.shape(xorig)[1]):
							tmp.append(xorig[ii,kk])
						tmp.append(3)
						optimal.append(tmp)
					else:
						for jj in range(nb-1):
							if bins[jj] == 1 and bins[jj+1] == 0:
								idx = ii - nt + jj + 1
								tmp = []
								for kk in range(np.shape(xorig)[1]):
									tmp.append(xorig[idx,kk])
								tmp.append(3)
								optimal.append(tmp)
								break
							# optimal.append([xorig[idx,0],xorig[idx,1],xorig[idx,2],xorig[idx,3],xorig[idx,4],xorig[idx,5],3])
				#	Case 4 : Normal Finish
				else:
					for jj in range(nb-1):
						if bins[jj] == 1 and bins[jj+1] == 0:
							idx = ii - nt + jj + 1
							tmp = []
							for kk in range(np.shape(xorig)[1]):
								tmp.append(xorig[idx,kk])
							tmp.append(4)
							optimal.append(tmp)
							break
							# optimal.append([xorig[idx,0],xorig[idx,1],xorig[idx,2],xorig[idx,3],xorig[idx,4],xorig[idx,5],4])
			bins = []
		ii += 1

	np.savetxt(labels[ll]+'-Optimal',np.array(optimal),delimiter=',')
