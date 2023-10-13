"""
	Training the neural network
	INPUT:
		X and Y
	OUTPUT:
		The trained network
		Summary of training results
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

ndata = 10000
n_neuron = 30
n_inputs = 9
epochs = 10000
learning_rate = 0.002
wd = 0.002
dtlim = [0.0]
saveModel = False

labels = ['PCH','MPH']

'''
	Generate training data
'''
fsummary = open('SummaryH-'+str(int(ndata)),'w')
for ll in range(len(labels)):
	for tt in range(len(dtlim)):
		fname = labels[ll]
		nIn = n_inputs
		nHi = n_neuron
		nOu = 2
		nninput = np.genfromtxt(fname+'-X',delimiter=',')
		nnoutput = np.genfromtxt(fname+'-Y')
		nninput = nninput[:ndata,:]
		nnoutput = nnoutput[:ndata]
		line = fname + '\n dtlim : ' + str(np.round(dtlim[tt],2)) + '\n'
		fsummary.write(line)
		print(' Initial : ',np.shape(nninput))
		line = 'Ndata_init : ' + str(np.shape(nninput)) + '\n'
		fsummary.write(line)
		#	Remove small dt intervals
		# xnew = []
		# ynew = []
		# for ii in range(np.shape(nninput)[0]-1):
		# 	if nninput[ii,0] == nninput[ii+1,0] and nninput[ii,1] == nninput[ii+1,1]:
		# 		dt = abs(nninput[ii,2] - nninput[ii+1,2])
		# 		if dt > dtlim[tt]:
		# 			xnew.append(nninput[ii,:])
		# 			ynew.append(nnoutput[ii])
		# 	else:
		# 		xnew.append(nninput[ii,:])
		# 		ynew.append(nnoutput[ii])
		# nninput = np.array(xnew)
		# nnoutput = np.array(ynew)
		print(' Final : ',np.shape(nninput))
		line = 'Ndata_final : ' + str(np.shape(nninput)) + '\n'
		fsummary.write(line)

		x = torch.from_numpy(nninput).type(torch.float)
		y = torch.from_numpy(nnoutput).type(torch.long)

		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

		'''
			Build and train NN
		'''
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
		x_train, y_train = x_train.to(device), y_train.to(device)
		x_test, y_test = x_test.to(device), y_test.to(device)

		model = clsf.Net(nIn, nHi, nOu)
		criterion = nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)

		losses = []
		for i in range(epochs):
			y_pred = model.forward(x_train)
			loss = criterion(y_pred,y_train)
			losses.append(loss.item())
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			#	Testing
			with torch.inference_mode():
				test_pred = model(x_test)
				test_loss = criterion(test_pred, y_test)
			if i % 1000 == 0:
				print(f"Epoch: {i} | Loss: {loss:.5f} | Test loss: {test_loss:.5f}")

		acc_train = accuracy_score(model.predict(x_train),y_train)
		acc_test = accuracy_score(model.predict(x_test),y_test)
		print('Completing ',labels[ll],' with dt_interval = ',dtlim[tt],' sec !')
		print(f"Accuray Score -- Training: {acc_train:.3f} | Testing: {acc_test:.3f}")
		line = 'TrainingScore : ' + str(np.round(acc_train,3)) + '\n'
		fsummary.write(line)
		line = 'TestingScore : ' + str(np.round(acc_test,3)) + '\n\n'
		fsummary.write(line)
		#	Save model
		if saveModel:
			torch.save(model, labels[ll]+'-N'+str(int(ndata))+'-Model.pt')
fsummary.close()
