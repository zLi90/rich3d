'''
    The class for NN classifier
'''
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
	def __init__(self, nIn, nHi, nOu):
		super(Net,self).__init__()
		#This applies Linear transformation to input data.
		self.fc1 = nn.Linear(nIn,nHi)
		self.fc2 = nn.Linear(nHi,nHi)
		self.fc3 = nn.Linear(nHi,nOu)

    #This must be implemented
	def forward(self,x):
		x = self.fc1(x)
		x = F.tanh(x)
		x = self.fc2(x)
		x = F.tanh(x)
		x = self.fc3(x)
		return x

    #This function takes an input and predicts the class, (0 or 1)
	def predict(self,x):
		#Apply softmax to output
		pred = F.softmax(self.forward(x), dim=1)
		ans = []
		for t in pred:
			if t[0]>t[1]:
				ans.append(0)
			else:
				ans.append(1)
		return torch.tensor(ans)
