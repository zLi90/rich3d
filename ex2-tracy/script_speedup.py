"""
	Plot speedup
"""
import numpy as np
import matplotlib.pyplot as plt


T_base = 14045.84

T = []
#	Newton-GMRES series
T.append([])
T[0] = [[2, 5364.65], [4, 2983.77], [8, 1632.87], [16, 1496.05], [32, 215.62]]

#	Newton-GMRES(VG) series
T.append([])
T[1] = [[8, 15783.44], [32, 1904.85]]

#	PC-CG(VG) series
T.append([])
T[2] = [[8, 222.77], [32, 92.76]]

#	PC-GMRES(VG) series
T.append([])
T[3] = [[8, 14327.16], [32, 1975.65]]

#	Newton-GMRES dt=1800
T.append([])
T[4] = [[8, 1063.01],[32, 179.08]]


colors = ['b','r','g','k','m']
mt = ['v','s','*','d','o']
lgd = ['NR-GMRES(EXP)','NR-GMRES(VG)','PC-CG(VG)','PC-GMRES(VG)','NR-GMRES(EXP) dt+']


plt.figure(1)


for jj in range(5):
	for ii in range(len(T)):
		if len(T[ii]) > jj:
			nthread = T[ii][jj][0]
			time = T[ii][jj][1]
			plt.scatter(nthread, T_base/time, s=75, marker=mt[ii], facecolors='none', edgecolors=colors[ii])
			
#plt.plot([1,256],[1,256],'k-')
#plt.xlim([1,40])
		
ax = plt.gca()
ax.set_xscale('log',base=2)
ax.set_yscale('log',base=2)
plt.xlabel('Number of threads')
plt.ylabel('Speed up')
plt.xticks([2,4,8,16,32],[2,4,8,16,'CUDA'])
plt.yticks([1,4,16,64,256],[1,4,16,64,256])

plt.legend(lgd, loc='upper left')


plt.show()
