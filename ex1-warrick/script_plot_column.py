"""
    Verify serghei subsurface solver with Warrick's solution

    Warrick's solution :
        time = [11700, 23400, 46800]
        wc   = [0.0825, 0.165, 0.2475]
        z1   = [74.58, 75.20, 77.12]
        z2   = [61.32, 62.13, 64.54]
        z3   = [39.02, 40.04, 42.80]
        
    Computational cost:
    	PCA-OMP1	:	60.02s
    	PCA-OMP4	:	57.96s
    	PCA-OMP16   :   70.44s
    	PCA-CUDA	:	97.75s
    	
    	PC-PCG-CUDA		:	64.12s
    	PC-GMRES_CUDA	:	138.22s

"""

import numpy as np
import matplotlib.pyplot as plt

fdir = ['pc-pcg-cuda/','pc-pcg-tracy/','newton-gmres-omp8/']
lgd = ['PCG','GMRES','Newton']

ind = ['1','2','4']
N = [200, 200, 200]
analytical = np.array([[0.0825, 0.165, 0.2475], [74.58, 75.20, 77.12],
    [61.32, 62.13, 64.54], [39.02, 40.04, 42.80]])
color = ['b','r','g']
lstyle = [':', '--', '-','-.']

fs = 12

h_out = []
wc_out = []

for ii in range(len(ind)):
    for ff in range(len(fdir)):
        fname = fdir[ff] + 'out-satu'+str(ind[ii])
        data = np.genfromtxt(fname,delimiter=',')
        wc_out.append(data[:,-1])


plt.figure(1, figsize=[4, 6])
idat = 0

for ii in range(len(ind)):
    for ff in range(len(fdir)):
        zVec = np.linspace(-1.0, 0.0, N[ff])
        plt.plot(np.flipud(np.array(wc_out[idat])), zVec, color=color[ii], ls=lstyle[ff])
        idat += 1
    plt.plot(analytical[0,:], -(1.0-analytical[ii+1,:]/100.0), linestyle='None',
        marker='x', markerfacecolor='None', color=color[ii])

plt.xlabel('Water Content')
plt.ylabel('Z [m]')
plt.legend(lgd, fontsize=fs)

plt.savefig('fig_warrick.eps',format='eps')
plt.show()
