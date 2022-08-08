"""
    Verify serghei subsurface solver with Tracy's analytical solution

"""
import numpy as np
import matplotlib.pyplot as plt
from analytical import *

"""
    --------------------------------------------------------------------
                        Load SERGHEI outputs
    --------------------------------------------------------------------
"""

"""
	Computational cost
		dx = 0.125, T = 18000s, dt = 1-180s:
			NT-GM		CUDA	215.62 -- 204.96
			NT-GM		OMP1	14045.84 -- 13988.23
			NT-GM		OMP2	5364.65 -- 5332.70
			NT-GM		OMP4	2983.77 -- 2965.37
			NT-GM		OMP8	1632.87 -- 1621.87
			NT-GM		OMP16	1496.05 -- 1485.97
		    --------------------
		    NT-GM(VG)	OMP8	15783.44 -- 15767.65
		    NT-GM(VG)	CUDA	1904.85 -- 1890.32
		    --------------------
		    PC-CG(VG)	OMP8	222.77 -- 179.07
		    PC-CG(VG)	CUDA	92.76 -- 38.20
		    PC-CG(VG)	OMP8	65.18 -- 52.12	(dwc range = 0.1-0.5 : UNSTABLE!)
		    --------------------
		    PC-GM(VG)	OMP8	14327.16 -- 14284.88
		    PC-GM(VG)	CUDA	1975.65 -- 1921.53
		    --------------------
		    
		dx = 0.125, T = 90000s, dt = 1-3600s:
			NT-GM		CUDA	135.73 -- 129.68
			NT-GM(VG)	CUDA	581.70 -- 574.85
			PC-CG(VG)	CUDA	361.49 -- 147.12
			--------------------	
		    
		dx = 0.25, T = 18000s, dt = 1-1800s:
			NT-GM		OMP8	1063.01 -- 1058.67
			NT-GM		CUDA	179.08 -- 173.47
			NT-GM(VG)	OMP8	
			NT-GM(VG)	CUDA	

"""

#fdir = 'out-nt-gm-vg-cuda-T90000/'
fdir = 'out-pc-cg-vg-cuda-T90000/'
ind = ['0']
dim = [80, 48, 64]
N = int(dim[0]*dim[1]*dim[2])
zVec = np.linspace(-6.0, 0.0, N)
x_bound = 40
y_slice = 23

h_out = []
wc_out = []
for ii in range(len(ind)):
    h = np.genfromtxt(fdir+'out-head'+str(ind[ii]))
    wc = np.genfromtxt(fdir+'out-satu'+str(ind[ii]))
    h_out.append(np.reshape(h[:,3], (dim[2],dim[0],dim[1]), order='F'))
    wc_out.append(np.reshape(wc[:,3], (dim[2],dim[0],dim[1]), order='F'))
    
h_out = np.array(h_out)
wc_out = np.array(wc_out)
print('Dimension of Rich3d output = ',np.shape(wc_out))

"""
    --------------------------------------------------------------------
                            Analytical Solution
    --------------------------------------------------------------------
"""
L = 10.0
W = 6.0
H = 8.0
#   soil properties
wcs = 0.35
wcr = 0.016
Ks = 1e-4
ga = 0.1634
#   grid resolutions
dx = [0.125, 0.125, 0.125]
#   initial / boundary condition
h0 = -12.0
#   Time step
tvec = np.linspace(0, 90000, 2)
#   build domain
#out = np.zeros((int(L/dx[0]), int(W/dx[1]), int(H/dx[2]), len(tvec)))
out = np.zeros((len(tvec), int(H/dx[2]), int(L/dx[0]), int(W/dx[1])))
print('Dimension of analytical solution = ',np.shape(out))
'''
    Get the analytical solution
'''
solver = Tracy_analytic([L,W,H], Ks, wcs, wcr, ga)
for ii in range(np.shape(out)[2]):
    x = (ii+0.5)*dx[0]
    for jj in range(np.shape(out)[3]):
        y = (jj+0.5)*dx[1]
        for kk in range(np.shape(out)[1]):
            z = (kk+0.5)*dx[2]
            for tt in range(len(tvec)):
                out[tt,kk,ii,jj] = solver.get_solution(x, y, z, tvec[tt], h0)
             

"""
    --------------------------------------------------------------------
                                Make plot
    --------------------------------------------------------------------
"""
"""
relerr3d = (np.transpose(h_out[:,:,:,y_slice]) - np.transpose(out[-1,:,:,y_slice]))/ (np.transpose(out[-1,:,:,y_slice]))
maxerr = np.amax(relerr3d)
avgerr = np.mean(relerr3d)
print(' Max err = ',maxerr)
print(' Avg err = ',avgerr)"""

print(np.shape(h_out))
print(np.shape(out))
print(y_slice)

relerr = ((h_out[0,:,:,y_slice]) - np.flipud((out[-1,:,:,y_slice]))) / np.flipud((out[-1,:,:,y_slice]))

plt.figure(1, figsize=[12, 5])

plt.subplot(1,3,1)
plt.imshow((h_out[0,:,:x_bound,y_slice]), vmin=-12, vmax=0, cmap='jet')
plt.colorbar()
xlabel = [0, 2.5, 5]
xtick = np.linspace(0,int(dim[0]/2-1),3)
ylabel = [0, -4, -8]
ytick = np.linspace(0,dim[2]-1,3)
plt.xticks(xtick,xlabel)
plt.yticks(ytick,ylabel)
plt.xlabel('X [m]')
plt.ylabel('Z [m]')
plt.title('Serghei')

plt.subplot(1,3,2)
plt.imshow(np.flipud(out[-1,:,:x_bound,y_slice]), vmin=-12, vmax=0, cmap='jet')
plt.colorbar()
xlabel = [0, 2.5, 5]
xtick = np.linspace(0,int(dim[0]/2-1),3)
ylabel = [0, -4, -8]
ytick = np.linspace(0,dim[2]-1,3)
plt.xticks(xtick,xlabel)
plt.yticks(ytick,ylabel)
plt.xlabel('X [m]')
plt.ylabel('Z [m]')
plt.title('Analytical')

plt.subplot(1,3,3)
plt.imshow(abs(relerr[:,:x_bound]), vmin=0.0, vmax=0.05, cmap='jet')
plt.colorbar()
xlabel = [0, 2.5, 5]
xtick = np.linspace(0,int(dim[0]/2-1),3)
ylabel = [0, -4, -8]
ytick = np.linspace(0,dim[2]-1,3)
plt.xticks(xtick,xlabel)
plt.yticks(ytick,ylabel)
plt.xlabel('X [m]')
plt.ylabel('Z [m]')
plt.title('Relative Error')

plt.savefig('fig_tracy.eps',format='eps')

plt.show()
