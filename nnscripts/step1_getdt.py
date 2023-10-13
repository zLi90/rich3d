"""
    Run batch infiltration simulations with Rich3d
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import subprocess
import os

"""
    Read the dataset : [Ks, alpha, n, wcr, wcs]
"""
#   cm/d to m/s
ntest = 3000
coef = 8640000
dname = 'vgdata'+str(ntest)
if os.path.exists(dname):
	vg = np.genfromtxt(dname, delimiter=',')
else:
	data = np.genfromtxt('WRC_dataset_simplified.csv', delimiter=',',skip_header=1)
	ndata = np.shape(data)[0]
	vg = []
	for ii in range(ndata):
		#   If alpha, n and wcs are within the range
		#if data[ii,1] < 90 and data[ii,2] > 1.3 and data[ii,4] < 0.9:
		if data[ii,1] < 100 and data[ii,4] < 1 and data[ii,0]/coef >= 1e-8:
		    #   If K is different
		    if ii > 0 and data[ii,0] != data[ii-1,0]:
		        vg.append([data[ii,0]/coef, data[ii,1], data[ii,2], data[ii,3], data[ii,4]])
	vg = np.array(vg)
	np.random.shuffle(vg)
	vg = vg[:ntest,:]
	np.savetxt('vgdata'+str(ntest), vg, delimiter=',')
print(" Total groups of data: ",np.shape(vg)[0])

"""
    Run batch rich3d simulations
"""
#	Testing scenarios
dt_base = 2000.0
dt_min = 0.01
fdir = 'p3-batchpc/'
fsubdir = 'pcvg5000'

#   For each groups of vg parameters, generate ndt values of dt_max
ndt = 10
mu = 1e-3
rho = 1e3
grav = 9.81
os.mkdir(fdir+fsubdir)
for ivg in range(np.shape(vg)[0]):
    out_dir = fsubdir + '/vg' + str(ivg)
    os.mkdir(fdir+out_dir)
    #   Soil properties
    Ks = vg[ivg,0]
    alpha = vg[ivg,1]
    n = vg[ivg,2]
    wcr = vg[ivg,3]
    wcs = vg[ivg,4]
    dtOldMax = dt_base
    dtOldMin = dt_min
    for tt in range(ndt):
        #   Randomly generates dt_max
        if tt == 0:
        	dt_max = 0.0
        	while dt_max <= dt_min:
        		dt_max = np.round(dt_base * random.random(), 2)
        else:
            fid = open(fdir + fout + 'EndInfo', 'r')
            lines = fid.readlines()
            fid.close()
            for kk in range(len(lines)):
                line = lines[kk]
                if line[:4] == 'Exit':
                    ecode = int(line[12])
                elif line[:4] == 'dtMa':
                    dtOld = float(line[8:13])

            if ecode == 3 and dtOld > dtOldMin:
            	dtOldMin = dtOld
            elif ecode == 0:
            	break
            elif dtOld < dtOldMax:
            	dtOldMax = dtOld

            #	Get new random dt_max
            if ecode == 3:
            	dtNew = 0.3*dt_max + 0.7*dtOldMax
            	if abs(dtNew-dtOld) <= dt_min or dtNew > dt_base*0.95:
            		break
            	else:
            		dt_max = dtNew
            else:
            	dtNew = 0.3*dt_max + 0.7*dtOldMin
            	if abs(dtNew-dtOld) <= dt_min or dtNew < dt_min*2.0:
            		break
            	else:
            		dt_max = dtNew
        #   Read and rewrite inputs
        fid = open(fdir+'input', 'r')
        lines = fid.readlines()
        fid.close()
        newlines = []
        for kk in range(len(lines)):
            line = lines[kk]
            if line[:2] == 'kz':
                line = 'kz = ' + str(Ks*mu/rho/grav) + '\n'
            elif line[:3] == 'vga':
                line = 'vga = ' + str(alpha) + '\n'
            elif line[:3] == 'vgn':
                line = 'vgn = ' + str(n) + '\n'
            elif line[:3] == 'wcr':
                line = 'wcr = ' + str(wcr) + '\n'
            elif line[:3] == 'phi':
                line = 'phi = ' + str(wcs) + '\n'
            elif line[:7] == 'wc_init':
                #line = 'wc_init = ' + str(wcr+0.01) + '\n'
                line = 'wc_init = ' + str(wcr+0.05*(wcs-wcr)) + '\n'
            elif line[:6] == 'dt_max':
                line = 'dt_max = ' + str(dt_max) + '\n'
            newlines.append(line)
        fid = open(fdir+'input', 'w')
        fid.writelines(newlines)
        fid.close()
        #   Generate output folder name
        fout = out_dir + '/out-' + str(tt) + '/'
        '''
            RUN RICH3D
        '''
        print('\n\n BEGINNING SIMULATION : DTMAX = ',dt_max,' \n')
        p = subprocess.Popen(['./rich3d', fdir, fdir+fout, '2'])
        p.wait()
