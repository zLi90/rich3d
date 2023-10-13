"""
    Validate the NN prediction by running Rich3D on the new dataset
	INPUT:
		NN predictions on the new dataset
	OUTPUT:
		Rich3D simulation results

"""
import numpy as np
import matplotlib.pyplot as plt
import random
import subprocess
import os

"""
    Read the dataset : [Ks, alpha, n, wcr, wcs, p1, p2, dtmax]
"""

dname = ['p3-batch/PC-Optimal', 'p3-batch/MP-Optimal']
mu = 1e-3
rho = 1e3
grav = 9.81
"""
    Run batch rich3d simulations
"""
#	Testing scenarios
fdir = 'p3-batch/'
fsubdir = ['pctest','mptest']
for ff in range(len(dname)):
    vg = np.genfromtxt(dname[ff], delimiter=',')
    print(np.shape(vg))
    os.mkdir(fdir+fsubdir[ff])
    for ivg in range(np.shape(vg)[0]):
        out_dir = fsubdir[ff] + '/vg' + str(ivg)
        os.mkdir(fdir+out_dir)
        tvec = [vg[ivg,5]*0.5, vg[ivg,5], vg[ivg,5]*1.5, vg[ivg,5]*3.0]
        for tt in range(len(tvec)):
            os.mkdir(fdir+out_dir+'/case'+str(int(tt)))
            #   Soil properties
            Ks = vg[ivg,0]
            alpha = vg[ivg,1]
            n = vg[ivg,2]
            wcr = vg[ivg,3]
            wcs = vg[ivg,4]
            dt_max = tvec[tt]
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
            fout = out_dir + '/case'+str(int(tt)) + '/'
            '''
                RUN RICH3D
            '''
            print('\n\n BEGINNING SIMULATION : DTMAX = ',dt_max,' \n')
            p = subprocess.Popen(['./rich3d', fdir, fdir+fout, '2'])
            p.wait()
