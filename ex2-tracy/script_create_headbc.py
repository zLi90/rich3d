"""
    Create top boundary condition for Tracy's problem
"""
import numpy as np
import matplotlib.pyplot as plt
import math
from analytical import *

nx = 80
ny = 48
dx = 0.125
dy = 0.125

gamma = 0.1634
h0 = -12.0
L = nx*dx
W = ny*dy
H = 8.0

head = np.zeros((nx,ny), dtype=float)

solver = Tracy_analytic([L,W,H], 1e-4, 0.35, 0.016, gamma)
for ii in range(nx):
    x = (ii+0.5)*dx
    for jj in range(ny):
        y = (jj+0.5)*dy
        head[ii,jj] = solver.get_solution(x, y, H, 0.0, h0)

head1d = np.reshape(head, (nx*ny,1), order='F');

np.savetxt('head_bczm',head1d)



plt.figure(1)
plt.imshow(head, cmap='jet')


plt.show()
