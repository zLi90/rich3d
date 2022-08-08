"""
	Compare VG and Exp water retention curve
"""
import numpy as np
import matplotlib.pyplot as plt
import math

wcs = 0.35
h_min = -20.0
vga = 4.0
vgn = 2.0
m = 1.0 - 1.0/vgn

n = 1000

h = np.linspace(h_min, 0.0, n)

wc_vg = []
wc_ep = []
kr_vg = []
kr_ep = []
for ii in range(len(h)):
	sbar = (1.0 + (abs(vga*h[ii]))**vgn)**(-m)
	wc_vg.append(sbar * wcs)
	wc_ep.append(wcs * math.exp(0.1634*h[ii]))
	kr_vg.append((sbar**0.5)*(1.0-(1-sbar**(1.0/m))**m)**2.0)
	kr_ep.append(math.exp(0.1634*h[ii]))
	
	
plt.figure(1)

plt.subplot(1,2,1)
plt.plot(h,wc_vg,'b-')
plt.plot(h,wc_ep,'r-')
plt.xlabel('Pressure head [m]')
plt.ylabel('Water content')
plt.legend(['van Genuchten','Exponential'])

plt.subplot(1,2,2)
plt.plot(h,kr_vg,'b-')
plt.plot(h,kr_ep,'r-')
plt.xlabel('Pressure head [m]')
plt.ylabel('Relative permeability')

plt.show()
