""" Analytical solution of 3D Richards by Tracy """
import numpy as np
import math


'''
    Define solution class
'''
class Tracy_analytic():
    #   Initialization
    def __init__(self, dim, Ks, wcs, wcr, ga):
        self.L = dim[0]
        self.W = dim[1]
        self.H = dim[2]
        self.Ks = Ks
        self.wcs = wcs
        self.wcr = wcr
        self.ga = ga

    #   Get steady-state solution at (x, y, z, t)
    def get_ss(self, x, y, z, t, h0):
        beta = (0.25*self.ga**2.0 + (math.pi/self.L)**2.0 + (math.pi/self.W)**2.0)**0.5
        term1 = (1.0 - math.exp(self.ga*h0)) * math.sin(math.pi*x/self.L) * math.sin(math.pi*y/self.W)
        term2 = math.exp(0.5*self.ga*(self.H - z)) * math.sinh(beta*z) / math.sinh(beta*self.H)
        return term1 * term2

    #   Get the phi term
    def get_phi(self, x, y, z, t, h0, N=10000):
        c = self.ga * (self.wcs - self.wcr) / self.Ks
        beta = (0.25*self.ga**2.0 + (math.pi/self.L)**2.0 + (math.pi/self.W)**2.0)**0.5
        term1 = (2.0*(1.0 - math.exp(self.ga*h0))/(self.H*c)) * math.sin(math.pi*x/self.L) * math.sin(math.pi*y/self.W)
        term2 = math.exp(0.5*self.ga*(self.H - z))
        term3 = 0.0
        for ii in range(1, N):
            lam = ii*math.pi/self.H
            gamma = (beta**2.0 + lam**2.0) / c
            term3 += (-1)**ii * (lam /gamma) * math.sin(lam*z) * math.exp(-gamma*t)
            if abs((-1)**ii * (lam /gamma) * math.sin(lam*z) * math.exp(-gamma*t)) / abs(term3) < 0.0001:
                break
            # if z==7.15 and x == 5.15 and y == 3.75:
            #     print(t, ' : ', ii, term3, (-1)**ii, (lam /gamma), math.sin(lam*z), math.exp(-gamma*t))
        return term1 * term2 * term3

    #   Get transient solution
    def get_solution(self, x, y, z, t, h0):
        hss = self.get_ss(x, y, z, t, h0)
        phi = self.get_phi(x, y, z, t, h0)
        if math.exp(self.ga*h0) + phi + hss <= 0.0:
            h = np.nan
        else:
            h = math.log(math.exp(self.ga*h0) + phi + hss) / self.ga
        return h

    #   Get steady-state solution
    def get_ss_solution(self, x, y, z, h0):
        hss = self.get_ss(x, y, z, 0.0, h0)
        h = math.log(math.exp(self.ga*h0) + hss) / self.ga
        return h

    #   Get water retention curve
    def get_wch(self, h):
        wc = self.wcr + (self.wcs - self.wcr)*math.exp(self.ga*h)
        return wc

    #   Get relative permeability
    def get_kr(self, h):
        kr = math.exp(self.ga*h)
        return kr
