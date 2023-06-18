/* -*- mode: c++; c-default-style: "linux" -*- */

/**
 * Subsurface model container
 **/

#ifndef _GW_STATE_H_
#define _GW_STATE_H_

#include "config.h"
#include "GwMatrix.h"
#include <set>
#include <math.h>


// declare functions
inline double h2wc(double h, Config config);
inline double wc2h(double wc, Config config);
inline double wc2kr(double wc, Config config);
inline double h2ch(double h, Config config);

class GwState   {

public:
    // GW variables
    DblArr2 h, q, wc, k;
    DblArr dh;
    // matrix system
    DblArr2 resi, coef;
    int iter;	// number of solver iterations
    int termType;	// if CFl condition is violated and allocation is needed
    std::string initialMode;
    double initialValue;
	char finput[200];
    // Allocate state variables for groundwater
    inline void allocate (Config &config) {
        h = DblArr2("h", config.nall, 2);
        dh = DblArr("dh", config.ndom);
        wc = DblArr2("wc", config.nall, 3);
        k = DblArr2("k", config.nall, 4);
        q = DblArr2("q", config.nall, 3);
        coef = DblArr2("coef", config.ndom, 8);

    }
    
    // Initialize the variables 
    inline void initialize(Config &config)	{
    	if (config.init_file == 1)	{
			strcpy(finput, config.fdir);
			strcat(finput, "init-h");
			FILE *fid;
			fid = fopen(finput, "r");
			if (fid == NULL)	{printf("ERROR: Initial h file does not exist! \n");}
			for (int idx = 0; idx < config.ndom; idx++)	{fscanf(fid, "%lf,", &h(idx,1));}
			for (int idx = 0; idx < config.ndom; idx++)	{wc(idx,1) = h2wc(h(idx,1), config);}
		}
		else if (config.wc_init < config.phi)	{
			for (int idx = 0; idx < config.ndom; idx++)	{
				wc(idx,1) = config.wc_init;
				h(idx,1) = wc2h(wc(idx,1), config);
			}
		}
		else if (config.wt_init < 0.0)	{
			int kwt = floor(-config.wt_init / config.dz);
			for (int idx = 0; idx < config.ndom; idx++)	{
				h(idx,1) = (config.k3d(idx) - kwt) * config.dz;
				wc(idx,1) = h2wc(h(idx,1), config);
			}
			#if TEST_BEEGUM
			double zwt;
			for (int ii = 0; ii < config.ndom; ii++)	{
				slope = (config.bc_val_xp - config.bc_val_xm) / (config.nx * config.dx);
				zwt = (config.i3d(ii)+0.5)*config.dx * slope + config.bc_val_xm;
				h(ii,1) = (config.k3d(ii)+0.5)*config.dz - zwt;
				if (h(ii,1) < -1.25)	{h(ii,1) = -1.25;}
				wc(ii,1) = h2wc(h(ii,1), config);
			}
			#endif
		}
		else	{
			for (int idx = 0; idx < config.ndom; idx++)	{
				h(idx,1) = config.h_init;
				wc(idx,1) = h2wc(h(idx,1), config);
			}
		}
		
		apply_BC(config);
		
		for (int idx = 0; idx < config.nall; idx++)	{
			h(idx,0) = h(idx,1);
			wc(idx,0) = wc(idx,1);	wc(idx,2) = wc(idx,1);
		}
    }
    
    // Apply boundary conditions
	inline void apply_BC(Config &config)	{
		//bc_type = 0 : no flow
		//bc_type = 1 : uniform head
		//bc_type = 2 : uniform flux
		//bc_type = 3 : fixed water table (m from top boundary)
		double z_cell;
		for (int ii = 0; ii < config.nbcell; ii++)	{
			switch (config.bcell(ii,0))	{
				case 0:
					h(config.bcell(ii,2),1) = h(config.bcell(ii,1),1);
					wc(config.bcell(ii,2),1) = wc(config.bcell(ii,1),1);	break;
				case 1:
					h(config.bcell(ii,2),1) = config.bcval(ii);
					wc(config.bcell(ii,2),1) = h2wc(h(config.bcell(ii,2),1), config);	break;
				case 2:
					wc(config.bcell(ii,2),1) = wc(config.bcell(ii,1),1);
					h(config.bcell(ii,2),1) = wc2h(wc(config.bcell(ii,2),1), config);	break;
				case 3:
					z_cell = (config.k3d(config.bcell(ii,1)) + 0.5) * config.dz;
					h(config.bcell(ii,2),1) = z_cell - config.bcval(ii);
					#if TEST_BEEGUM
					if (h(config.bcell(ii,2),1) < -1.25)	{h(config.bcell(ii,2),1) = -1.25;}
					#endif
					wc(config.bcell(ii,2),1) = h2wc(h(config.bcell(ii,2),1), config);	break;
				default:
					printf(" ERROR : BC_TYPE must be 0, 1, 2 or 3!\n");
			}
		
		}
		//#if TEST_TRACY
		int idx;
		double *htop = (double *) malloc(config.nx*config.ny*sizeof(double));
		strcpy(finput, config.fdir);
		strcat(finput, "head_bczm");
		FILE *fid;
		fid = fopen(finput, "r");
		if (fid == NULL)	{printf("ERROR: Top h file does not exist! \n");}
		for (int jj = 0; jj < config.nx*config.ny; jj++)	{
			fscanf(fid, "%lf,", &htop[jj]);
			idx = jj * config.nz;
			for (int ii = 0; ii < config.nbcell; ii++)	{
				if (config.bcell(ii,1) == idx & config.bcell(ii,3) == -3)	{
					h(config.bcell(ii,2),1) = htop[jj];
					wc(config.bcell(ii,2),1) = h2wc(h(config.bcell(ii,2),1), config); break;
				}
			}
		}
		free(htop);
		//#endif
	}

};



/* --------------------------------------------------
    Soil constitutive functions
-------------------------------------------------- */
// Get water content from head

inline double h2wc(double h, Config config)     {
    double m, s, sbar, sr;
    sr = config.wcr / config.phi;
    m = 1.0 - 1.0/config.vgn;
    sbar = pow(1.0 + pow(fabs(config.vga*h), config.vgn), -m);
    s = sbar * (1 - sr) + sr;
    if (h > 0.0 | s > 1.0)  {s = 1.0;}
    else if (s < sr)    {s = sr;}
    #if TEST_TRACY
    return config.wcr + (config.phi-config.wcr) * exp(0.1634*h);
    #else
    return s * config.phi;
    #endif
}
// Get head from water content
inline double wc2h(double wc, Config config)	{
	double m, h, eps = 1e-7;
	m = 1.0 - 1.0/config.vgn;
	if (wc - config.wcr < eps)	{wc = config.wcr + eps;}
	if (wc < config.phi)	{
		h = -(1.0/config.vga) * (pow(pow((config.phi-config.wcr)/(wc-config.wcr),(1/m)) - 1.0, 1/config.vgn));
	}
	else	{h = 0.0;}
	#if TEST_TRACY
	h = log((wc - config.wcr)/(config.phi - config.wcr)) / 0.1634;
	#endif
	return h;
}

// Get relative permeability
inline double wc2kr(double wc, Config config)	{
	double m, h, kr, sbar;
	#if TEST_TRACY
	h = log((wc - config.wcr) / (config.phi - config.wcr)) / 0.1634;
	kr = exp(0.1634 * h);
    #else
    m = 1.0 - 1.0 / config.vgn;
    sbar = (wc - config.wcr) / (config.phi - config.wcr);
    kr = pow(sbar,0.5) * pow(1-pow(1-pow(sbar,1.0/m),m), 2.0);
    #endif
    if (kr < 0.0)	{kr = 0.0;}
    else if (kr > 1.0)	{kr = 1.0;}
    return kr;
}
// Get specific capacity, ch
inline double h2ch(double h, Config config)	{
	double m, deno, nume;
	if (h > 0.0)	{return 0.0;}
	m = 1.0 - 1.0 / config.vgn;
	nume = config.vga*config.vgn*m*(config.phi - config.wcr)*pow(fabs(config.vga*h),config.vgn-1.0);
	deno = pow((1.0 + pow(fabs(config.vga*h),config.vgn)), m+1);
	return nume / deno;
}

#endif
