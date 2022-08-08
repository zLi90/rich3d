
#ifndef _STATE_H_
#define _STATE_H_

#include "config.h"

class State {

public:
	dualDbl h, wc, wce, matcoef, K, q, Kr, dh;
	char finput[200];
	// Initialization
	inline void init(Config config)	{
		// state variables
		h = dualDbl("h", config.nall, 2);
		dh = dualDbl("dh", config.ndom, 1);
		wc = dualDbl("wc", config.nall, 2);
		wce = dualDbl("wce", config.ndom, 1);
		Kr = dualDbl("Kr", config.nall, 1);
		K = dualDbl("K", config.nall, 3);
		q = dualDbl("q", config.nall, 3);
		// matrix entries
		matcoef = dualDbl("coef", config.ndom, 8);
		// Initialize everything with zero
		dualDbl::t_host h_h = h.h_view;
		dualDbl::t_host h_wc = wc.h_view;
		dualDbl::t_host h_kr = Kr.h_view;
		dualDbl::t_host h_k = K.h_view;
		dualDbl::t_host h_q = q.h_view;
		for (int idx = 0; idx < config.nall; idx++)	{
			h_h(idx,0) = 0.0;	h_h(idx,1) = 0.0;	
			h_wc(idx,0) = 0.0;	h_wc(idx,1) = 0.0;	h_kr(idx,0) = 0.0;
			h_k(idx,0) = 0.0;	h_k(idx,1) = 0.0;	h_k(idx,2) = 0.0;
			h_q(idx,0) = 0.0;	h_q(idx,1) = 0.0;	h_q(idx,2) = 0.0;
		}
		// Apply initial conditions
		// If init_file = 1, read init-h
		// If s < phi, use s_init;
		// Else if wt < 0.0, use hydrostatic pressure with wt_init
		// Else, use h_init
		// >>>>> Read input data from data file <<<<<
		if (config.init_file == 1)	{
			strcpy(finput, config.fdir);
			strcat(finput, "init-h");
			FILE *fid;
			fid = fopen(finput, "r");
			if (fid == NULL)	{printf("ERROR: Initial h file does not exist! \n");}
			for (int idx = 0; idx < config.ndom; idx++)	{fscanf(fid, "%lf,", &h_h(idx,1));}
			for (int idx = 0; idx < config.ndom; idx++)	{h_wc(idx,1) = h2wc(h_h(idx,1), config);}
		}
		else if (config.wc_init < config.phi)	{
			for (int idx = 0; idx < config.ndom; idx++)	{
				h_wc(idx,1) = config.wc_init;
				h_h(idx,1) = wc2h(h_wc(idx,1), config);
			}
		}
		else if (config.wt_init < 0.0)	{
			int kwt = floor(-config.wt_init / config.dz);
			for (int idx = 0; idx < config.ndom; idx++)	{
				h_h(idx,1) = (config.k3d(idx) - kwt) * config.dz;
				h_wc(idx,1) = h2wc(h_h(idx,1), config);
			}
		}
		else	{
			for (int idx = 0; idx < config.ndom; idx++)	{
				h_h(idx,1) = config.h_init;
				h_wc(idx,1) = h2wc(h_h(idx,1), config);
			}
		}
		h.modify<dualDbl::host_mirror_space> ();
		wc.modify<dualDbl::host_mirror_space> ();
		K.modify<dualDbl::host_mirror_space> ();
		Kr.modify<dualDbl::host_mirror_space> ();
		q.modify<dualDbl::host_mirror_space> ();
		// Apply boundary conditions
		apply_BC(config);
	}
	
	// Read one field from the input file
	inline double read_one_input(const char field[], const char fname[]) {
		int n = 1000, found = 0;
		FILE *fid;
		char *ptr, arr[n], *out;
		char out_str[50];
		fid = fopen(fname, "r");
		if (fid == NULL)	{printf(" File %s not correctly opened!\n",fname); return 0;}
		while (fgets(arr, n, fid) != NULL)  {
		    char *elem = strtok(arr, " ");
		    if (strcmp(elem, field) == 0)   {
		        elem = strtok(NULL, " ");
		        out = strtok(NULL, " ");
		        found = 1;  break;
		    }
		}
		if (found == 0)	{printf("WARNING: Input field %s not found!\n", field); return 0;}
		fclose(fid);
		strcpy(out_str, out);
		printf(" Reading input from %s: %s = %s",fname,field,out_str);
		return strtod(out_str, &ptr);
	}
	
	
	// Update variables
	inline void update(Config config)	{
		Kokkos::parallel_for(config.ndom, Update<dspace>(h, wc, config));
		Kokkos::fence();
	}
	
	
	// Apply boundary conditions
	inline void apply_BC(Config config)	{
		//bc_type = 0 : no flow
		//bc_type = 1 : uniform head
		//bc_type = 2 : uniform flux
		//bc_type = 3 : fixed water table
		dualDbl::t_host h_h = h.h_view;
		dualDbl::t_host h_wc = wc.h_view;
		for (int ii = 0; ii < config.nbcell; ii++)	{
			switch (config.bcell(ii,0))	{
				case 0:
					h_h(config.bcell(ii,2),1) = h_h(config.bcell(ii,1),1);
					h_wc(config.bcell(ii,2),1) = h_wc(config.bcell(ii,1),1);	break;
				case 1:
					h_h(config.bcell(ii,2),1) = config.bcval(ii);
					h_wc(config.bcell(ii,2),1) = h2wc(h_h(config.bcell(ii,2),1), config);	break;
				case 2:
					h_wc(config.bcell(ii,2),1) = h_wc(config.bcell(ii,1),1);
					h_h(config.bcell(ii,2),1) = wc2h(h_wc(config.bcell(ii,2),1), config);	break;
				case 3:
					h_h(config.bcell(ii,2),1) = (config.k3d(config.bcell(ii,1)) - config.bcval(ii)) * config.dz;
					// for Beegum Problem
					if (h_h(config.bcell(ii,2),1) < -1.25)	{h_h(config.bcell(ii,2),1) = -1.25;}
					h_wc(config.bcell(ii,2),1) = h2wc(h_h(config.bcell(ii,2),1), config);	break;
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
					h_h(config.bcell(ii,2),1) = htop[jj];
					h_wc(config.bcell(ii,2),1) = h2wc(h_h(config.bcell(ii,2),1), config); break;
				}
			}
		}
		free(htop);
		//#endif
		h.modify<dualDbl::host_mirror_space> ();
		wc.modify<dualDbl::host_mirror_space> ();
	}
	
	// print out all variables for debugging
	inline void print_all(Config config)	{
		dualDbl::t_host h_h = h.h_view;
		dualDbl::t_host h_wc = wc.h_view;
		dualDbl::t_host h_Kr = Kr.h_view;
		dualDbl::t_host h_K = K.h_view;
		dualDbl::t_host h_q = q.h_view;
		for (int idx = 0; idx < config.nall; idx++)	{
			printf("-%d- : h=(%f,%f), wc=(%f,%f), q=(%f,%f,%f), K=(%f,%f,%f), Kr=%f\n",idx,
				h_h(idx,0),h_h(idx,1),h_wc(idx,0),h_wc(idx,1),h_q(idx,0),h_q(idx,1),h_q(idx,2),
				h_K(idx,0),h_K(idx,1),h_K(idx,2),h_Kr(idx,0));
		}
	
	}
	
	// Get hydraulic conductivity on cell faces
	inline void get_conductivity(Config config)	{
		int idx, idm, idp;
		double coef = config.rho * GRAV / config.mu;
		IntArr2 bmap = config.bcell;
		Kokkos::parallel_for(config.nall, KR<dspace>(Kr, h, config));
		config.sync(Kr);
		Kokkos::parallel_for(config.ndom, KFace<dspace>(K, Kr, coef, config));
		config.sync(K);
		
		dualDbl::t_host h_wc = wc.h_view;
		dualDbl::t_host h_K = K.h_view;
		for (int ii = 0; ii < config.nbcell; ii++)	{
			idx = bmap(ii,1);
			// x
			if (bmap(ii,3) == -1)	{
				idm = bmap(ii,2);
				if (bmap(ii,0) == 0)	{h_K(idm,0) = 0.0;}
				else	{h_K(idm,0) = get_kface(h_wc(idx,1), h_wc(idm,1), config.kx, config);}
			}
			else if (bmap(ii,3) == 1)	{
				idp = bmap(ii,2);
				if (bmap(ii,0) == 0)	{h_K(idx,0) = 0.0;}
				else	{h_K(idx,0) = get_kface(h_wc(idx,1), h_wc(idp,1), config.kx, config);}
			}
			//y
			if (bmap(ii,3) == -2)	{
				idm = bmap(ii,2);
				if (bmap(ii,0) == 0)	{h_K(idm,1) = 0.0;}
				else	{h_K(idm,1) = get_kface(h_wc(idx,1), h_wc(idm,1), config.ky, config);}
			}
			else if (bmap(ii,3) == 2)	{
				idp = bmap(ii,2);
				if (bmap(ii,0) == 0)	{h_K(idx,1) = 0.0;}
				else	{h_K(idx,1) = get_kface(h_wc(idx,1), h_wc(idp,1), config.ky, config);}
			}
			//z
			if (bmap(ii,3) == -3)	{
				idm = bmap(ii,2);
				if (bmap(ii,0) == 0)	{h_K(idm,2) = 0.0;}
				else	{h_K(idm,2) = get_kface(h_wc(idx,1), h_wc(idm,1), config.kz, config);}
				// inundated top boundary
				if (bmap(ii,0) == 1 & config.bc_val_zm >= 0.0)	{
					h_K(idm,2) = config.rho * GRAV * config.kz / config.mu;
				}
			}
			else if (bmap(ii,3) == 3)	{
				idp = bmap(ii,2);
				if (bmap(ii,0) == 0)	{h_K(idx,2) = 0.0;}
				else	{h_K(idx,2) = get_kface(h_wc(idx,1), h_wc(idp,1), config.kz, config);}
			}
		}
		K.modify<dualDbl::host_mirror_space> ();
	}
	
	
	
	// Get face flux 
	inline void get_flux(Config config)	{
		int idx, idex;
		double coef = config.rho * GRAV / config.mu;
		Kokkos::parallel_for(config.ndom, Face_Flux<dspace>(q, K, h, config));
		config.sync(q);
		dualDbl::t_host h_h = h.h_view;
		dualDbl::t_host h_K = K.h_view;
		dualDbl::t_host h_q = q.h_view;
		dualDbl::t_host h_wc = wc.h_view;
		for (int ii = 0; ii < config.nbcell; ii++)	{
			idx = config.bcell(ii,1);
			idex = config.bcell(ii,2);
			switch (config.bcell(ii,3))	{
				case -1:
					if (config.bcell(ii,0) == 2)	{h_q(idex,0) = config.bcval(ii);}
					else	{
						h_q(idex,0) = 2.0*h_K(idex,0)*(h_h(idx,1) - h_h(idex,1)) / config.dx;
					}
					break;
				case 1:
					if (config.bcell(ii,0) == 2)	{h_q(idx,0) = config.bcval(ii);}	
					else {h_q(idx,0) = h_q(idx,0)*2.0;}	break;
				case -2:
					if (config.bcell(ii,0) == 2)	{h_q(idex,1) = config.bcval(ii);}
					else	{h_q(idex,1) = 2.0*h_K(idex,1)*(h_h(idx,1) - h_h(idex,1)) / config.dy;}	break;
				case 2:
					if (config.bcell(ii,0) == 2)	{h_q(idx,1) = config.bcval(ii);}
					else {h_q(idx,1) = h_q(idx,1)*2.0;}	break;
				case -3:
					if (config.bcell(ii,0) == 2)	{h_q(idex,2) = config.bcval(ii);}
					else if (config.bcell(ii,0) == 1 & config.bcval(ii) >= 0.0)	{
						h_q(idex,2) = coef*config.kz*(h_h(idx,1) - h_h(idex,1)) / (config.dz/2.0) - coef*config.kz;
					}
					else	{h_q(idex,2) = h_K(idex,2)*(h_h(idx,1) - h_h(idex,1)) / (config.dz/2.0) - h_K(idex,2);}
					break;
				case 3:
					if (config.bcell(ii,0) == 2)	{h_q(idx,2) = config.bcval(ii);}	
					else {h_q(idx,2) = h_K(idx,2)*(h_h(idex,1) - h_h(idx,1)) / (config.dz/2.0) - h_K(idx,2);}	break;
			}
		}
		q.modify<dualDbl::host_mirror_space> ();
	}
	
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
	
	// Get mean conductivity on cell faces
	inline double get_kface(double wc1, double wc2, double ks, Config config)  {
        double k1, k2, coef;
        coef = config.rho * GRAV / config.mu;
        k1 = wc2kr(wc1, config);
        k2 = wc2kr(wc2, config);
        return 0.5 * coef * ks * (k1 + k2);
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
    
    // Adjust dt based on net change of water content
    inline void dt_waco(Config &config)	{
    	double dwc_max, dt_old;
    	dt_old = config.dt;
    	Kokkos::parallel_reduce(config.ndom, Max_dwc<dspace>(wc), dwc_max);
    	if (dwc_max > 0.02)	{config.dt = config.dt * 0.5;}
    	else if (dwc_max > 0.0 & dwc_max < 0.01)	{config.dt = config.dt * 1.2;}
    	if (config.dt > config.dt_max)	{config.dt = config.dt_max;}
    	else if (config.dt < config.dt_init)	{config.dt = config.dt_init;}
    	if (config.dt != dt_old)	{
    		printf("     >> max dwc = %f, dt is changed to %f sec\n",dwc_max,config.dt);
    	}
    }
    
    // Adjust dt based on the number of iterations
    inline void dt_iter(int iter, Config &config)	{
    	double dt_old;
    	dt_old = config.dt;
    	if (iter < 5)	{config.dt = config.dt * 1.2;}
    	else if (iter > 8)	{config.dt = config.dt * 0.75;}
    	if (config.dt > config.dt_max)	{config.dt = config.dt_max;}
    	else if (config.dt < config.dt_init)	{config.dt = config.dt_init;}
    	if (config.dt != dt_old)	{
    		printf("     >> iter = %d, dt is changed to %f sec\n",iter,config.dt);
    	}
    }
    
    
    /*
    	Parallel functions
    */
    // get maximum change of water content
    template<class ExecutionSpace>
    struct Max_dwc {
    	typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value, dualDbl::memory_space, dualDbl::host_mirror_space>::type ms;
   		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> wc;
   		Max_dwc(dualDbl d_wc)	{
   			wc = d_wc.template view<ms> ();	d_wc.sync<ms> ();
   		}
   		
   		KOKKOS_INLINE_FUNCTION void
		operator() (const int idx, double& dwc_max) const	
		{
			double dwc = fabs(wc(idx,1)-wc(idx,0));
			dwc_max = (dwc > dwc_max) ? dwc : dwc_max;
		}

		KOKKOS_INLINE_FUNCTION void
		join (volatile double& dst, const volatile double& src) const	
		{dst = (src > dst) ? src : dst;}
    
    	KOKKOS_INLINE_FUNCTION void	init (double& dst) const	{dst = 0.0;}
    };
    
    // Kernel for update primary variables
    template<class ExecutionSpace>
    struct Update {
    	typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value, dualDbl::memory_space, dualDbl::host_mirror_space>::type ms;
		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> h;
		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> wc;
   		Update(dualDbl d_h, dualDbl d_wc, Config config)	{
   			h = d_h.template view<ms> ();	d_h.sync<ms> ();	d_h.modify<ms> ();
   			wc = d_wc.template view<ms> ();	d_wc.sync<ms> ();	d_wc.modify<ms> ();
   		}
    	KOKKOS_INLINE_FUNCTION	void operator() (const int idx) const {
        	h(idx,0) = h(idx,1);	wc(idx,0) = wc(idx,1);
    	}
    };
    
    // Kernel for computing conductivity
    template<class ExecutionSpace>
    struct KR {
    	typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value, dualDbl::memory_space, dualDbl::host_mirror_space>::type ms;
		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> kr;
		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> h;
		double m, wcs, wcr, alpha, n;
   		KR(dualDbl d_kr, dualDbl d_h, Config config)	{
   			kr = d_kr.template view<ms> ();	d_kr.sync<ms> ();	d_kr.modify<ms> ();
   			h = d_h.template view<ms> ();	d_h.sync<ms> ();
   			wcs = config.phi; wcr = config.wcr; m = 1.0 - 1.0/config.vgn;
   			n = config.vgn;	alpha = config.vga;
   		}
    	KOKKOS_INLINE_FUNCTION	void operator() (const int idx) const {
        	double sbar = pow(1.0 + pow(fabs(alpha*h(idx,1)), n), -m);
        	#if TEST_TRACY
        	kr(idx,0) = exp(0.1634 * h(idx,1));
        	#else
        	kr(idx,0) = pow(sbar,0.5) * pow(1-pow(1-pow(sbar,1.0/m),m), 2.0);
        	#endif
        	if (kr(idx,0) < 0.0)	{kr(idx,0) = 0.0;}
        	else if (kr(idx,0) > 1.0)	{kr(idx,0) = 1.0;}
    	}
    };
    template<class ExecutionSpace>
    struct KFace {
    	typedef ExecutionSpace execution_space;
    	typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value, dualDbl::memory_space, dualDbl::host_mirror_space>::type ms;
		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> k;
		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> kr;
		double coef, kx, ky, kz;
		IntArr iP, jP, kP;
   		KFace(dualDbl d_k, dualDbl d_kr, double p1, Config config)	{
   			kr = d_kr.template view<ms> ();	d_kr.sync<ms> ();
   			k = d_k.template view<ms> ();	d_k.sync<ms> ();	d_k.modify<ms> ();
   			coef = p1; 
   			kx = config.kx;	ky = config.ky;	kz = config.kz;
   			iP = config.iP;	jP = config.jP;	kP = config.kP;
   		}
    	KOKKOS_INLINE_FUNCTION	void operator() (const int idx) const {
        	k(idx,0) = 0.5 * coef * kx * (kr(idx,0) + kr(iP(idx),0));
        	k(idx,1) = 0.5 * coef * ky * (kr(idx,0) + kr(jP(idx),0));
        	k(idx,2) = 0.5 * coef * kz * (kr(idx,0) + kr(kP(idx),0));
    	}
    };
    
    // Kernel for computing flux
    template<class ExecutionSpace>
    struct Face_Flux {
    	typedef ExecutionSpace execution_space;
    	typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value, dualDbl::memory_space, dualDbl::host_mirror_space>::type ms;
		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> q;
		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> k;
		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> h;
		double dx, dy, dz;
		IntArr iP, jP, kP;
   		Face_Flux(dualDbl d_q, dualDbl d_k, dualDbl d_h, Config config)	{
   			q = d_q.template view<ms> ();	d_q.sync<ms> ();	d_q.modify<ms> ();
   			k = d_k.template view<ms> ();	d_k.sync<ms> ();
   			h = d_h.template view<ms> ();	d_h.sync<ms> ();
   			dx = config.dx;	dy = config.dy;	dz = config.dz;
   			iP = config.iP;	jP = config.jP;	kP = config.kP;
   		}
    	KOKKOS_INLINE_FUNCTION	void operator() (const int idx) const {
    		q(idx,0) = k(idx,0) * (h(iP(idx),1) - h(idx,1)) / dx;
    		q(idx,1) = k(idx,1) * (h(jP(idx),1) - h(idx,1)) / dy;
    		q(idx,2) = k(idx,2) * (h(kP(idx),1) - h(idx,1)) / dz - k(idx,2);
    	}
    };
    
};
#endif

