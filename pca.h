/*
	PCA scheme
*/
#ifndef _PCA_H_
#define _PCA_H_

#include "config.h"
#include "state.h"

class Pca : public State {

public:

	/*
		------------------------------------------------------
    	------------------------------------------------------
		Compute Matrix coefficients for PCA scheme
		------------------------------------------------------
    	------------------------------------------------------
	*/
	inline void linear_system(Config config)	{
		int icol, iflux, idx;
		Kokkos::parallel_for(config.ndom, MatCoef<dspace>(matcoef, h, wc, K, config));
		config.sync(matcoef);
		dualDbl::t_host h_h = h.h_view;
		dualDbl::t_host h_k = K.h_view;
		dualDbl::t_host h_coef = matcoef.h_view;
		
		dualDbl::t_host h_wc = wc.h_view;
		dualDbl::t_host h_q = q.h_view;
		// adjust boundaries
		for (int ii = 0; ii < config.nbcell; ii++)	{
			idx = config.bcell(ii,1);
			switch (config.bcell(ii,3))	{
				case -1:	icol = 2;	break;
				case  1:	icol = 1;	break;
				case -2:	icol = 4;	break;
				case  2:	icol = 3;	break;
				case -3:	icol = 6;	break;
				case  3:	icol = 5;	break;
				default:	printf(" ERROR : Invalid bc axis!\n");
			}
			h_coef(idx,icol) = h_coef(idx,icol) * 2.0;	
			h_coef(idx,7) += h_coef(idx,icol) * h_h(config.bcell(ii,2),1);
			
			/*if (config.i3d(idx)==0 & config.j3d(idx)==0 & config.k3d(idx)==0&icol==6)	{
				printf(" h=%f, wc=%f, q=(%f,%f)\n",h_h(idx,1),h_wc(idx,1),1e10*h_q(idx,2),1e10*h_q(config.bcell(ii,2),2));
				//printf(" (%d,%d,%d,%d) : h_coef=%f, hh=%f \n",config.bcell(ii,0),config.bcell(ii,1),
					//config.bcell(ii,2),config.bcell(ii,3),1e5*h_coef(idx,icol),h_h(config.bcell(ii,2),1));
			
			}*/
		}
		for (int ii = 0; ii < config.ndom; ii++)	{
			h_coef(ii,0) -= (h_coef(ii,1)+h_coef(ii,2)+h_coef(ii,3)+h_coef(ii,4)+h_coef(ii,5)+h_coef(ii,6));
		}
		
		// flux boundary condition
		for (int ii = 0; ii < config.nbcell; ii++)	{
			if (config.bcell(ii,0) == 2)	{
				idx = config.bcell(ii,1);
				switch (config.bcell(ii,3))	{
					case -1:	icol = 2;	iflux = 0;	break;
					case  1:	icol = 1;	iflux = 0;	break;
					case -2:	icol = 4;	iflux = 1;	break;
					case  2:	icol = 3;	iflux = 1;	break;
					case -3:	icol = 6;	iflux = 2;	break;
					case  3:	icol = 5;	iflux = 3;	break;
					default:	printf(" ERROR : Invalid bc axis!\n");
				}
				h_coef(idx,0) += h_coef(idx,icol);
				h_coef(idx,7) -= h_coef(idx,icol) * h_h(config.bcell(ii,2),1);
				h_coef(idx,7) -= config.dt * (config.bcval(ii) - h_k(config.kM(idx),iflux)) / config.dz;
			}
		}
		matcoef.modify<dualDbl::host_mirror_space> ();
	}
	// Kernel for computing matrix coefficients
    template<class ExecutionSpace>
    struct MatCoef {
    	typedef ExecutionSpace execution_space;
    	typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value, dualDbl::memory_space, dualDbl::host_mirror_space>::type ms;
		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> coef;
		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> h;
		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> wc;
		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> k;
		double dt, dx, dy, dz, ss, wcs, wcr, m, n, alpha;
		IntArr iM, jM, kM;
   		MatCoef(dualDbl d_coef, dualDbl d_h, dualDbl d_wc, dualDbl d_k, Config config)	{
   			coef = d_coef.template view<ms> ();	d_coef.sync<ms> ();	d_coef.modify<ms> ();
   			k = d_k.template view<ms> ();	d_k.sync<ms> ();
   			h = d_h.template view<ms> ();	d_h.sync<ms> ();
   			wc = d_wc.template view<ms> ();	d_wc.sync<ms> ();
			dt = config.dt;	dx = config.dx;	dy = config.dy;	dz = config.dz;
			iM = config.iM;	jM = config.jM;	kM = config.kM;
			ss = config.ss;	wcs = config.phi;	wcr = config.wcr;
			n = config.vgn;	alpha = config.vga;	m = 1.0 - 1.0/config.vgn;
   		}
    	KOKKOS_INLINE_FUNCTION	void operator() (const int idx) const {
    		double nume, deno, h_h = h(idx,1), ch = 0.0;
    		if (h_h < 0.0)	{
				#if TEST_TRACY
				ch = 0.1634 * (wcs - wcr) * exp(0.1634*h_h);
				#else
    			nume = alpha*n*m*(wcs - wcr)*pow(fabs(alpha*h_h),n-1.0);
				deno = pow((1.0 + pow(fabs(alpha*h_h),n)), m+1);
				ch = nume / deno;
				#endif
			}
    		coef(idx,0) = ch + ss*wc(idx,1)/wcs;
    		coef(idx,1) = - dt * k(idx,0) / pow(dx, 2.0);
    		coef(idx,2) = - dt * k(iM(idx),0) / pow(dx, 2.0);
    		coef(idx,3) = - dt * k(idx,1) / pow(dy, 2.0);
    		coef(idx,4) = - dt * k(jM(idx),1) / pow(dy, 2.0);
    		coef(idx,5) = - dt * k(idx,2) / pow(dz, 2.0);
    		coef(idx,6) = - dt * k(kM(idx),2) / pow(dz, 2.0);
    		coef(idx,7) = (ch + ss*wc(idx,1)/wcs)*h_h - dt*(k(idx,2) - k(kM(idx),2)) / dz;
    	}
    };
    
    
    /*
    	------------------------------------------------------
    	------------------------------------------------------
		Compute water content for PCA scheme
		------------------------------------------------------
		------------------------------------------------------
	*/
	// update water content
	inline void update_wc(Config config)	{
		Kokkos::parallel_for(config.ndom, Update_WC<dspace>(wc, h, q, config));
		config.sync(wc);
		Kokkos::parallel_for(config.ndom, Check_Sat<dspace>(wc, wce, h, config));
		config.sync(wc);
		config.sync(h);
	}
	// Kernel for updating water content
    template<class ExecutionSpace>
    struct Update_WC	{
    	typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value, dualDbl::memory_space, dualDbl::host_mirror_space>::type ms;
    	Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> wc;
    	Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> h;
    	Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> q;
    	double ss, wcs, dx, dy, dz, dt;
    	IntArr iM, jM, kM;
    	Update_WC(dualDbl d_wc, dualDbl d_h, dualDbl d_q, Config config)	{
    		q = d_q.template view<ms> ();	d_q.sync<ms> ();
    		h = d_h.template view<ms> ();	d_h.sync<ms> ();
    		wc = d_wc.template view<ms> ();	d_wc.sync<ms> ();	d_wc.modify<ms> ();
    		ss = config.ss;	wcs = config.phi;
    		iM = config.iM;	jM = config.jM;	kM = config.kM;
    		dx = config.dx;	dy = config.dy;	dz = config.dz;	dt = config.dt;
    	}
    	KOKKOS_INLINE_FUNCTION	void operator() (const int idx) const {
    		double coef, qqx, qqy, qqz;
    		coef = 1.0 + ss*(h(idx,1) - h(idx,0)) / wcs;
    		qqx = (q(idx,0) - q(iM(idx),0)) / dx;
    		qqy = (q(idx,1) - q(jM(idx),1)) / dy;
    		qqz = (q(idx,2) - q(kM(idx),2)) / dz;
    		wc(idx,1) = (wc(idx,0) + dt * (qqx + qqy + qqz)) / coef;
    	}
    };
    
    // Choose between h and wc
    template<class ExecutionSpace>
    struct Check_Sat	{
    	typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value, dualDbl::memory_space, dualDbl::host_mirror_space>::type ms;
    	Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> wc;
    	Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> wce;
    	Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> h;
    	double wcs, alpha, n, m, sr, wcr;
    	int ndom;
    	IntArr iM, jM, kM, iP, jP, kP, i, j, k;
    	Check_Sat(dualDbl d_wc, dualDbl d_wce, dualDbl d_h, Config config)	{
    		h = d_h.template view<ms> ();	d_h.sync<ms> ();	d_h.modify<ms> ();
    		wc = d_wc.template view<ms> ();	d_wc.sync<ms> ();	d_wc.modify<ms> ();
    		wce = d_wce.template view<ms> ();	d_wce.sync<ms> ();	d_wce.modify<ms> ();
    		wcs = config.phi;	ndom = config.ndom;
    		iM = config.iM;	jM = config.jM;	kM = config.kM;
    		iP = config.iP;	jP = config.jP;	kP = config.kP;
    		i = config.i3d;	j = config.j3d;	k = config.k3d;
    		alpha = config.vga;	n = config.vgn;	m = 1.0 - 1.0/config.vgn;
    		sr = config.wcr / config.phi;	wcr = config.wcr;
    	}
    	KOKKOS_INLINE_FUNCTION	void operator() (const int idx) const {
    		double sbar;
    		wce(idx,0) = 0.0;
    		if (wc(idx,1) >= wcs - 1e-7)	{
    			wce(idx,0) += wc(idx,1) - wcs;
    			wc(idx,1) = wcs;
    		}
    		else if (wc(kP(idx),1) >= wcs | wc(kM(idx),1) >= wcs)	{
    			#if TEST_TRACY
    			sbar = exp(0.1634*h(idx,1));
    			#else
    			sbar = pow(1.0 + pow(fabs(alpha*h(idx,1)), n), -m);
    			#endif
    			wce(idx,0) += wc(idx,1) - (sbar * (wcs - wcr) + wcr);
    			wc(idx,1) = sbar * (wcs - wcr) + wcr;
    		}
    		else if (wc(iP(idx),1) >= wcs | wc(iM(idx),1) >= wcs)	{
    			#if TEST_TRACY
    			sbar = exp(0.1634*h(idx,1));
    			#else
    			sbar = pow(1.0 + pow(fabs(alpha*h(idx,1)), n), -m);
    			#endif
    			wce(idx,0) += wc(idx,1) - (sbar * (wcs - wcr) + wcr);
    			wc(idx,1) = sbar * (wcs - wcr) + wcr;
    		}
    		else if (wc(jP(idx),1) >= wcs | wc(jM(idx),1) >= wcs)	{
    			#if TEST_TRACY
    			sbar = exp(0.1634*h(idx,1));
    			#else
    			sbar = pow(1.0 + pow(fabs(alpha*h(idx,1)), n), -m);
    			#endif
    			wce(idx,0) += wc(idx,1) - (sbar * (wcs - wcr) + wcr);
    			wc(idx,1) = sbar * (wcs - wcr) + wcr;
    		}
    		else {
    			if (wc(idx,1) <= wcr)	{wc(idx,1) = wcr + 0.01;}
    			else if (wc(idx,1) >= wcs)	{h(idx,1) = 0.0;}
    			else {
    				#if TEST_TRACY
					h(idx,1) = log((wc(idx,1) - wcr)/(wcs - wcr)) / 0.1634;
					#else
					h(idx,1) = -(1.0/alpha) * (pow(pow((wcs-wcr)/(wc(idx,1)-wcr),(1/m)) - 1.0, 1/n));
					#endif
    			}
    		}
    		if (wc(idx,1) > wcs)	{wc(idx,1) = wcs;}
    		else if (wc(idx,1) < wcr+0.01)	{wc(idx,1) = wcr+0.01;}
    	}
    };
    
    
    /*
    	------------------------------------------------------
    	------------------------------------------------------
		Re-distribute water content
		------------------------------------------------------
		------------------------------------------------------
	*/
	inline void allocate_wc(Config config)	{
		Kokkos::parallel_for(config.ndom, Alloc_WC<dspace>(wc, h, wce, config));
		config.sync(wc);
		Kokkos::parallel_for(config.ndom, Cutoff_WC<dspace>(wc, config));
		config.sync(wc);
	}
	
	// Kernel for re-distributing water content
    template<class ExecutionSpace>
    struct Alloc_WC	{
    	typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value, dualDbl::memory_space, dualDbl::host_mirror_space>::type ms;
    	Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> wc;
    	Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> wce;
    	Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> h;
    	double dx, dy, dz;
    	int nx, ny, nz;
    	IntArr iM, jM, kM, iP, jP, kP;
    	Alloc_WC(dualDbl d_wc, dualDbl d_h, dualDbl d_wce, Config config)	{
    		h = d_h.template view<ms> ();	d_h.sync<ms> ();
    		wce = d_wce.template view<ms> ();	d_wce.sync<ms> ();	
    		wc = d_wc.template view<ms> ();	d_wc.sync<ms> ();	d_wc.modify<ms> ();
    		iM = config.iM;	jM = config.jM;	kM = config.kM;
    		iP = config.iP;	jP = config.jP;	kP = config.kP;
    		dx = config.dx;	dy = config.dy;	dz = config.dz;
    		nx = config.nx;	ny = config.ny;	nz = config.nz;
    	}
    	KOKKOS_INLINE_FUNCTION	void operator() (const int idx) const {
    		double dhxp=0.0, dhxm=0.0, dhyp=0.0, dhym=0.0, dhzp=0.0, dhzm=0.0;
    		double dhtot = 0.0, dwc = 0.0;
    		if (wce(idx,0) != 0.0)	{
    			// get head gradients
    			if (nx > 1)	{
    				dhxp = (h(idx,1) - h(iP(idx),1)) / dx;
    				dhxm = (h(idx,1) - h(iM(idx),1)) / dx;
    				dhtot += fabs(dhxp) + fabs(dhxm);
				}
				if (ny > 1)	{
					dhyp = (h(idx,1) - h(jP(idx),1)) / dy;
    				dhym = (h(idx,1) - h(jM(idx),1)) / dy;
    				dhtot += fabs(dhyp) + fabs(dhym);
				}
				if (nz > 1)	{
					dhzp = (h(idx,1) - h(kP(idx),1)) / dz + 1;
    				dhzm = (h(idx,1) - h(kM(idx),1)) / dz - 1;
    				dhtot += fabs(dhzp) + fabs(dhzm);
				}
				// send or receive
				// x
				if (dhxp * dhxm < 0)	{
					dwc = wce(idx,0) * (fabs(dhxp)+fabs(dhxm)) / dhtot;
					if (dhxp > 0)	{wc(iP(idx),1) += dwc;}
					else	{wc(iM(idx),1) += dwc;}
				}
				else {
					if (dhxp > 0 & wce(idx,0) > 0.0)	{
						wc(iP(idx),1) += wce(idx,0) * dhxp / dhtot;	
						wc(iM(idx),1) += wce(idx,0) * dhxm / dhtot;	
					}
					else if (dhxp < 0 & wce(idx,0) < 0.0)	{
						wc(iP(idx),1) += wce(idx,0) * dhxp / dhtot;	
						wc(iM(idx),1) += wce(idx,0) * dhxm / dhtot;	
					}
				}	
				// y
				// z
				if (dhzp * dhzm < 0)	{
					dwc = wce(idx,0) * (fabs(dhzp)+fabs(dhzm)) / dhtot;
					if (dhzp > 0)	{wc(kP(idx),1) += dwc;}
					else	{wc(kM(idx),1) += dwc;}
				}
				else {
					if (dhzp > 0 & wce(idx,0) > 0.0)	{
						wc(kP(idx),1) += wce(idx,0) * dhzp / dhtot;	
						wc(kM(idx),1) += wce(idx,0) * dhzm / dhtot;	
					}
					else if (dhxp < 0 & wce(idx,0) < 0.0)	{
						wc(kP(idx),1) += wce(idx,0) * dhzp / dhtot;	
						wc(kM(idx),1) += wce(idx,0) * dhzm / dhtot;	
					}
				}	
    		
    		}
    		
    	}
    };
    
    // cutoff over- or under-saturation
    template<class ExecutionSpace>
    struct Cutoff_WC	{
    	typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value, dualDbl::memory_space, dualDbl::host_mirror_space>::type ms;
    	Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> wc;
    	double wcs, wcr;
    	Cutoff_WC(dualDbl d_wc, Config config)	{
    		wc = d_wc.template view<ms> ();	d_wc.sync<ms> ();	d_wc.modify<ms> ();
    		wcs = config.phi;	wcr = config.wcr;
    	}
    	KOKKOS_INLINE_FUNCTION	void operator() (const int idx) const {
    		if (wc(idx,1) > wcs)	{wc(idx,1) = wcs;}
    		else if (wc(idx,1) < wcr+0.01)	{wc(idx,1) = wcr + 0.01;}
    	}
    };

};
#endif
