/*
	Picard scheme
*/
#ifndef _PICARD_H_
#define _PICARD_H_

#include "config.h"
#include "state.h"

class Picard : public State {

public:

	/*
		------------------------------------------------------
    	------------------------------------------------------
		Compute Matrix coefficients for Picard scheme
		------------------------------------------------------
    	------------------------------------------------------
	*/
	inline void linear_system(Config config)	{
		int icol, iflux, idx;
		Kokkos::parallel_for(config.ndom, MatCoef<dspace>(matcoef, h, wc, K, config));
		config.sync(matcoef);

		Kokkos::parallel_for(config.nbcell, DirchBC<dspace>(matcoef, h, config));
		config.sync(matcoef);

		Kokkos::parallel_for(config.ndom, getDiag<dspace>(matcoef));
		config.sync(matcoef);
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
    		coef(idx,7) = (ch + ss*wc(idx,1)/wcs)*h_h - dt*(k(idx,2) - k(kM(idx),2)) / dz
				- (wc(idx,1) - wc(idx,0));
    	}
    };

    // Adjust head bc
    template<class ExecutionSpace>
    struct DirchBC {
    	typedef ExecutionSpace execution_space;
    	typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value, dualDbl::memory_space, dualDbl::host_mirror_space>::type ms;
		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> coef;
		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> h;
		IntArr2 bcell;
   		DirchBC(dualDbl d_coef, dualDbl d_h, Config config)	{
   			coef = d_coef.template view<ms> ();	d_coef.sync<ms> ();	d_coef.modify<ms> ();
   			h = d_h.template view<ms> ();	d_h.sync<ms> ();
			bcell = config.bcell;
   		}
    	KOKKOS_INLINE_FUNCTION	void operator() (const int ii) const {
    		int icol, idx;
    		idx = bcell(ii,1);
			switch (bcell(ii,3))	{
				case -1:	icol = 2;	break;
				case  1:	icol = 1;	break;
				case -2:	icol = 4;	break;
				case  2:	icol = 3;	break;
				case -3:	icol = 6;	break;
				case  3:	icol = 5;	break;
				default:	printf(" ERROR : Invalid bc axis!\n");
			}
			coef(idx,icol) = coef(idx,icol) * 2.0;
			coef(idx,7) += coef(idx,icol) * h(bcell(ii,2),1);
    	}
    };

    // Get diagonal elements
    template<class ExecutionSpace>
    struct getDiag {
    	typedef ExecutionSpace execution_space;
    	typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value, dualDbl::memory_space, dualDbl::host_mirror_space>::type ms;
		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> coef;
   		getDiag(dualDbl d_coef)	{
   			coef = d_coef.template view<ms> ();	d_coef.sync<ms> ();	d_coef.modify<ms> ();
   		}
    	KOKKOS_INLINE_FUNCTION	void operator() (const int ii) const {
    		coef(ii,0) -= (coef(ii,1)+coef(ii,2)+coef(ii,3)+coef(ii,4)+coef(ii,5)+coef(ii,6));
    	}
    };



	/*
		------------------------------------------------------
    	------------------------------------------------------
		Get water content from water retention curve
		------------------------------------------------------
    	------------------------------------------------------
	*/
	inline void update_wc(Config config)	{
		Kokkos::parallel_for(config.ndom, Update_WC<dspace>(wc, h, config));
		config.sync(wc);
	}
	template<class ExecutionSpace>
    struct Update_WC	{
    	typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value, dualDbl::memory_space, dualDbl::host_mirror_space>::type ms;
    	Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> wc;
    	Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> h;
    	double ss, wcs, wcr, alpha, n, m;
    	Update_WC(dualDbl d_wc, dualDbl d_h, Config config)	{
    		h = d_h.template view<ms> ();	d_h.sync<ms> ();
    		wc = d_wc.template view<ms> ();	d_wc.sync<ms> ();	d_wc.modify<ms> ();
    		ss = config.ss;	wcs = config.phi; wcr = config.wcr;
    		alpha = config.vga;	n = config.vgn; m = 1.0 - 1.0/config.vgn;
    	}
    	KOKKOS_INLINE_FUNCTION	void operator() (const int idx) const {
    		double sbar;
    		#if TEST_TRACY
    		sbar = exp(0.1634 * h(idx,1));
    		#else
    		sbar = pow(1.0 + pow(fabs(alpha*h(idx,1)), n), -m);
    		#endif
    		wc(idx,1) = sbar * (wcs - wcr) + wcr;
    		if (h(idx,1) > 0.0 | wc(idx,1) > wcs)	{wc(idx,1) = wcs;}
    		else if (wc(idx,1) < wcr)	{wc(idx,1) = wcr;}
    	}
    };

	/*
		------------------------------------------------------
    	------------------------------------------------------
		Get maximum relative residual
		------------------------------------------------------
    	------------------------------------------------------
	*/
	inline double get_eps(Config config)	{
		double eps, eps_min;
		Kokkos::parallel_reduce(config.ndom, Max_eps<dspace>(h), eps);
		return sqrt(eps)/config.ndom;
	}
    template<class ExecutionSpace>
    struct Max_eps {
    	typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value, dualDbl::memory_space, dualDbl::host_mirror_space>::type ms;
   		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> h;
   		Max_eps(dualDbl d_h)	{
   			h = d_h.template view<ms> ();	d_h.sync<ms> ();
   		}
   		KOKKOS_INLINE_FUNCTION void
		operator() (const int idx, double& rdh_sum) const
		{
			rdh_sum += fabs(h(idx,1) - h(idx,0));
		}
		KOKKOS_INLINE_FUNCTION void
		join (volatile double& dst, const volatile double& src) const
		{dst += src;}
    	KOKKOS_INLINE_FUNCTION void	init (double& dst) const	{dst = 0.0;}
    };

};
#endif
