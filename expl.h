/*
	Fully explicit scheme
*/
#ifndef _EXPL_H_
#define _EXPL_H_

#include "config.h"
#include "state.h"

class Expl : public State {

public:
    
    /*
    	------------------------------------------------------
    	------------------------------------------------------
		Compute water content for fully explicit scheme
		------------------------------------------------------
		------------------------------------------------------
	*/
	// update water content
	inline void update_wc(Config config)	{
		Kokkos::parallel_for(config.ndom, Update_WC<dspace>(wc, h, q, config));
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
    	double ss, wcs, wcr, alpha, m, n, dx, dy, dz, dt;
    	IntArr iM, jM, kM;
    	Update_WC(dualDbl d_wc, dualDbl d_h, dualDbl d_q, Config config)	{
    		q = d_q.template view<ms> ();	d_q.sync<ms> ();
    		h = d_h.template view<ms> ();	d_h.sync<ms> ();	d_h.modify<ms> ();
    		wc = d_wc.template view<ms> ();	d_wc.sync<ms> ();	d_wc.modify<ms> ();
    		ss = config.ss;	wcs = config.phi;	wcr = config.wcr;
    		iM = config.iM;	jM = config.jM;	kM = config.kM;
    		dx = config.dx;	dy = config.dy;	dz = config.dz;	dt = config.dt;
    		alpha = config.vga;	n = config.vgn;	m = 1.0 - 1.0/config.vgn;
    	}
    	KOKKOS_INLINE_FUNCTION	void operator() (const int idx) const {
    		double qqx, qqy, qqz;
    		qqx = (q(idx,0) - q(iM(idx),0)) / dx;
    		qqy = (q(idx,1) - q(jM(idx),1)) / dy;
    		qqz = (q(idx,2) - q(kM(idx),2)) / dz;
    		
    		// wc based
    		wc(idx,1) = (wc(idx,0) + dt * (qqx + qqy + qqz));
    		#if TEST_TRACY
			h(idx,1) = log((wc(idx,1) - wcr)/(wcs - wcr)) / 0.1634;
			#else
			h(idx,1) = -(1.0/alpha) * (pow(pow((wcs-wcr)/(wc(idx,1)-wcr),(1/m)) - 1.0, 1/n));
			#endif
			if (wc(idx,1) > wcs)	{wc(idx,1) = wcs;}
    		else if (wc(idx,1) < wcr+0.01)	{wc(idx,1) = wcr + 0.01;}    
    		
    		// head-based
    		/*double nume, deno, ch, s, sbar, coef;
    		#if TEST_TRACY
    		ch = 0.1634 * (wcs-wcr) * exp(0.1634 * h(idx,0));
    		#else
    		nume = alpha*n*m*(wcs-wcr)*pow(fabs(alpha*h(idx,0)),n-1);
    		deno = pow((1.0 + pow(fabs(alpha*h(idx,0)),n)), m+1);
    		ch = nume / deno;
    		#endif
    		
    		coef = 1.0/(ss*wc(idx,0)/wcs + ch);
    		h(idx,1) = h(idx,0) + dt * coef * (qqx + qqy + qqz);
    		
    		#if TEST_TRACY
			wc(idx,1) = wcr + (wcs-wcr) * exp(0.1634*h(idx,1));
			#else
			sbar = pow(1.0 + pow(fabs(alpha*h(idx,1)), n), -m);
		    wc(idx,1) = sbar * (wcs - wcr) + wcr;
			#endif
			if (wc(idx,1) > wcs)	{wc(idx,1) = wcs;}
    		else if (wc(idx,1) < wcr+0.01)	{wc(idx,1) = wcr + 0.01;}   */
    	}
    };

};
#endif
