/*
	Newton-Raphson scheme
*/
#ifndef _NEWTON_H_
#define _NEWTON_H_

#include "config.h"
#include "state.h"

class Newton : public State {

public:

	/*
		------------------------------------------------------
    	------------------------------------------------------
		Compute Jacobian coefficients for Newton scheme
		------------------------------------------------------
    	------------------------------------------------------
	*/
	inline void jacobian_system(Config config)	{
		int n = config.ndom;
		Kokkos::parallel_for(n, Residual<dspace>(wc, h, q, matcoef, config));
		config.sync(matcoef);
		Kokkos::parallel_for(n, Jacobian_side<dspace>(h, Kr, K, matcoef, 2, config));
		config.sync(matcoef);
		if (config.nx > 1)	{
			Kokkos::parallel_for(n, Jacobian_side<dspace>(h, Kr, K, matcoef, 0, config));
			config.sync(matcoef);
		}
		if (config.ny > 1)	{
			Kokkos::parallel_for(n, Jacobian_side<dspace>(h, Kr, K, matcoef, 1, config));
			config.sync(matcoef);
		}
		Kokkos::parallel_for(n, Jacobian_cntr<dspace>(h, wc, Kr, matcoef, config));
		config.sync(matcoef);
	}
	
	// Kernel for calculating Newton residual
    template<class ExecutionSpace>
    struct Residual	{
    	typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value, dualDbl::memory_space, dualDbl::host_mirror_space>::type ms;
    	Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> wc;
    	Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> h;
    	Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> q;
    	Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> resi;
    	double ss, wcs, dx, dy, dz, dt, alpha, n, m, wcr;
    	IntArr iM, jM, kM;
    	Residual(dualDbl d_wc, dualDbl d_h, dualDbl d_q, dualDbl d_coef, Config config)	{
    		q = d_q.template view<ms> ();	d_q.sync<ms> ();
    		h = d_h.template view<ms> ();	d_h.sync<ms> ();
    		wc = d_wc.template view<ms> ();	d_wc.sync<ms> ();	
    		resi = d_coef.template view<ms> ();	d_coef.sync<ms> ();	d_coef.modify<ms> ();
    		ss = config.ss;	wcs = config.phi;	wcr = config.wcr;
    		iM = config.iM;	jM = config.jM;	kM = config.kM;
    		dx = config.dx;	dy = config.dy;	dz = config.dz;	dt = config.dt;
    		alpha = config.vga;	n = config.vgn;	m = 1.0 - 1.0/config.vgn;
    	}
    	KOKKOS_INLINE_FUNCTION	void operator() (const int idx) const {
    		double qqx, qqy, qqz, wcp, sbar;
    		qqx = (q(idx,0) - q(iM(idx),0)) / dx;
    		qqy = (q(idx,1) - q(jM(idx),1)) / dy;
    		qqz = (q(idx,2) - q(kM(idx),2)) / dz;
    		#if TEST_TRACY
    		sbar = exp(0.1634 * h(idx,1));
    		#else
		    sbar = pow(1.0 + pow(fabs(alpha*h(idx,1)), n), -m);
		    #endif
		    wcp = sbar * (wcs - wcr) + wcr;
		    if (h(idx,1) > 0.0 | wcp > wcs)  {wcp = wcs;}
		    else if (wcp < wcr)    {wcp = wcr;}
    		resi(idx,7) = (wcp-wc(idx,0))/dt 
    			+ ss*wcp*(h(idx,1)-h(idx,0))/wcs/dt - qqx - qqy - qqz;
    		resi(idx,7) = -resi(idx,7);
    	}
    };
    
    template<class ExecutionSpace>
    struct Jacobian_side	{
    	typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value, dualDbl::memory_space, dualDbl::host_mirror_space>::type ms;
    	Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> ko;
    	Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> h;
    	Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> kr;
    	Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> jaco;
    	double ss, wcs, wcr, ks, d, alpha, n, m, coef;
    	int colp, colm, ax, ndom;
    	IntArr mapP, mapM;
    	Jacobian_side(dualDbl d_h, dualDbl d_kr, dualDbl d_k, dualDbl d_coef, int axis, Config config)	{
    		kr = d_kr.template view<ms> ();	d_kr.sync<ms> ();
    		ko = d_k.template view<ms> ();	d_k.sync<ms> ();
    		h = d_h.template view<ms> ();	d_h.sync<ms> ();
    		jaco = d_coef.template view<ms> ();	d_coef.sync<ms> ();	d_coef.modify<ms> ();
    		ss = config.ss;	wcs = config.phi;	wcr = config.wcr;
    		alpha = config.vga;	n = config.vgn;	m = 1.0 - 1.0/config.vgn;
    		coef = config.rho * GRAV / config.mu;	ndom = config.ndom;
    		if (axis == 2)	{
    			mapP = config.kP;	mapM = config.kM;
    			ks = config.kz;	d = config.dz;
    			colp = 5;	colm = 6;	ax = 2;
    		}
    		else if (axis == 1)	{
    			mapP = config.jP;	mapM = config.jM;
    			ks = config.ky;	d = config.dy;
    			colp = 3;	colm = 4;	ax = 1;
    		}
    		else if (axis == 0)	{
    			mapP = config.iP;	mapM = config.iM;
    			ks = config.kx;	d = config.dx;
    			colp = 1;	colm = 2;	ax = 0;
    		}
    	}
    	KOKKOS_INLINE_FUNCTION	void operator() (const int idx) const {
    		double kp, km, krp, krm, sbar, dh, dh_min = 1e-8;
    		// get dh
    		if (fabs(h(idx,1)) < 1.0)  {
		    	if (h(idx,1) > 0)   {dh = dh_min;}  else  {dh = -dh_min;}
		    }
            else  {dh = h(idx,1) * dh_min;}
            //dh = 1e-4;
    		// compute p face
    		// get conductivity 
    		#if TEST_TRACY
    		krp = exp(0.1634 * (h(mapP(idx),1)+dh));
    		#else
    		sbar = pow(1.0 + pow(fabs(alpha*(h(mapP(idx),1)+dh)), n), -m);
    		krp = pow(sbar,0.5) * pow(1-pow(1-pow(sbar,1.0/m),m), 2.0);
    		#endif
    		kp = 0.5 * coef * ks * (krp + kr(idx,0));
    		// get jacobian on p face
    		jaco(idx,colp) = -(kp*(h(mapP(idx),1) + dh - h(idx,1))/d/d)
    			+ (ko(idx,ax)*(h(mapP(idx),1) - h(idx,1))/d/d);
    		if (mapP(idx) > ndom)	{jaco(idx,colp) = jaco(idx,colp) * 2.0;}
    		if (ax == 2)	{jaco(idx,colp) += (kp - ko(idx,ax))/d;}
    		
    		// compute m face
    		// get conductivity 
    		#if TEST_TRACY
    		krm = exp(0.1634 * (h(mapM(idx),1)+dh));
    		#else
    		sbar = pow(1.0 + pow(fabs(alpha*(h(mapM(idx),1)+dh)), n), -m);
    		krm = pow(sbar,0.5) * pow(1-pow(1-pow(sbar,1.0/m),m), 2.0);
    		#endif
    		km = 0.5 * coef * ks * (krm + kr(idx,0));
    		// get jacobian on m face
    		jaco(idx,colm) = km*(h(idx,1) - h(mapM(idx),1) - dh)/d/d
    			- ko(mapM(idx),ax)*(h(idx,1) - h(mapM(idx),1))/d/d;
    		if (mapM(idx) > ndom)	{jaco(idx,colm) = jaco(idx,colm) * 2.0;}
    		if (ax == 2)	{jaco(idx,colm) += (ko(mapM(idx),ax) - km)/d;}
    		// divide by dh
    		jaco(idx,colp) = jaco(idx,colp) / dh;
    		jaco(idx,colm) = jaco(idx,colm) / dh;
    	}
    };
    
    template<class ExecutionSpace>
    struct Jacobian_cntr	{
    	typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value, dualDbl::memory_space, dualDbl::host_mirror_space>::type ms;
    	Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> wc;
    	Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> h;
    	Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> kr;
    	Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> jaco;
    	double ss, wcs, wcr, ks, d, alpha, n, m, coef;
    	double dx, dy, dz, dt, kx, ky, kz;
    	double bcval_xp, bcval_xm, bcval_yp, bcval_ym, bcval_zp, bcval_zm;
    	int nx, ny, nz, ndom, bctype_xp, bctype_xm, bctype_yp, bctype_ym, bctype_zp, bctype_zm;
    	IntArr iP, iM, jP, jM, kP, kM, i, j, k;
    	Jacobian_cntr(dualDbl d_h, dualDbl d_wc, dualDbl d_kr, dualDbl d_coef, Config config)	{
    		kr = d_kr.template view<ms> ();	d_kr.sync<ms> ();
    		wc = d_wc.template view<ms> ();	d_wc.sync<ms> ();
    		h = d_h.template view<ms> ();	d_h.sync<ms> ();
    		jaco = d_coef.template view<ms> ();	d_coef.sync<ms> ();	d_coef.modify<ms> ();
    		ss = config.ss;	wcs = config.phi;	wcr = config.wcr;
    		alpha = config.vga;	n = config.vgn;	m = 1.0 - 1.0/config.vgn;
    		coef = config.rho * GRAV / config.mu;	dt = config.dt;
    		iP = config.iP;	jP = config.jP;	kP = config.kP;
    		iM = config.iM;	jM = config.jM;	kM = config.kM;
    		i = config.i3d;	j = config.j3d;	k = config.k3d;
    		dx = config.dx;	dy = config.dy;	dz = config.dz;
    		kx = config.kx;	ky = config.ky;	kz = config.kz;
    		nx = config.nx;	ny = config.ny;	nz = config.nz;	ndom = config.ndom;
    		bctype_xp = config.bc_type_xp;	bcval_xp = config.bc_val_xp;
    		bctype_xm = config.bc_type_xm;	bcval_xm = config.bc_val_xm;
    		bctype_yp = config.bc_type_yp;	bcval_yp = config.bc_val_yp;
    		bctype_ym = config.bc_type_ym;	bcval_ym = config.bc_val_ym;
    		bctype_zp = config.bc_type_zp;	bcval_zp = config.bc_val_zp;
    		bctype_zm = config.bc_type_zm;	bcval_zm = config.bc_val_zm;
    	}
    	KOKKOS_INLINE_FUNCTION	void operator() (const int idx) const {
    		double sbar, wcp, rdh, krdh, hdh, dh, dh_min = 1e-8;
    		double qxp, qyp, qzp, qxm, qym, qzm;
    		double kxp, kxm, kyp, kym, kzp, kzm;
    		// get dh
    		if (fabs(h(idx,1)) < 1.0)  {
		    	if (h(idx,1) > 0)   {dh = dh_min;}  else  {dh = -dh_min;}
		    }
            else  {dh = h(idx,1) * dh_min;}
            //dh = 1e-4;
    		hdh = h(idx,1) + dh;
    		// compute new wc
    		#if TEST_TRACY
    		sbar = exp(0.1634 * hdh);
    		krdh = exp(0.1634 * hdh);
    		#else
    		sbar = pow(1.0 + pow(fabs(alpha*hdh), n), -m);
    		krdh = pow(sbar,0.5) * pow(1-pow(1-pow(sbar,1.0/m),m), 2.0);
    		#endif
    		wcp = sbar * (wcs - wcr) + wcr;
    		if (wcp > wcs)	{wcp = wcs;}
    		else if (wcp < wcr)	{wcp = wcr;}
    		if (krdh > 1.0)	{krdh = 1.0;}
    		else if (krdh < 0.0)	{krdh = 0.0;}
    		// storage term
    		rdh = (wcp-wc(idx,0))/dt + ss*wcp*(hdh-h(idx,0))/wcs/dt;
    		// face flux in x
    		kxp = 0.5 * coef * kx * (krdh + kr(iP(idx),0));
    		if (iP(idx) >= ndom)	{
    			if (bctype_xp == 2)	{qxp = bcval_xp / dz;}
    			else	{qxp = 2.0 * kxp * (h(iP(idx),1) - hdh) / dx / dx;}
    		}
    		else	{qxp = kxp * (h(iP(idx),1) - hdh) / dx / dx;}
    		kxm = 0.5 * coef * kx * (krdh + kr(iM(idx),0));
    		if (iM(idx) >= ndom)	{
    			if (bctype_xm == 2)	{qxm = bcval_xm / dz;}
    			else	{qxm = 2.0 * kxm * (hdh - h(iM(idx),1)) / dx / dx;}
    		}
    		else	{qxm = kxm * (hdh - h(iM(idx),1)) / dx / dx;}
    		// face flux in y
    		kyp = 0.5 * coef * ky * (krdh + kr(jP(idx),0));
    		if (jP(idx) >= ndom)	{
    			if (bctype_yp == 2)	{qyp = bcval_yp / dy;}
    			else	{qyp = 2.0 * kyp * (h(jP(idx),1) - hdh) / dy / dy;}
    		}
    		else	{qyp = kyp * (h(jP(idx),1) - hdh) / dy / dy;}
    		kym = 0.5 * coef * ky * (krdh + kr(jM(idx),0));
    		if (jM(idx) >= ndom)	{
    			if (bctype_ym == 2)	{qym = bcval_ym / dy;}
    			else	{qym = 2.0 * kym * (hdh - h(jM(idx),1)) / dy / dy;}
    		}
    		else	{qym = kym * (hdh - h(jM(idx),1)) / dy / dy;}
    		// face flux in z
    		kzp = 0.5 * coef * kz * (krdh + kr(kP(idx),0));
    		if (kP(idx) >= ndom)	{
    			//kzp = 0.0;
    			if (bctype_zp == 2)	{qzp = bcval_zp / dz;}
    			else	{qzp = (kzp * (h(kP(idx),1) - hdh) / (0.5*dz) - kzp) / dz;}
    		}
    		else	{qzp = (kzp * (h(kP(idx),1) - hdh) / dz - kzp) / dz;}
    		
    		kzm = 0.5 * coef * kz * (krdh + kr(kM(idx),0));
    		if (kM(idx) >= ndom)	{
    			if (bctype_zm == 2)	{qzm = bcval_zm / dz;}
    			else if (bctype_zm == 1 & hdh >= 0.0)	{
    				kzm = coef*kz;
    				qzm = (kzm * (hdh - h(kM(idx),1)) / (0.5*dz) - kzm) / dz;
    			}
    			else	{qzm = (kzm * (hdh - h(kM(idx),1)) / (0.5*dz) - kzm) / dz;}
    		}
    		else	{qzm = (kzm * (hdh - h(kM(idx),1)) / dz - kzm) / dz;}
    		// get residual
    		rdh -= (qxp - qxm + qyp - qym + qzp - qzm);
    		// get Jacobian
    		jaco(idx,0) = (rdh + jaco(idx,7)) / dh;
    		
    	}
    };
    
   
	/*
		------------------------------------------------------
    	------------------------------------------------------
		Update h = h + dh
		------------------------------------------------------
    	------------------------------------------------------
	*/
	inline void incre_h(Config config)	{
		Kokkos::parallel_for(config.ndom, Incre<dspace>(h, dh));
		config.sync(h);
	}
	
	template<class ExecutionSpace>
    struct Incre {
    	typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value, dualDbl::memory_space, dualDbl::host_mirror_space>::type ms;
		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> h;
		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> dh;
   		Incre(dualDbl d_h, dualDbl d_dh)	{
   			h = d_h.template view<ms> ();	d_h.sync<ms> ();	d_h.modify<ms> ();
   			dh = d_dh.template view<ms> ();	d_dh.sync<ms> ();	
   		}
    	KOKKOS_INLINE_FUNCTION	void operator() (const int idx) const {
        	h(idx,1) += dh(idx,0);
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
		Kokkos::parallel_reduce(config.ndom, Max_eps<dspace>(h, dh), eps);
		//Kokkos::parallel_reduce(config.ndom, Min_eps<dspace>(h, dh), eps_min);
		//return sqrt(eps) - (1e-5+1e-5*sqrt(eps_min));
		return sqrt(eps)/config.ndom;
	}
    template<class ExecutionSpace>
    struct Max_eps {
    	typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value, dualDbl::memory_space, dualDbl::host_mirror_space>::type ms;
   		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> h;
   		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> dh;
   		Max_eps(dualDbl d_h, dualDbl d_dh)	{
   			h = d_h.template view<ms> ();	d_h.sync<ms> ();
   			dh = d_dh.template view<ms> ();	d_dh.sync<ms> ();
   		}
   		KOKKOS_INLINE_FUNCTION void
		operator() (const int idx, double& rdh_sum) const	
		{
			rdh_sum += dh(idx,0) * dh(idx,0);
		}
		KOKKOS_INLINE_FUNCTION void
		join (volatile double& dst, const volatile double& src) const	
		{dst += src;}
    	KOKKOS_INLINE_FUNCTION void	init (double& dst) const	{dst = 0.0;}
    };
    
    template<class ExecutionSpace>
    struct Min_eps {
    	typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value, dualDbl::memory_space, dualDbl::host_mirror_space>::type ms;
   		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> h;
   		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> dh;
   		Min_eps(dualDbl d_h, dualDbl d_dh)	{
   			h = d_h.template view<ms> ();	d_h.sync<ms> ();
   			dh = d_dh.template view<ms> ();	d_dh.sync<ms> ();
   		}
   		KOKKOS_INLINE_FUNCTION void
		operator() (const int idx, double& rdh_max) const	
		{
			//double rdh = fabs(dh(idx,0) / h(idx,1));
			//double rdh = fabs(dh(idx,0));
			double rdh = h(idx,1) * h(idx,1);
			rdh_max = (rdh < rdh_max) ? rdh : rdh_max;
		}
		KOKKOS_INLINE_FUNCTION void
		join (volatile double& dst, const volatile double& src) const	
		{dst = (src < dst) ? src : dst;}
    	KOKKOS_INLINE_FUNCTION void	init (double& dst) const	{dst = 1e5;}
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
};
#endif
