/*
	Newton-Raphson scheme
*/
#ifndef _GWNEWTON_H_
#define _GWNEWTON_H_

#include "config.h"
#include "GwState.h"
#include "GwFunction.h"

class GwNewton : public GwFunction {

public:
	
	/*
		------------------------------------------------------
    	------------------------------------------------------
		Compute Residual for Newton scheme
		------------------------------------------------------
    	------------------------------------------------------
	*/
	inline void newton_residual(GwState &gw, Config &config)	{
		Kokkos::parallel_for(config.ndom, KOKKOS_LAMBDA(int idx) {
			double wcp, sbar, m = 1.0 - 1.0/config.vgn;
            double qqx = (gw.q(idx,0) - gw.q(config.iM(idx),0)) / config.dx;
    		double qqy = (gw.q(idx,1) - gw.q(config.jM(idx),1)) / config.dy;
    		double qqz = (gw.q(idx,2) - gw.q(config.kM(idx),2)) / config.dz;
    		#if TEST_TRACY
    		sbar = exp(0.1634 * gw.h(idx,1));
    		#else
		    sbar = pow(1.0 + pow(fabs(config.vga*gw.h(idx,1)), config.vgn), -m);
		    #endif
		    wcp = sbar * (config.phi - config.wcr) + config.wcr;
		    if (gw.h(idx,1) > 0.0 | wcp > config.phi)  {wcp = config.phi;}
		    else if (wcp < config.wcr)    {wcp = config.wcr;}
    		gw.coef(idx,7) = (wcp-gw.wc(idx,0))/config.dt 
    			+ config.ss*wcp*(gw.h(idx,1)-gw.h(idx,0))/config.phi/config.dt - qqx - qqy - qqz;
    		gw.coef(idx,7) = -gw.coef(idx,7);
        });
	}
	
	/*
		------------------------------------------------------
    	------------------------------------------------------
		Compute off-diagonal elements of the Jacobian
		------------------------------------------------------
    	------------------------------------------------------
	*/
	inline void jacobian_offdiag(GwState &gw, Config &config, int axis)	{
		IntArr mapP, mapM;
		double ks, d;
		int colp, colm, ax;
		switch (axis)	{
			case 0:
				mapP = config.iP;	mapM = config.iM;
    			ks = config.kx;	d = config.dx;
    			colp = 1;	colm = 2;	ax = 0;
			case 1:
				mapP = config.jP;	mapM = config.jM;
    			ks = config.ky;	d = config.dy;
    			colp = 3;	colm = 4;	ax = 1;
			case 2:
				mapP = config.kP;	mapM = config.kM;
				ks = config.kz;	d = config.dz;
				colp = 5;	colm = 6;	ax = 2;
		}
		Kokkos::parallel_for(config.ndom, KOKKOS_LAMBDA(int idx) {
			double kp, km, krp, krm, sbar, dh, dh_min = 1e-8;
			double m = 1.0 - 1.0/config.vgn;
			double coef = config.rho * GRAV / config.mu;
			// get dh
    		if (fabs(gw.h(idx,1)) < 1.0)  {
		    	if (gw.h(idx,1) > 0)   {dh = dh_min;}  else  {dh = -dh_min;}
		    }
            else  {dh = gw.h(idx,1) * dh_min;}
    		// compute p face
    		// get conductivity 
    		#if TEST_TRACY
    		krp = exp(0.1634 * (gw.h(mapP(idx),1)+dh));
    		#else
    		sbar = pow(1.0 + pow(fabs(config.vga*(gw.h(mapP(idx),1)+dh)), config.vgn), -m);
    		krp = pow(sbar,0.5) * pow(1-pow(1-pow(sbar,1.0/m),m), 2.0);
    		#endif
    		kp = 0.5 * coef * ks * (krp + gw.k(idx,3));
    		// get jacobian on p face
    		gw.coef(idx,colp) = -(kp*(gw.h(mapP(idx),1) + dh - gw.h(idx,1))/d/d)
    			+ (gw.k(idx,ax)*(gw.h(mapP(idx),1) - gw.h(idx,1))/d/d);
    		if (mapP(idx) > config.ndom)	{gw.coef(idx,colp) = gw.coef(idx,colp) * 2.0;}
    		if (ax == 2)	{gw.coef(idx,colp) += (kp - gw.k(idx,ax))/d;}
    		
    		// compute m face
    		// get conductivity 
    		#if TEST_TRACY
    		krm = exp(0.1634 * (gw.h(mapM(idx),1)+dh));
    		#else
    		sbar = pow(1.0 + pow(fabs(config.vga*(gw.h(mapM(idx),1)+dh)), config.vgn), -m);
    		krm = pow(sbar,0.5) * pow(1-pow(1-pow(sbar,1.0/m),m), 2.0);
    		#endif
    		km = 0.5 * coef * ks * (krm + gw.k(idx,3));
    		// get jacobian on m face
    		gw.coef(idx,colm) = km*(gw.h(idx,1) - gw.h(mapM(idx),1) - dh)/d/d
    			- gw.k(mapM(idx),ax)*(gw.h(idx,1) - gw.h(mapM(idx),1))/d/d;
    		if (mapM(idx) > config.ndom)	{gw.coef(idx,colm) = gw.coef(idx,colm) * 2.0;}
    		if (ax == 2)	{gw.coef(idx,colm) += (gw.k(mapM(idx),ax) - km)/d;}
    		// divide by dh
    		gw.coef(idx,colp) = gw.coef(idx,colp) / dh;
    		gw.coef(idx,colm) = gw.coef(idx,colm) / dh;
		});
		
	}
	
    
    /*
		------------------------------------------------------
    	------------------------------------------------------
		Compute diagonal elements of the Jacobian
		------------------------------------------------------
    	------------------------------------------------------
	*/
	inline void jacobian_diag(GwState &gw, GwMatrix &A, Config &config)	{
		Kokkos::parallel_for(config.ndom, KOKKOS_LAMBDA(int idx) {
			double sbar, wcp, rdh, krdh, hdh, dh, dh_min = 1e-8;
    		double qxp, qyp, qzp, qxm, qym, qzm;
    		double kxp, kxm, kyp, kym, kzp, kzm;
    		double m = 1.0 - 1.0/config.vgn;
			double coef = config.rho * GRAV / config.mu;
    		// get dh
    		if (fabs(gw.h(idx,1)) < 1.0)  {
		    	if (gw.h(idx,1) > 0)   {dh = dh_min;}  else  {dh = -dh_min;}
		    }
            else  {dh = gw.h(idx,1) * dh_min;}
    		hdh = gw.h(idx,1) + dh;
    		// compute new wc
    		#if TEST_TRACY
    		sbar = exp(0.1634 * hdh);
    		krdh = exp(0.1634 * hdh);
    		#else
    		sbar = pow(1.0 + pow(fabs(config.vga*hdh), config.vgn), -m);
    		krdh = pow(sbar,0.5) * pow(1-pow(1-pow(sbar,1.0/m),m), 2.0);
    		#endif
    		wcp = sbar * (config.phi - config.wcr) + config.wcr;
    		if (wcp > config.phi)	{wcp = config.phi;}
    		else if (wcp < config.wcr)	{wcp = config.wcr;}
    		if (krdh > 1.0)	{krdh = 1.0;}
    		else if (krdh < 0.0)	{krdh = 0.0;}
    		// storage term
    		rdh = (wcp-gw.wc(idx,0))/config.dt + config.ss*wcp*(hdh-gw.h(idx,0))/config.phi/config.dt;
    		// face flux in x
    		kxp = 0.5 * coef * config.kx * (krdh + gw.k(config.iP(idx),3));
    		if (config.iP(idx) >= config.ndom)	{
    			if (config.bc_type_xp == 2)	{qxp = config.bc_val_xp / config.dz;}
    			else	{qxp = 2.0 * kxp * (gw.h(config.iP(idx),1) - hdh) / config.dx / config.dx;}
    		}
    		else	{qxp = kxp * (gw.h(config.iP(idx),1) - hdh) / config.dx / config.dx;}
    		kxm = 0.5 * coef * config.kx * (krdh + gw.k(config.iM(idx),3));
    		if (config.iM(idx) >= config.ndom)	{
    			if (config.bc_type_xm == 2)	{qxm = config.bc_val_xm / config.dz;}
    			else	{qxm = 2.0 * kxm * (hdh - gw.h(config.iM(idx),1)) / config.dx / config.dx;}
    		}
    		else	{qxm = kxm * (hdh - gw.h(config.iM(idx),1)) / config.dx / config.dx;}
    		// face flux in y
    		kyp = 0.5 * coef * config.ky * (krdh + gw.k(config.jP(idx),3));
    		if (config.jP(idx) >= config.ndom)	{
    			if (config.bc_type_yp == 2)	{qyp = config.bc_val_yp / config.dy;}
    			else	{qyp = 2.0 * kyp * (gw.h(config.jP(idx),1) - hdh) / config.dy / config.dy;}
    		}
    		else	{qyp = kyp * (gw.h(config.jP(idx),1) - hdh) / config.dy / config.dy;}
    		kym = 0.5 * coef * config.ky * (krdh + gw.k(config.jM(idx),3));
    		if (config.jM(idx) >= config.ndom)	{
    			if (config.bc_type_ym == 2)	{qym = config.bc_val_ym / config.dy;}
    			else	{qym = 2.0 * kym * (hdh - gw.h(config.jM(idx),1)) / config.dy / config.dy;}
    		}
    		else	{qym = kym * (hdh - gw.h(config.jM(idx),1)) / config.dy / config.dy;}
    		// face flux in z
    		kzp = 0.5 * coef * config.kz * (krdh + gw.k(config.kP(idx),3));
    		if (config.kP(idx) >= config.ndom)	{
    			if (config.bc_type_zp == 2)	{qzp = config.bc_val_zp / config.dz;}
    			else	{qzp = (kzp * (gw.h(config.kP(idx),1) - hdh) / (0.5*config.dz) - kzp) / config.dz;}
    		}
    		else	{qzp = (kzp * (gw.h(config.kP(idx),1) - hdh) / config.dz - kzp) / config.dz;}
    		
    		kzm = 0.5 * coef * config.kz * (krdh + gw.k(config.kM(idx),3));
    		if (config.kM(idx) >= config.ndom)	{
    			if (config.bc_type_zm == 2)	{qzm = config.bc_val_zm / config.dz;}
    			else if (config.bc_type_zm == 1 & gw.h(config.kM(idx),1) >= 0.0)	{
    			//else if (bctype_zm == 1)	{
    				//kzm = coef*kz;
    				kzm = coef * config.kz * gw.k(config.kM(idx),3);
    				qzm = (kzm * (hdh - gw.h(config.kM(idx),1)) / (0.5*config.dz) - kzm) / config.dz;
    			}
    			else	{qzm = (kzm * (hdh - gw.h(config.kM(idx),1)) / (0.5*config.dz) - kzm) / config.dz;}
    		}
    		else	{qzm = (kzm * (hdh - gw.h(config.kM(idx),1)) / config.dz - kzm) / config.dz;}
    		// get residual
    		rdh -= (qxp - qxm + qyp - qym + qzp - qzm);
    		// get Jacobian
    		gw.coef(idx,0) = (rdh + gw.coef(idx,7)) / dh;
		});
		
		// Insert coefficients into Matrix A
        Kokkos::parallel_for( config.ndom , KOKKOS_LAMBDA(int idx) {
        	int irow = A.ptr(idx);
        	if (config.j3d(idx) > 0)	{A.ind(irow) = idx - config.nx*config.nz;	A.val(irow) = gw.coef(idx,4);  irow++;}
        	if (config.i3d(idx) > 0)	{A.ind(irow) = idx - config.nz;	A.val(irow) = gw.coef(idx,2);  irow++;}
        	if (config.k3d(idx) > 0)	{A.ind(irow) = idx - 1;		A.val(irow) = gw.coef(idx,6);  irow++;}
        	A.ind(irow) = idx;	A.val(irow) = gw.coef(idx,0);	irow++;
        	if (config.k3d(idx) < config.nz-1)	{A.ind(irow) = idx + 1;		A.val(irow) = gw.coef(idx,5);  irow++;}
        	if (config.i3d(idx) < config.nx-1)	{A.ind(irow) = idx + config.nz;	A.val(irow) = gw.coef(idx,1);  irow++;}
        	if (config.j3d(idx) < config.ny-1)	{A.ind(irow) = idx + config.nx*config.nz;	A.val(irow) = gw.coef(idx,3);  irow++;}
        	A.rhs(idx) = gw.coef(idx,7);
        });
	
	}
    
   
	/*
		------------------------------------------------------
    	------------------------------------------------------
		Update h = h + dh
		------------------------------------------------------
    	------------------------------------------------------
	*/
	inline void incre_h(GwState &gw, Config &config)	{
		Kokkos::parallel_for(config.ndom, KOKKOS_LAMBDA(int idx) {
			gw.h(idx,1) += gw.dh(idx);
		});
	}
    
    /*
		------------------------------------------------------
    	------------------------------------------------------
		Get maximum relative residual
		------------------------------------------------------
    	------------------------------------------------------
	*/
	inline double newton_eps(GwState &gw, Config &config)	{
		double dh2_sum;
		Kokkos::parallel_reduce(config.ndom, KOKKOS_LAMBDA (int idx, double &tmp) {
			tmp += gw.dh(idx) * gw.dh(idx);	
		} , Kokkos::Sum<double>(dh2_sum) );
		return dh2_sum;
	}
    
    
};
#endif
