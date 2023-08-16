#ifndef _GW_FUNCTION_H_
#define _GW_FUNCTION_H_

#include "config.h"
#include "GwMatrix.h"
#include "GwState.h"
#include <set>
#include <math.h>


class GwFunction   {

public:

    inline void update(GwState &gw, Config &config)	{
    	Kokkos::parallel_for(config.nall, KOKKOS_LAMBDA(int idx) {
            gw.h(idx,0) = gw.h(idx,1);  gw.wc(idx,0) = gw.wc(idx,1);
        });
    }

    inline double getErr(GwState &gw, Config &config)	{
    	double dh_max;
    	Kokkos::parallel_reduce(config.ndom, KOKKOS_LAMBDA (int idx, double &tmp) {
            double dh = fabs(gw.h(idx,1) - gw.h(idx,0));
			tmp = (dh > tmp) ? dh : tmp;
		} , Kokkos::Max<double>(dh_max) );
		return dh_max;
    }



    /* --------------------------------------------------
        Get face conductivity
    -------------------------------------------------- */
    inline void face_conductivity(GwState &gw, Config &config)	{

        // Get relatively permeability at cell centers
        Kokkos::parallel_for( config.nall , KOKKOS_LAMBDA(int idx) {
        	double m = 1.0 - 1.0/config.vgn;
        	double sbar = pow(1.0 + pow(fabs(config.vga*gw.h(idx,1)), config.vgn), -m);
        	#if TEST_TRACY
        	gw.k(idx,3) = exp(0.1634 * gw.h(idx,1));
        	#else
        	gw.k(idx,3) = pow(sbar,0.5) * pow(1-pow(1-pow(sbar,1.0/m),m), 2.0);
        	#endif
        	if (gw.k(idx,3) < 0.0)	{gw.k(idx,3) = 0.0;}
        	else if (gw.k(idx,3) > 1.0)	{gw.k(idx,3) = 1.0;}
    	});
        // Get K on interior cell faces
        Kokkos::parallel_for( config.ndom , KOKKOS_LAMBDA(int ii) {
        	double coef = config.rho * GRAV / config.mu;
        	gw.k(ii,0) = 0.5 * coef * config.kx * (gw.k(ii,3) + gw.k(config.iP(ii),3));
        	gw.k(ii,1) = 0.5 * coef * config.ky * (gw.k(ii,3) + gw.k(config.jP(ii),3));
        	gw.k(ii,2) = 0.5 * coef * config.kz * (gw.k(ii,3) + gw.k(config.kP(ii),3));
        });
        // Get K on boundaries
        Kokkos::parallel_for( config.nbcell , KOKKOS_LAMBDA(int ii) {
        	int idx = config.bcell(ii,1), idm, idp;
        	double coef = config.rho * GRAV / config.mu;
        	double kr1 = gw.k(idx,3), kr2;
        	// x
			if (config.kx != 0.0)	{
				if (config.bcell(ii,3) == -1)	{
					idm = config.bcell(ii,2);
					if (config.bcell(ii,0) == 0)	{gw.k(idm,0) = 0.0;}
					else	{
						kr2 = gw.k(idm,3);
						gw.k(idm,0) = 0.5 * coef * config.kx * (kr1 + kr2);
					}
				}
				else if (config.bcell(ii,3) == 1)	{
					idp = config.bcell(ii,2);
					if (config.bcell(ii,0) == 0)	{gw.k(idx,0) = 0.0;}
					else	{
						kr2 = gw.k(idp,3);
						gw.k(idx,0) = 0.5 * coef * config.kx * (kr1 + kr2);
					}
				}
			}

			//y
			if (config.ky != 0.0)	{
				if (config.bcell(ii,3) == -2)	{
					idm = config.bcell(ii,2);
					if (config.bcell(ii,0) == 0)	{gw.k(idm,1) = 0.0;}
					else	{
						kr2 = gw.k(idm,3);
						gw.k(idm,1) = 0.5 * coef * config.ky * (kr1 + kr2);
					}
				}
				else if (config.bcell(ii,3) == 2)	{
					idp = config.bcell(ii,2);
					if (config.bcell(ii,0) == 0)	{gw.k(idx,1) = 0.0;}
					else	{
						kr2 = gw.k(idp,3);
						gw.k(idx,1) = 0.5 * coef * config.ky * (kr1 + kr2);
					}
				}
			}
			//z
			if (config.bcell(ii,3) == -3)	{
				idm = config.bcell(ii,2);
				if (config.bcell(ii,0) == 0)	{gw.k(idm,2) = 0.0;}
				else	{
					kr2 = gw.k(idm,3);
					gw.k(idm,2) = 0.5 * coef * config.kz * (kr1 + kr2);
				}
				// inundated top boundary
				if (config.bcell(ii,0) == 1 & config.bc_val_zm >= 0.0)	{
					gw.k(idm,2) = config.rho * GRAV * config.kz / config.mu;
				}
			}
			else if (config.bcell(ii,3) == 3)	{
				idp = config.bcell(ii,2);
				if (config.bcell(ii,0) == 0)	{gw.k(idx,2) = 0.0;}
				else	{
					kr2 = gw.k(idp,3);
					gw.k(idx,2) = 0.5 * coef * config.kz * (kr1 + kr2);
				}
			}
        });

	}

	// An alternative implementation with fewer synchronizations
	inline void face_conductivity2(GwState &gw, Config &config)	{


        Kokkos::parallel_for( config.ndom , KOKKOS_LAMBDA(int idx) {
        	double m = 1.0 - 1.0/config.vgn;
        	double coef = config.rho * GRAV / config.mu;
        	double sbar2, sbar = pow(1.0 + pow(fabs(config.vga*gw.h(idx,1)), config.vgn), -m);
        	gw.k(idx,3) = pow(sbar,0.5) * pow(1-pow(1-pow(sbar,1.0/m),m), 2.0);
        	if (gw.k(idx,3) < 0.0)	{gw.k(idx,3) = 0.0;}
        	else if (gw.k(idx,3) > 1.0)	{gw.k(idx,3) = 1.0;}
        	// x
        	sbar2 = pow(1.0 + pow(fabs(config.vga*gw.h(config.iP(idx),1)), config.vgn), -m);
        	gw.k(config.iP(idx),3) = pow(sbar2,0.5) * pow(1-pow(1-pow(sbar2,1.0/m),m), 2.0);
        	gw.k(idx,0) = 0.5 * coef * config.kx * (gw.k(idx,3) + gw.k(config.iP(idx),3));
        	if (gw.k(idx,0) > coef*config.kx)	{gw.k(idx,0) = coef*config.kx;}
        	// y
        	sbar2 = pow(1.0 + pow(fabs(config.vga*gw.h(config.jP(idx),1)), config.vgn), -m);
        	gw.k(config.jP(idx),3) = pow(sbar2,0.5) * pow(1-pow(1-pow(sbar2,1.0/m),m), 2.0);
        	gw.k(idx,1) = 0.5 * coef * config.ky * (gw.k(idx,3) + gw.k(config.jP(idx),3));
        	if (gw.k(idx,1) > coef*config.ky)	{gw.k(idx,1) = coef*config.ky;}
        	// z
        	sbar2 = pow(1.0 + pow(fabs(config.vga*gw.h(config.kP(idx),1)), config.vgn), -m);
        	gw.k(config.kP(idx),3) = pow(sbar2,0.5) * pow(1-pow(1-pow(sbar2,1.0/m),m), 2.0);
        	gw.k(idx,2) = 0.5 * coef * config.kz * (gw.k(idx,3) + gw.k(config.kP(idx),3));
        	if (gw.k(idx,2) > coef*config.kz)	{gw.k(idx,2) = coef*config.kz;}
        	// Boundaries
        	if (config.i3d(idx) == 0)	{
    			sbar2 = pow(1.0 + pow(fabs(config.vga*gw.h(config.iM(idx),1)), config.vgn), -m);
    			gw.k(config.iM(idx),3) = pow(sbar2,0.5) * pow(1-pow(1-pow(sbar2,1.0/m),m), 2.0);
    			gw.k(config.iM(idx),0) = 0.5 * coef * config.kx * (gw.k(idx,3) + gw.k(config.iM(idx),3));
        	}
        	if (config.j3d(idx) == 0)	{
    			sbar2 = pow(1.0 + pow(fabs(config.vga*gw.h(config.jM(idx),1)), config.vgn), -m);
    			gw.k(config.jM(idx),3) = pow(sbar2,0.5) * pow(1-pow(1-pow(sbar2,1.0/m),m), 2.0);
    			gw.k(config.jM(idx),1) = 0.5 * coef * config.ky * (gw.k(idx,3) + gw.k(config.jM(idx),3));
        	}
        	if (config.k3d(idx) == 0)	{
				sbar2 = pow(1.0 + pow(fabs(config.vga*gw.h(config.kM(idx),1)), config.vgn), -m);
				gw.k(config.kM(idx),3) = pow(sbar2,0.5) * pow(1-pow(1-pow(sbar2,1.0/m),m), 2.0);
				gw.k(config.kM(idx),2) = 0.5 * coef * config.kz * (gw.k(idx,3) + gw.k(config.kM(idx),3));
        	}
    	});

	}
    // /* --------------------------------------------------
    //     End of conductivity block
    // -------------------------------------------------- */





    // /* --------------------------------------------------
    //     Get face flux
    // -------------------------------------------------- */
    inline void face_flux(GwState &gw, Config &config)	{
        Kokkos::parallel_for( config.ndom , KOKKOS_LAMBDA(int idx) {
            gw.q(idx,0) = gw.k(idx,0) * (gw.h(config.iP(idx),1) - gw.h(idx,1)) / config.dx;
    		gw.q(idx,1) = gw.k(idx,1) * (gw.h(config.jP(idx),1) - gw.h(idx,1)) / config.dy;
    		gw.q(idx,2) = gw.k(idx,2) * (gw.h(config.kP(idx),1) - gw.h(idx,1)) / config.dz - gw.k(idx,2);
    	});
    	Kokkos::parallel_for( config.nbcell , KOKKOS_LAMBDA(int ii) {
            int idx, idex;
            double coef = config.rho * GRAV / config.mu;
    		idx = config.bcell(ii,1);
			idex = config.bcell(ii,2);
			switch (config.bcell(ii,3))	{
				case -1:
					if (config.bcell(ii,0) == 2)	{gw.q(idex,0) = config.bcval(ii);}
					else	{
						gw.q(idex,0) = 2.0*gw.k(idex,0)*(gw.h(idx,1) - gw.h(idex,1)) / config.dx;
					}
					break;
				case 1:
					if (config.bcell(ii,0) == 2)	{gw.q(idx,0) = config.bcval(ii);}
					else {gw.q(idx,0) = gw.q(idx,0)*2.0;}	break;
				case -2:
					if (config.bcell(ii,0) == 2)	{gw.q(idex,1) = config.bcval(ii);}
					else	{gw.q(idex,1) = 2.0*gw.k(idex,1)*(gw.h(idx,1) - gw.h(idex,1)) / config.dy;}	break;
				case 2:
					if (config.bcell(ii,0) == 2)	{gw.q(idx,1) = config.bcval(ii);}
					else {gw.q(idx,1) = gw.q(idx,1)*2.0;}	break;
				case -3:
					if (config.bcell(ii,0) == 2)	{gw.q(idex,2) = config.bcval(ii);}
					else if (config.bcell(ii,0) == 1 & config.bcval(ii) >= 0.0)	{
						gw.q(idex,2) = coef*config.kz*(gw.h(idx,1) - gw.h(idex,1)) / (config.dz/2.0) - coef*config.kz;
					}
					else	{
						gw.q(idex,2) = gw.k(idex,2)*(gw.h(idx,1) - gw.h(idex,1)) / (config.dz/2.0) - gw.k(idex,2);
						//printf(" -%d- : k = %f, h = %f, %f ---> q = %f\n",idx,1e8*gw.k(idex,2),gw.h(idx,1),gw.h(idex,1),1e8*gw.q(idex,2));
					}
					break;
				case 3:
					if (config.bcell(ii,0) == 2)	{gw.q(idx,2) = config.bcval(ii);}
					else {gw.q(idx,2) = gw.k(idx,2)*(gw.h(idex,1) - gw.h(idx,1)) / (config.dz/2.0) - gw.k(idx,2);}	break;
			}
        });
	}
    // /* --------------------------------------------------
    //     End of flux block
    // -------------------------------------------------- */




    // /* --------------------------------------------------
    //     Get matrix coefficients
    // -------------------------------------------------- */
    inline void linear_system(GwState &gw, GwMatrix &A, Config &config)	{
    	// Matrix coefficients
    	Kokkos::parallel_for( config.ndom , KOKKOS_LAMBDA(int idx) {
    		double nume, deno, m, ch = 0.0;
    		m = 1.0 - 1.0/config.vgn;
    		if (gw.h(idx,1) < 0.0)	{
				#if TEST_TRACY
				ch = 0.1634 * (config.phi - config.wcr) * exp(0.1634*gw.h(idx,1));
				#else
    			nume = config.vga*config.vgn*m*(config.phi - config.wcr)*pow(fabs(config.vga*gw.h(idx,1)),config.vgn-1.0);
				deno = pow((1.0 + pow(fabs(config.vga*gw.h(idx,1)),config.vgn)), m+1);
				ch = nume / deno;
				#endif
			}
    		gw.coef(idx,0) = ch + config.ss*gw.wc(idx,1)/config.phi;
    		gw.coef(idx,1) = - config.dt * gw.k(idx,0) / pow(config.dx, 2.0);
    		gw.coef(idx,2) = - config.dt * gw.k(config.iM(idx),0) / pow(config.dx, 2.0);
    		gw.coef(idx,3) = - config.dt * gw.k(idx,1) / pow(config.dy, 2.0);
    		gw.coef(idx,4) = - config.dt * gw.k(config.jM(idx),1) / pow(config.dy, 2.0);
    		gw.coef(idx,5) = - config.dt * gw.k(idx,2) / pow(config.dz, 2.0);
    		gw.coef(idx,6) = - config.dt * gw.k(config.kM(idx),2) / pow(config.dz, 2.0);
    		gw.coef(idx,7) = (ch + config.ss*gw.wc(idx,1)/config.phi)*gw.h(idx,1) - config.dt*(gw.k(idx,2) - gw.k(config.kM(idx),2)) / config.dz;
    		if (config.scheme == 3)	{gw.coef(idx,7) -= (gw.wc(idx,1) - gw.wc(idx,2));}
    	});
    	// Boundary treatment
    	Kokkos::parallel_for( config.nbcell , KOKKOS_LAMBDA(int ii) {
    		int icol, idx;
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
			// Flux top BC
			if (config.bcell(ii,3) == -3 && config.bc_type_zm == 2)	{
				gw.coef(idx,7) += config.dt*(-config.bc_val_zm - gw.k(config.kM(idx),2))/config.dz;
			}
			else {
				gw.coef(idx,icol) = gw.coef(idx,icol) * 2.0;
				gw.coef(idx,7) += gw.coef(idx,icol) * gw.h(config.bcell(ii,2),1);
			}
		});
    	// Diagonal elements
    	//printf(" DT = %f\n",config.dt);
    	Kokkos::parallel_for( config.ndom , KOKKOS_LAMBDA(int ii) {
    		gw.coef(ii,0) -= (gw.coef(ii,1)+gw.coef(ii,2)+gw.coef(ii,3)+gw.coef(ii,4)+gw.coef(ii,5)+gw.coef(ii,6));
    		/*if (ii < 300 & ii >= 200)	{
    		printf(" (%f,%f,%f,%f,%f,%f,%f) - %f\n",1e8*gw.coef(ii,4),1e8*gw.coef(ii,2),1e8*gw.coef(ii,6),1e8*gw.coef(ii,0),1e8*gw.coef(ii,5),1e8*gw.coef(ii,1),1e8*gw.coef(ii,3),1e8*gw.coef(ii,7));}*/
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
    // /* --------------------------------------------------
    //     End matrix coefficients
    // -------------------------------------------------- */





    // /* --------------------------------------------------
    //     Get water content
    // -------------------------------------------------- */
    inline void update_wc(GwState &gw, Config &config)	{

    	// Explicit update of water content
    	Kokkos::parallel_for( config.ndom , KOKKOS_LAMBDA(int idx) {
    		double coef, qqx, qqy, qqz;
    		coef = 1.0 + config.ss*(gw.h(idx,1) - gw.h(idx,0)) / config.phi;
    		qqx = (gw.q(idx,0) - gw.q(config.iM(idx),0)) / config.dx;
    		qqy = (gw.q(idx,1) - gw.q(config.jM(idx),1)) / config.dy;
    		qqz = (gw.q(idx,2) - gw.q(config.kM(idx),2)) / config.dz;
    		gw.wc(idx,1) = (gw.wc(idx,0) + config.dt * (qqx + qqy + qqz)) / coef;
    		if (gw.wc(idx,1) < config.wcr+0.001)	{gw.wc(idx,1) = config.wcr+0.001;}
		});
		if (config.scheme == 1)	{
			Kokkos::parallel_for( config.ndom , KOKKOS_LAMBDA(int idx) {
				double m = 1.0 - 1.0/config.vgn;
				if (gw.wc(idx,1) > config.phi)	{gw.wc(idx,1) = config.phi;}
				else if (gw.wc(idx,1) < config.wcr+0.001)	{gw.wc(idx,1) = config.wcr+0.001;}
				#if TEST_TRACY
				gw.h(idx,1) = log((gw.wc(idx,1) - config.wcr)/(config.phi - config.wcr)) / 0.1634;
				#else
				gw.h(idx,1) = -(1.0/config.vga) * (pow(pow((config.phi-config.wcr)/(gw.wc(idx,1)-config.wcr),(1/m)) - 1.0, 1/config.vgn));
				#endif
			});
		}
		// PC Scheme : Choose between h and wc
		else if (config.scheme == 2)	{
			Kokkos::parallel_for( config.ndom , KOKKOS_LAMBDA(int idx) {
				double sbar, m = 1.0 - 1.0/config.vgn;
				if (gw.wc(idx,1) >= config.phi - 1e-7)	{
					gw.wc(idx,1) = config.phi;
				}
				else if (gw.wc(config.kP(idx),1) >= config.phi | gw.wc(config.kM(idx),1) >= config.phi)	{
					#if TEST_TRACY
					sbar = exp(0.1634*gw.h(idx,1));
					#else
					sbar = pow(1.0 + pow(fabs(config.vga*gw.h(idx,1)), config.vgn), -m);
					#endif
					gw.wc(idx,1) = sbar * (config.phi - config.wcr) + config.wcr;
				}
				else if (gw.wc(config.iP(idx),1) >= config.phi | gw.wc(config.iM(idx),1) >= config.phi)	{
					#if TEST_TRACY
					sbar = exp(0.1634*gw.h(idx,1));
					#else
					sbar = pow(1.0 + pow(fabs(config.vga*gw.h(idx,1)), config.vgn), -m);
					#endif
					gw.wc(idx,1) = sbar * (config.phi - config.wcr) + config.wcr;
				}
				else if (gw.wc(config.jP(idx),1) >= config.phi | gw.wc(config.jM(idx),1) >= config.phi)	{
					#if TEST_TRACY
					sbar = exp(0.1634*gw.h(idx,1));
					#else
					sbar = pow(1.0 + pow(fabs(config.vga*gw.h(idx,1)), config.vgn), -m);
					#endif
					gw.wc(idx,1) = sbar * (config.phi - config.wcr) + config.wcr;
				}
				else {
					#if TEST_TRACY
					gw.h(idx,1) = log((gw.wc(idx,1) - config.wcr)/(config.phi - config.wcr)) / 0.1634;
					#else
					gw.h(idx,1) = -(1.0/config.vga) * (pow(pow((config.phi-config.wcr)/(gw.wc(idx,1)-config.wcr),(1/m)) - 1.0, 1/config.vgn));
					#endif
				}
				if (gw.wc(idx,1) > config.phi)	{gw.wc(idx,1) = config.phi;}
				else if (gw.wc(idx,1) < config.wcr+0.001)	{gw.wc(idx,1) = config.wcr+0.001;}
			});
		}
		// Picard & Newton schemes: wc = f(h)
        else	{
            Kokkos::parallel_for( config.ndom , KOKKOS_LAMBDA(int idx) {
                double sbar, m = 1.0 - 1.0/config.vgn;
				#if TEST_TRACY
				sbar = exp(0.1634 * gw.h(idx,1));
				#else
				sbar = pow(1.0 + pow(fabs(config.vga*gw.h(idx,1)), config.vgn), -m);
				#endif
				gw.wc(idx,1) = sbar * (config.phi - config.wcr) + config.wcr;
				if (gw.h(idx,1) > 0.0 | gw.wc(idx,1) > config.phi)	{gw.wc(idx,1) = config.phi;}
				else if (gw.wc(idx,1) < config.wcr)	{gw.wc(idx,1) = config.wcr;}
            });
		}
    }
    // /* --------------------------------------------------
    //     End water content block
    // -------------------------------------------------- */





    // /* --------------------------------------------------
    //     Update dt based on either water content or iteration
    // -------------------------------------------------- */
    inline void dt_waco(GwState &gw, Config &config)	{
    	double dwc_max, dt_old;
    	dt_old = config.dt;
    	Kokkos::parallel_reduce(config.ndom, KOKKOS_LAMBDA (int idx, double &tmp) {
            double dwc = fabs(gw.wc(idx,1) - gw.wc(idx,0));
			tmp = (dwc > tmp) ? dwc : tmp;
		} , Kokkos::Max<double>(dwc_max) );
    	// if (dwc_max > 0.02)	{config.dt = config.dt * 0.5;}
    	// else if (dwc_max > 0.0 & dwc_max < 0.01)	{config.dt = config.dt * 2.0;}

        config.dt = config.dt * 1.1;

    	if (config.dt > config.dt_max)	{config.dt = config.dt_max;}
    	else if (config.dt < config.dt_init)	{config.dt = config.dt_init;}
    	//if (config.dt != dt_old)	{
    	//	printf("     >> max dwc = %f, dt is changed to %f sec\n",dwc_max,config.dt);
    	//}
    }

    inline void dt_iter(GwState &gw, int iter, Config &config)	{
    	double dt_old;
    	dt_old = config.dt;
        // if (iter < 10)   {config.dt = config.dt * 2.0;}
        // else if (iter > 15)  {config.dt = config.dt * 0.5;}

        config.dt = config.dt * 1.1;

        if (config.dt > config.dt_max)	{config.dt = config.dt_max;}
    	else if (config.dt < config.dt_init)	{config.dt = config.dt_init;}
    	//if (config.dt != dt_old)	{
    	//	printf("     >> iter = %f, dt is changed to %f sec\n",iter,config.dt);
    	//}
    }


    // /* --------------------------------------------------
    //     End dt update
    // -------------------------------------------------- */



    // /* --------------------------------------------------
    //     Get the maximum difference between iterations
    // -------------------------------------------------- */
    inline double get_eps(GwState &gw, Config &config)	{
    	double eps;
        Kokkos::parallel_reduce(config.ndom, KOKKOS_LAMBDA (int idx, double &tmp) {
            double dwc = fabs(gw.h(idx,1) - gw.h(idx,0));
			tmp = (dwc > tmp) ? dwc : tmp;
		} , Kokkos::Max<double>(eps) );
        return eps;
    }

    // /* --------------------------------------------------
    //     End eps computation
    // -------------------------------------------------- */


    // /* --------------------------------------------------
    //     Early stopping for the batch tests
    // -------------------------------------------------- */
    inline int early_stop(GwState &gw, Config &config) {
        // Check nans
        gw.ibreak = 0;
        for (int idx = 0; idx < config.ndom; idx++) {
            if (isnan(gw.h(idx,1)) == 1)    {
                gw.ibreak = 1;
                break;
            }
        }
        // Check oscillation
        if (gw.ibreak == 0) {
            // Non convergence for Picard
            if (config.scheme == 3 && gw.liniter >= config.iter_max)    {gw.ibreak = 2;}
            else if (config.scheme == 2)    {
                double vol, err;
                Kokkos::parallel_reduce( config.ndom , KOKKOS_LAMBDA (int idx, double &out) {
                    out += gw.wc(idx,1) * config.dx * config.dy * config.dz;
                } , Kokkos::Sum<double>(vol) );
                err = vol - gw.volume + gw.q(config.kM(0),2) * config.dt * config.dx * config.dy;
                gw.volume = vol;
                // printf(" ERR = %f\n",fabs(err/gw.volume));
                if (config.dt > 0.95*config.dt_max && fabs(err/gw.volume) > 5e-4) {
                    printf(" ERR = %f\n",fabs(err/gw.volume));
                    gw.ibreak = 2;
                }
            }

            // for (int idx = 2; idx < config.ndom-1; idx++) {
            //     // if (gw.wc(idx,1) > gw.wc(idx-1,1) && gw.wc(idx,1) > gw.wc(idx+1,1)) {
            //     //     gw.ibreak = 2;
            //     // }
            //     // else if (gw.wc(idx,1) < gw.wc(idx-1,1) && gw.wc(idx,1) < gw.wc(idx+1,1)) {
            //     //     gw.ibreak = 2;
            //     // }
            //     // else if (gw.wc(idx,1) < config.phi && gw.wc(idx+1,1) >= config.phi) {
            //     //     gw.ibreak = 2;
            //     // }
            //     if (gw.wc(idx,1) >= config.phi * 0.9999) {
            //         if (gw.wc(idx-1,1) < config.phi * 0.999)   {gw.ibreak = 2;}
            //     }
            // }

        }
        // Check if infltration reach midpoint
        if (gw.ibreak == 0) {
            int idx = config.nz/2;
            if (gw.wc(idx,1) >= config.wcr + 0.5*(config.phi - config.wcr)) {gw.ibreak = 3;}
        }

        // Write output file
        FILE *fp;
        char fname[100];
        strcpy(fname, config.fout);
        strcat(fname, "EndInfo");
        fp = fopen(fname, "w");
        fprintf(fp, "END INFO: \n");
        fprintf(fp, "Ks = %8.8f\n", config.rho * GRAV * config.kz / config.mu);
        fprintf(fp, "vga = %8.8f\n", config.vga);
        fprintf(fp, "vgn = %8.8f\n", config.vgn);
        fprintf(fp, "wcs = %8.8f\n", config.phi);
        fprintf(fp, "wcr = %8.8f\n", config.wcr);
        fprintf(fp, "wci = %8.8f\n", config.wc_init);
        fprintf(fp, "dtMax = %8.8f\n", config.dt_max);
        fprintf(fp, "Exit Code : %d\n", gw.ibreak);
        fclose(fp);

        if (gw.ibreak > 0)  {return 1;}
        else {return 0;}
    }

    // /* --------------------------------------------------
    //     End early stop
    // -------------------------------------------------- */



};

#endif
