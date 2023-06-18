/*
	Solving 3D Richards Equation with PCA Scheme on Kokkos
*/
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <Kokkos_Core.hpp>

#include "config.h"
#include "GwState.h"
#include "GwFunction.h"
#include "GwMatrix.h"
#include "GwNewton.h"
#include "GwSolver.h"

int main(int argc, char** argv)	{
	int i_out = 0, iter, nthreads, iter_gmres;
	double t_now = 0.0, t_out = 0.0, t0, t1, t_init, t_final;
	double t_solver = 0.0, t_matrix = 0.0, t_flux = 0.0, t_wc = 0.0, t_total = 0.0, t_dt = 0.0, t_k = 0.0;
	double eps_gmres, eps, eps_old, eps_diff;
	double dt_newton, t_switch;
	Kokkos::InitArguments args;
	#ifdef __NVCC__
		nthreads = 1;
		Kokkos::initialize(argc, argv);
	#else
		nthreads = atoi(argv[3]);
		args.num_threads = nthreads;
		Kokkos::initialize(args);
	#endif
	// Read user inputs
	Config config;
	config.fdir = argv[1];
	config.fout = argv[2];
	config.init("input");
	// Initialize state variables
	GwState gw;
	gw.allocate(config);
	gw.initialize(config);
	GwFunction gwf;
	GwNewton nwt;
	// Initialize solver
	GwMatrix A(config);
	GwSolver gsolver;
	gsolver.init(A, config);
	// Save initial states
	config.write_output(gw.wc, "out-satu", i_out, config);
	config.write_output(gw.h, "out-head", i_out, config);
	i_out += 1;
	t_init = clock();

	/*
		----------    Time Stepping    ----------
	*/
	printf("  >>> Initialization completed! Begin time stepping...\n");
	gwf.update(gw, config);
	// Main time loop
	while (t_now <= config.t_end + config.dt)	{
		// Fully explicit scheme
		if (config.scheme == 1)	{
			t0 = clock();
			gwf.face_conductivity(gw, config);
			t1 = clock();
			t_k += (t1 - t0)/(float)CLOCKS_PER_SEC;
			
			t0 = clock();
		    gwf.face_flux(gw, config);
		    t1 = clock();
			t_flux += (t1 - t0)/(float)CLOCKS_PER_SEC;
			
			t0 = clock();
		    gwf.update_wc(gw, config);
		    t1 = clock();
			t_wc += (t1 - t0)/(float)CLOCKS_PER_SEC;
			
			t0 = clock();
		    gwf.dt_waco(gw, config);
		    gwf.update(gw, config);
		    t1 = clock();
			t_dt += (t1 - t0)/(float)CLOCKS_PER_SEC;
			
			iter = 1;	iter_gmres = 0;
			config.write_monitor3(t_now, iter, iter_gmres, "iter", config);
			// Optinal hybrid scheme
			/*if (t_now - t_switch > 1800.0)	{
				config.scheme = 4;
				config.dt_init = 1.0;
				Kokkos::deep_copy(nwt.h, expl.h);
				Kokkos::deep_copy(nwt.wc, expl.wc);
				config.dt = dt_newton;
				config.dt_max = 600.0;
			}*/
		}
		// PCA scheme (20 + 2*niter) sync operations
		else if (config.scheme == 2)	{
			t0 = clock();
			gwf.face_conductivity(gw, config);
			t1 = clock();
			t_k += (t1 - t0)/(float)CLOCKS_PER_SEC;
			
			t0 = clock();
		    gwf.linear_system(gw, A, config);
		    t1 = clock();
			t_matrix += (t1 - t0)/(float)CLOCKS_PER_SEC;
			
			t0 = clock();
			Kokkos::parallel_for(config.ndom, KOKKOS_LAMBDA(int idx) {
		        A.x(idx) = gw.h(idx,1);
		    });
			if (config.pcg_solve == 1)	{iter_gmres = gsolver.cg(A, config);}
			else {iter_gmres = gsolver.gmres(A, config);}
		    Kokkos::parallel_for(config.ndom, KOKKOS_LAMBDA(int idx) {
		        gw.h(idx,1) = A.x(idx);
		    });
			t1 = clock();
			t_solver += (t1 - t0)/(float)CLOCKS_PER_SEC;

			t0 = clock();
			gwf.face_conductivity(gw, config);
			t1 = clock();
			t_k += (t1 - t0)/(float)CLOCKS_PER_SEC;
			
			t0 = clock();
		    gwf.face_flux(gw, config);
		    t1 = clock();
			t_flux += (t1 - t0)/(float)CLOCKS_PER_SEC;

			t0 = clock();
		    gwf.update_wc(gw, config);
		    t1 = clock();
			t_wc += (t1 - t0)/(float)CLOCKS_PER_SEC;

			t0 = clock();
		    gwf.dt_waco(gw, config);
		    gwf.update(gw, config);
		    t1 = clock();
			t_dt += (t1 - t0)/(float)CLOCKS_PER_SEC;
			
		    iter = 1;
		    config.write_monitor3(t_now, iter, iter_gmres, "iter", config);
			
			//t_now = config.t_end + 100.0;
			
			/*if (t_now - t_switch > 1800.0)	{
				printf(" Switching back to Newton ... %f, %f\n",t_now,t_switch);
				config.scheme = 4;
				config.pcg_solve = 0;
				Kokkos::deep_copy(nwt.h, pca.h);
				Kokkos::deep_copy(nwt.wc, pca.wc);
			}*/
		}
		// Picard scheme
		else if (config.scheme == 3)	{
			eps = 1.0;	iter = 0;
			
			//printf(" wc = %f, %f, h = %f, %f\n",gw.wc(config.kM(0),1), gw.wc(0,1),gw.h(config.kM(0),1),gw.h(0,1));
			
			gwf.update(gw, config);
			
			Kokkos::parallel_for(config.ndom, KOKKOS_LAMBDA(int idx) {
				gw.wc(idx,2) = gw.wc(idx,1);
			});
			
			while (iter < config.iter_max & eps > config.eps_min)	{
				t0 = clock();
				gwf.linear_system(gw, A, config);
				t1 = clock();
				t_matrix += (t1 - t0)/(float)CLOCKS_PER_SEC;
				
				t0 = clock();
				if (config.pcg_solve == 1)	{iter_gmres = gsolver.cg(A, config);}
				else {iter_gmres = gsolver.gmres(A, config);}
				Kokkos::parallel_for(config.ndom, KOKKOS_LAMBDA(int idx) {
					gw.h(idx,0) = gw.h(idx,1);
				    gw.h(idx,1) = A.x(idx);
				});
				t1 = clock();
				t_solver += (t1 - t0)/(float)CLOCKS_PER_SEC;
				
				t0 = clock();
				gwf.face_conductivity(gw, config);
				t1 = clock();
				t_k += (t1 - t0)/(float)CLOCKS_PER_SEC;
				
				t0 = clock();
				gwf.face_flux(gw, config);
				t1 = clock();
				t_flux += (t1 - t0)/(float)CLOCKS_PER_SEC;
				
				t0 = clock();
				Kokkos::parallel_for(config.ndom, KOKKOS_LAMBDA(int idx) {
					gw.wc(idx,0) = gw.wc(idx,1);
				});
				gwf.update_wc(gw, config);
				t1 = clock();
				t_wc += (t1 - t0)/(float)CLOCKS_PER_SEC;
				
				eps = gwf.getErr(gw, config);
				
				iter += 1;
				config.write_monitor3(t_now, iter, iter_gmres, "iter", config);
				//if (eps_diff / eps_old < config.eps_min)	{break;}
				if (eps < config.eps_min)	{break;}
			}
			t0 = clock();
			gwf.face_conductivity(gw, config);
			t1 = clock();
			t_k += (t1 - t0)/(float)CLOCKS_PER_SEC;
			
			t0 = clock();
		    gwf.face_flux(gw, config);
		    t1 = clock();
			t_flux += (t1 - t0)/(float)CLOCKS_PER_SEC;
			
			printf("     >>> Picard converges in %d iterations! eps=%f \n", iter,fabs(eps));
		}
		// Newton scheme
		else 	{
			eps = 1.0;	iter = 0;
			nwt.update(gw, config);
			
			t0 = clock();
			gwf.face_conductivity(gw, config);
			t1 = clock();
			t_k += (t1 - t0)/(float)CLOCKS_PER_SEC;
			
			t0 = clock();
		    gwf.face_flux(gw, config);
		    t1 = clock();
			t_flux += (t1 - t0)/(float)CLOCKS_PER_SEC;
			
			while (iter < config.iter_max & eps > config.eps_min)   {
				t0 = clock();
				nwt.newton_residual(gw, config);
				nwt.jacobian_offdiag(gw, config, 0);
				nwt.jacobian_offdiag(gw, config, 1);
				nwt.jacobian_offdiag(gw, config, 2);
				nwt.jacobian_diag(gw, A, config);
				t1 = clock();
				t_matrix += (t1 - t0)/(float)CLOCKS_PER_SEC;
				
				// Solve with CG or GMRES
				t0 = clock();
				if (config.pcg_solve == 1)	{iter_gmres = gsolver.cg(A, config);}
				else {iter_gmres = gsolver.gmres(A, config);}
				Kokkos::parallel_for(config.ndom, KOKKOS_LAMBDA(int idx) {
					gw.dh(idx) = A.x(idx);	A.x(idx) = 0.0;
				});
				nwt.incre_h(gw, config);
				t1 = clock();
				t_solver += (t1 - t0)/(float)CLOCKS_PER_SEC;

				eps_old = eps;
				eps = nwt.newton_eps(gw, config);
				eps_diff = fabs(eps_old - eps);
				//eps = nwt.getErr(gw, config);

				t0 = clock();
				gwf.face_conductivity(gw, config);
				t1 = clock();
				t_k += (t1 - t0)/(float)CLOCKS_PER_SEC;
				
				t0 = clock();
				gwf.face_flux(gw, config);
				t1 = clock();
				t_flux += (t1 - t0)/(float)CLOCKS_PER_SEC;

				iter += 1;
				config.write_monitor3(t_now, iter, iter_gmres, "iter", config);
				if (eps_diff / eps_old < config.eps_min)	{break;}
				//if (eps < config.eps_min)	{break;}
			}
			// Optinal hybrid scheme
			/*if (iter_gmres > 250)	{
				config.scheme = 1;
				dt_newton = config.dt;
				config.dt_init = 0.01;
				config.dt = config.dt_init;
				Kokkos::deep_copy(expl.h, nwt.h);
				Kokkos::deep_copy(expl.wc, nwt.wc);
				t_switch = t_now;
				config.dt_max = 1.0;
			}*/
			/*if (iter_gmres > 50)	{
				config.scheme = 2;
				Kokkos::deep_copy(pca.h, nwt.h);
				Kokkos::deep_copy(pca.wc, nwt.wc);
				t_switch = t_now;
				config.pcg_solve = 1;
			}*/
			nwt.update_wc(gw, config);
			printf("     >>> Newton converges in %d iterations! eps=%f \n", iter,fabs(eps));
			
		}
		// Write outputs
		if (t_now - t_out >= config.t_itvl)	{
	    	printf("    >> Writing output at t = %f\n",t_now);
	    	config.write_output(gw.wc, "out-satu", i_out, config);
			config.write_output(gw.h, "out-head", i_out, config);
			i_out += 1;		t_out += config.t_itvl;
	    }
	    t_now += config.dt;
		if (config.scheme == 1)	{
			//gwf.dt_waco(gw, config);
			//gwf.update(gw, config);
		}
		else if (config.scheme == 2)	{
			//gwf.dt_waco(gw, config);
			//gwf.update(gw, config);
		}
		else if (config.scheme == 3)	{
			gwf.dt_iter(gw, iter, config);
			//gwf.update(gw, config);
		}
		else 	{
			nwt.dt_iter(gw, iter, config);
			//gwf.update(gw, config);
		}
	    printf(" >>>> Time %f completed!\n\n", t_now);
	    
	}

	t_final = clock();
	t_total = (t_final - t_init)/(float)CLOCKS_PER_SEC/nthreads;
	printf("\n >>>>> Simulation completed in %f sec! num_sync=%d \n\n",t_total,config.num_sync);
	config.write_siminfo(t_total, t_matrix/nthreads, t_solver/nthreads, t_k/nthreads, t_flux/nthreads, t_wc/nthreads, t_dt/nthreads, "simtime", config);
	Kokkos::finalize();
	return 0;
}
