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
	Kokkos::Timer timer;
	Kokkos::Timer timer2;
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
	//t_init = clock();
	t_init = timer2.seconds();

	/*
		----------    Time Stepping    ----------
	*/
	printf("  >>> Initialization completed! Begin time stepping...\n");
	gwf.update(gw, config);
	// Main time loop
	while (t_now <= config.t_end + config.dt)	{
		// Fully explicit scheme
		if (config.scheme == 1)	{
			t0 = timer.seconds();
			#if ALGO2
			gwf.face_conductivity2(gw, config);
			#else
			gwf.face_conductivity(gw, config);
			#endif
			Kokkos::fence();
			t1 = timer.seconds();
			t_k += (t1 - t0);

			t0 = timer.seconds();
		    gwf.face_flux(gw, config);
		    Kokkos::fence();
		    t1 = timer.seconds();
			t_flux += (t1 - t0);

			t0 = timer.seconds();
		    gwf.update_wc(gw, config);
		    Kokkos::fence();
		    t1 = timer.seconds();
			t_wc += (t1 - t0);

			t0 = timer.seconds();
		    gwf.dt_waco(gw, config);
		    gwf.update(gw, config);
		    Kokkos::fence();
		    t1 = timer.seconds();
			t_dt += (t1 - t0);

			iter = 1;	iter_gmres = 0;
			config.write_monitor3(t_now, iter, iter_gmres, "iter", config);
		}
		// PCA scheme
		else if (config.scheme == 2)	{
		    t0 = timer.seconds();
			//t0 = timer.seconds();
			#if ALGO2
			gwf.face_conductivity2(gw, config);
			#else
			gwf.face_conductivity(gw, config);
			#endif
			//t1 = timer.seconds();
			//t_k += (t1 - t0);
			Kokkos::fence();
			t1 = timer.seconds();
			t_k += (t1 - t0);


			t0 = timer.seconds();
		    gwf.linear_system(gw, A, config);
		    Kokkos::fence();
		    t1 = timer.seconds();
			t_matrix += (t1 - t0);


			t0 = timer.seconds();
			Kokkos::parallel_for(config.ndom, KOKKOS_LAMBDA(int idx) {
		        A.x(idx) = gw.h(idx,1);
		    });
			if (config.pcg_solve == 1)	{iter_gmres = gsolver.cg(A, config);}
			else {iter_gmres = gsolver.gmres(A, config);}
		    Kokkos::parallel_for(config.ndom, KOKKOS_LAMBDA(int idx) {
		        gw.h(idx,1) = A.x(idx);
		    });
			t1 = timer.seconds();
			t_solver += (t1 - t0);
			Kokkos::fence();

			t0 = timer.seconds();
			#if ALGO2
			gwf.face_conductivity2(gw, config);
			#else
			gwf.face_conductivity(gw, config);
			#endif
			Kokkos::fence();
			t1 = timer.seconds();
			t_k += (t1 - t0);


			t0 = timer.seconds();
		    gwf.face_flux(gw, config);
		    Kokkos::fence();
		    t1 = timer.seconds();
			t_flux += (t1 - t0);


			t0 = timer.seconds();
		    gwf.update_wc(gw, config);
		    Kokkos::fence();
		    t1 = timer.seconds();
			t_wc += (t1 - t0);

			// early stop
			int stp = gwf.early_stop(gw, config);
			if (stp == 1)	{
				config.write_output(gw.wc, "out-satu", 99, config);
				config.write_output(gw.h, "out-head", 99, config);
				exit(0);
			}
			else {
				if (t_now - t_out >= config.t_itvl)	{
					if (gw.wc(5,1) < config.wcr + 0.1*(config.phi-config.wcr))	{
						config.write_output(gw.wc, "out-satu", 99, config);
						config.write_output(gw.h, "out-head", 99, config);
						exit(0);
					}
				}
			}


			t0 = timer.seconds();
		    gwf.dt_waco(gw, config);
		    gwf.update(gw, config);
		    Kokkos::fence();
		    t1 = timer.seconds();
			t_dt += (t1 - t0);


		    iter = 1;
		    config.write_monitor3(t_now, iter, iter_gmres, "iter", config);

		}
		// Picard scheme
		else if (config.scheme == 3)	{
			eps = 1.0;	iter = 0;

			gwf.update(gw, config);

			Kokkos::parallel_for(config.ndom, KOKKOS_LAMBDA(int idx) {
				gw.wc(idx,2) = gw.wc(idx,1);
			});

			while (iter < config.iter_max & eps > config.eps_min)	{
				t0 = timer.seconds();
				gwf.linear_system(gw, A, config);
				Kokkos::fence();
				t1 = timer.seconds();
				t_matrix += (t1 - t0);

				t0 = timer.seconds();
				if (config.pcg_solve == 1)	{iter_gmres = gsolver.cg(A, config);}
				else {iter_gmres = gsolver.gmres(A, config);}
				Kokkos::parallel_for(config.ndom, KOKKOS_LAMBDA(int idx) {
					gw.h(idx,0) = gw.h(idx,1);
				    gw.h(idx,1) = A.x(idx);
				});
				Kokkos::fence();
				t1 = timer.seconds();
				t_solver += (t1 - t0);

				t0 = timer.seconds();
				#if ALGO2
				gwf.face_conductivity2(gw, config);
				#else
				gwf.face_conductivity(gw, config);
				#endif
				Kokkos::fence();
				t1 = timer.seconds();
				t_k += (t1 - t0);

				t0 = timer.seconds();
				gwf.face_flux(gw, config);
				Kokkos::fence();
				t1 = timer.seconds();
				t_flux += (t1 - t0);

				t0 = timer.seconds();
				Kokkos::parallel_for(config.ndom, KOKKOS_LAMBDA(int idx) {
					gw.wc(idx,0) = gw.wc(idx,1);
				});
				gwf.update_wc(gw, config);
				Kokkos::fence();
				t1 = timer.seconds();
				t_wc += (t1 - t0);

				eps = gwf.getErr(gw, config);

				iter += 1;
				config.write_monitor3(t_now, iter, iter_gmres, "iter", config);
				//if (eps_diff / eps_old < config.eps_min)	{break;}
				if (eps < config.eps_min)	{break;}
			}
			gw.liniter = iter;

			t0 = timer.seconds();
			#if ALGO2
			gwf.face_conductivity2(gw, config);
			#else
			gwf.face_conductivity(gw, config);
			#endif
			Kokkos::fence();
			t1 = timer.seconds();
			t_k += (t1 - t0);

			t0 = timer.seconds();
		    gwf.face_flux(gw, config);
		    Kokkos::fence();
		    t1 = timer.seconds();
			t_flux += (t1 - t0);

			// early stop
			int stp = gwf.early_stop(gw, config);
			if (stp == 1)	{
				config.write_output(gw.wc, "out-satu", 99, config);
				config.write_output(gw.h, "out-head", 99, config);
				exit(0);
			}
			else {
				if (t_now - t_out >= config.t_itvl)	{
					if (gw.wc(5,1) < config.wcr + 0.02)	{
						config.write_output(gw.wc, "out-satu", 99, config);
						config.write_output(gw.h, "out-head", 99, config);
						exit(0);
					}
				}
			}

			// printf("     >>> Picard converges in %d iterations! eps=%f \n", iter,fabs(eps));
		}
		// Newton scheme
		else 	{
			eps = 1.0;	iter = 0;
			nwt.update(gw, config);

			t0 = timer.seconds();
			#if ALGO2
			gwf.face_conductivity2(gw, config);
			#else
			gwf.face_conductivity(gw, config);
			#endif
			Kokkos::fence();
			t1 = timer.seconds();
			t_k += (t1 - t0);

			t0 = timer.seconds();
		    gwf.face_flux(gw, config);
		    Kokkos::fence();
		    t1 = timer.seconds();
			t_flux += (t1 - t0);

			int topc = 0;
			while (iter < config.iter_max & eps > config.eps_min)   {
				t0 = timer.seconds();
				nwt.newton_residual(gw, config);
				nwt.jacobian_offdiag(gw, config, 0);
				nwt.jacobian_offdiag(gw, config, 1);
				nwt.jacobian_offdiag(gw, config, 2);
				nwt.jacobian_diag(gw, A, config);
				Kokkos::fence();
				t1 = timer.seconds();
				t_matrix += (t1 - t0);

				// Solve with CG or GMRES
				t0 = timer.seconds();
				if (config.pcg_solve == 1)	{iter_gmres = gsolver.cg(A, config);}
				else {iter_gmres = gsolver.gmres(A, config);}
				Kokkos::parallel_for(config.ndom, KOKKOS_LAMBDA(int idx) {
					gw.dh(idx) = A.x(idx);	A.x(idx) = 0.0;
				});
				nwt.incre_h(gw, config);
				Kokkos::fence();
				t1 = timer.seconds();
				t_solver += (t1 - t0);

				eps_old = eps;
				eps = nwt.newton_eps(gw, config);
				eps_diff = fabs(eps_old - eps);
				//eps = nwt.getErr(gw, config);

				t0 = timer.seconds();
				#if ALGO2
				gwf.face_conductivity2(gw, config);
				#else
				gwf.face_conductivity(gw, config);
				#endif
				Kokkos::fence();
				t1 = timer.seconds();
				t_k += (t1 - t0);

				t0 = timer.seconds();
				gwf.face_flux(gw, config);
				Kokkos::fence();
				t1 = timer.seconds();
				t_flux += (t1 - t0);

				iter += 1;
				config.write_monitor3(t_now, iter, iter_gmres, "iter", config);
				if (eps_diff / eps_old < config.eps_min)	{break;}
				//if (eps < config.eps_min)	{break;}
				if (iter_gmres > 50)	{topc = 1;}

			}
			nwt.update_wc(gw, config);
			printf("     >>> Newton converges in %d iterations! eps=%f \n", iter,fabs(eps));

		}
		// Write outputs
		if (t_now - t_out >= config.t_itvl)	{
	    	printf("    >> Writing output at t = %f\n",t_now);
	    	config.write_output(gw.wc, "out-satu", i_out, config);
			config.write_output(gw.h, "out-head", i_out, config);
			printf(" >>>> Time %f completed!\n\n", t_now);
			i_out += 1;		t_out += config.t_itvl;
	    }
	    t_now += config.dt;
		if (config.scheme == 3)	{
			gwf.dt_iter(gw, iter, config);
		}
		else if (config.scheme == 4) 	{
			nwt.dt_iter(gw, iter, config);
		}

	}


	t_final = timer2.seconds();
	t_total = (t_final - t_init);
	printf("\n >>>>> Simulation completed in %f sec! num_sync=%d \n\n",t_total,config.num_sync);
	config.write_siminfo(t_total, t_matrix, t_solver, t_k, t_flux, t_wc, t_dt, "simtime", config);
	Kokkos::finalize();
	return 0;
}
