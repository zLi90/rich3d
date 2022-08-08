/*
	Solving 3D Richards Equation with PCA Scheme on Kokkos
*/
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <Kokkos_Core.hpp>

#include "config.h"
#include "state.h"
#include "matrix.h"
#include "newton.h"
#include "pca.h"
#include "solver.h"

int main(int argc, char** argv)	{
	int i_out = 0, iter, nthreads, iter_gmres;
	double t_now = 0.0, t_out = 0.0, t0, t1, t_init, t_final, t_cg = 0.0;
	double eps_gmres, eps, eps_old, eps_diff;
	Kokkos::InitArguments args;
	#ifdef __NVCC__
		nthreads = 1;
		Kokkos::initialize(argc, argv);
	#else
		nthreads = atoi(argv[2]);
		args.num_threads = nthreads;
		Kokkos::initialize(args);
	#endif
	// Read user inputs
	Config config;
	config.fdir = argv[1];
	config.init("input");
	// Initialize state variables
	//State H2O(config);
	Newton nwt;
	Pca pca;
	// Initialize solver
	Matrix A(config);
	Matrix K(config);
	Solver<dualDbl::execution_space> solver;
	solver.init(A, config);
	// Save initial states
	if (config.iter_solve == 1)	{
		nwt.init(config);
		config.write_output(nwt.wc, "out-satu", i_out, config);
		config.write_output(nwt.h, "out-head", i_out, config);
	}
	else	{
		pca.init(config);
		config.write_output(pca.wc, "out-satu", i_out, config);
		config.write_output(pca.h, "out-head", i_out, config);
	}
	i_out += 1;
	// Initialize timer
	//Kokkos::Timer timer;
	//timer.reset();
	t_init = clock();

	/*
		----------    Time Stepping    ----------
	*/
	printf("  >>> Initialization completed! Begin time stepping...\n");
	if (config.iter_solve == 1)	{nwt.update(config);}
	else	{pca.update(config);}
	// Main time loop
	while (t_now <= config.t_end + config.dt)	{
		if (config.iter_solve == 0)	{
			// Compute hydraulic conductivity on cell faces
			pca.get_conductivity(config);
			// Build the linear system of equations
			pca.linear_system(config);
			A.build_matrix(pca, config);
			A.decompose(config);
			// Solve with CG
			t0 = clock();
			if (config.pcg_solve == 1)	{solver.cg(A, K, config);}
			else	{solver.gmres(A, config);}
			t1 = clock();
			t_cg += (t1 - t0)/(float)CLOCKS_PER_SEC;
			solver.copy(A.x, 0, pca.h, 1);
			config.sync(pca.h);
			
			/*dualDbl::t_host h_h = pca.h.h_view;
			dualDbl::t_host h_wc = pca.wc.h_view;
			for (int ii = 0; ii < config.ndom; ii++)	{
				if (ii < 500 & config.k3d(ii) == config.nz-1)	{
					printf(" BEFOR (%d,%d,%d) : h = %f, %f, wc = %f\n",config.i3d(ii),config.j3d(ii),config.k3d(ii),h_h(ii,0),h_h(ii,1),h_wc(ii,1));
				}
			
			}
			printf(" -----\n");*/
			//config.print_view(pca.h, 32, 1, 1.0);
			
			// Update flux and water content
			pca.get_conductivity(config);
			pca.get_flux(config);
			// Update water content
			pca.update_wc(config);
			//pca.allocate_wc(config);
			
			/*for (int ii = 0; ii < config.ndom; ii++)	{
				if (ii < 500 & config.k3d(ii) == config.nz-1)	{
					printf(" AFTER (%d,%d,%d) : h = %f, %f, wc = %f\n",config.i3d(ii),config.j3d(ii),config.k3d(ii),h_h(ii,0),h_h(ii,1),h_wc(ii,1));
				}
			}*/
		}
		else	{
			eps = 1.0;	iter = 0;
			nwt.update(config);
			
			// Compute hydraulic conductivity on cell faces
			nwt.get_conductivity(config);
			nwt.get_flux(config);
			while (iter < config.iter_max & eps > config.eps_min)   {
				nwt.jacobian_system(config);
				A.build_matrix(nwt, config);
				A.decompose(config);
				// Solve with CG or GMRES
				t0 = clock();
				if (config.pcg_solve == 1)	{solver.cg(A, K, config);}
				else	{iter_gmres = solver.gmres(A, config);}
				t1 = clock();
				t_cg += (t1 - t0)/(float)CLOCKS_PER_SEC;
				
				solver.copy(A.x, 0, nwt.dh, 0);
				config.sync(nwt.dh);
				solver.reset(A.x, 0);
				config.sync(A.x);
				
				//config.print_view(nwt.matcoef, 5, 0, 1e5);
				//config.print_view(nwt.matcoef, 5, 1, 1e5);
				//config.print_view(nwt.matcoef, 5, 2, 1e5);
				//config.print_view(nwt.matcoef, 5, 5, 1e5);
				//config.print_view(nwt.matcoef, 5, 6, 1e5);
				//config.print_view(nwt.matcoef, 5, 7, 1e5);
				//config.print_view(nwt.dh, 5, 0, 1e5);
				
				
			    nwt.incre_h(config);
			    eps_old = eps;
			    eps = nwt.get_eps(config);
			    eps_diff = fabs(eps_old - eps);
			    
			    nwt.get_conductivity(config);
				nwt.get_flux(config);
				
			    iter += 1;
			    if (eps_diff / eps_old < config.eps_min)	{break;}
			}
			//config.sync(nwt.wc);
			nwt.update_wc(config);
			printf("     >>> Newton converges in %d iterations! eps=%f \n", iter,fabs(eps));
		}
		// Output
		if (t_now - t_out >= config.t_itvl)	{
	    	printf("    >> Writing output at t = %f\n",t_now);
	    	if (config.iter_solve == 1)	{
				config.write_output(nwt.wc, "out-satu", i_out, config);
				config.write_output(nwt.h, "out-head", i_out, config);
			}
			else	{
				config.write_output(pca.wc, "out-satu", i_out, config);
				config.write_output(pca.h, "out-head", i_out, config);
			}
			i_out += 1;		t_out += config.t_itvl;
	    }
	    t_now += config.dt;
	    if (config.iter_solve == 1)	{
	    	nwt.dt_iter(iter, config);
			nwt.update(config);
		}
		else	{
			pca.dt_waco(config);
	    	pca.update(config);
		}
	    
	    printf(" >>>> Time %f completed!\n\n", t_now);
	}
	
	t_final = clock();
	printf("\n >>>>> Simulation completed in %f sec! CG solves in %f sec\n\n",(t_final - t_init)/(float)CLOCKS_PER_SEC/nthreads, t_cg/nthreads);
	Kokkos::finalize();
	return 0;
}
