/*
    User settings and related parameters
*/
#ifndef _CONFIG_H_
#define _CONFIG_H_

#include <algorithm>
#include <math.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include<stdio.h>
#include<stdlib.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

// Constants
#define GRAV 9.81

#ifndef TEST_TRACY
#define TEST_TRACY 0
#endif

#ifndef TEST_BEEGUM
#define TEST_BEEGUM 0
#endif

#ifdef __NVCC__
typedef Kokkos::View<double* ,Kokkos::LayoutRight,Kokkos::Device<Kokkos::Cuda,Kokkos::CudaUVMSpace>> DblArr;
typedef Kokkos::View<int* ,Kokkos::LayoutRight,Kokkos::Device<Kokkos::Cuda,Kokkos::CudaUVMSpace>> IntArr;
typedef Kokkos::View<int** ,Kokkos::LayoutRight,Kokkos::Device<Kokkos::Cuda,Kokkos::CudaUVMSpace>> IntArr2;
#else
typedef Kokkos::View<double* ,Kokkos::LayoutRight> DblArr;
typedef Kokkos::View<int* ,Kokkos::LayoutRight> IntArr;
typedef Kokkos::View<int** ,Kokkos::LayoutRight> IntArr2;
#endif


typedef Kokkos::DualView<int**> dualInt;
typedef Kokkos::DualView<double**> dualDbl;

typedef dualDbl::execution_space dspace;


// User configurations
class Config {

public:
	// solver
	int iter_solve, pcg_solve, niter_solver, exp_solve, precondition, scheme;
    // Domain info
    int nx, ny, nz, nt, nall, ndom, nbcell, init_file;
    double dx, dy, dz, dt, t_end, t_itvl, dt_max, dt_init;
    // Initial and Boundary conditions
    int bc_type_xp, bc_type_xm, bc_type_yp, bc_type_ym, bc_type_zp, bc_type_zm;
    double bc_val_xp, bc_val_xm, bc_val_yp, bc_val_ym, bc_val_zp, bc_val_zm;
    double h_init, wc_init, wt_init;
    // properties
    double rho, mu, kx, ky, kz, vga, vgn, phi, wcr, ss;
    // Map
    IntArr Ct, iP, iM, jP, jM, kP, kM, i3d, j3d, k3d;
    DblArr bcval;
    IntArr2 bcell;
    // Simulation control
    char *fdir, *fout;
    char finput[200];
    int iter_max;
    double eps_min;

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

	// Write one state variable to the output file
	inline void write_output(dualDbl array, char *fieldname, int t_ind, Config config)	{
		int ii;
		FILE *fid;
		char t_str[10], filename[100];
		double x, y, z;
		// generate output file name
		sprintf(t_str, "%d", t_ind);
		//strcpy(filename, config.fdir);
		strcpy(filename, config.fout);
		strcat(filename, fieldname);
		strcat(filename, t_str);
		// write data
		dualDbl::t_host h_array = array.h_view;
		array.sync<dualDbl::host_mirror_space> ();
		fid = fopen(filename, "w");
		for (ii = 0; ii < config.ndom; ii++)	{
			x = (config.i3d(ii)+0.5) * config.dx;
			y = (config.j3d(ii)+0.5) * config.dy;
			z = (config.k3d(ii)+0.5) * config.dz;
			fprintf(fid, "%6.6f, %6.6f, %6.6f, %6.6f \n", x, y, z, h_array(ii,1));
		}
		fclose(fid);
	}

	// >>>>> Write point monitoring results <<<<<
	inline void write_monitor(double val, char *fieldname, Config config)	{
		FILE *fp;
		char filename[100];
		strcpy(filename, config.fout);
		strcat(filename, fieldname);
		if (exist(filename))    {fp = fopen(filename, "a");}
		else    {fp = fopen(filename, "w");}
		fprintf(fp, "%8.8f \n", val);
		fclose(fp);
	}

	inline void write_monitor3(double val1, double val2, double val3, char *fieldname, Config config)	{
		FILE *fp;
		char filename[100];
		strcpy(filename, config.fout);
		strcat(filename, fieldname);
		if (exist(filename))    {fp = fopen(filename, "a");}
		else    {fp = fopen(filename, "w");}
		fprintf(fp, "%8.8f %8.8f %8.8f\n", val1, val2, val3);
		fclose(fp);
	}
	
	inline void write_siminfo(double t_tot, double t_mat, double t_sol, char *fieldname, Config config)	{
		double t_oth = t_tot - t_mat - t_sol;
		FILE *fp;
		char filename[100];
		strcpy(filename, config.fout);
		strcat(filename, fieldname);
		if (exist(filename))    {fp = fopen(filename, "a");}
		else    {fp = fopen(filename, "w");}
		fprintf(fp, "Total Time | Matrix Time | Solver Time | Other Time (sec)\n");
		fprintf(fp, "%8.8f %8.8f %8.8f %8.8f\n", t_tot, t_mat, t_sol, t_oth);
		fprintf(fp, "%8.8f %8.8f %8.8f %8.8f\n", t_tot/t_tot, t_mat/t_tot, t_sol/t_tot, t_oth/t_tot);
		fclose(fp);
	}

	inline int exist(char *fname)	{
		FILE *fid;
		if ((fid = fopen(fname, "r")))
		{fclose(fid);   return 1;}
		return 0;
	}

	// Setting up model parameters and connection map
    inline void init(const char inFile[]) {
    	int kk = 0;
    	// Read input file
    	strcpy(finput, fdir);
		strcat(finput, inFile);
		scheme = (int) read_one_input("scheme", finput);
		//iter_solve = (int) read_one_input("iter_solve", finput);
		pcg_solve = (int) read_one_input("pcg_solve", finput);
		//exp_solve = (int) read_one_input("expl_solve", finput);
		precondition = (int) read_one_input("precondition", finput);
		// Build output directory
		mkdir(fout, 0777);
		// Domain information
		nx = (int) read_one_input("nx", finput);
		ny = (int) read_one_input("ny", finput);
		nz = (int) read_one_input("nz", finput);
		dx = read_one_input("dx", finput);
		dy = read_one_input("dy", finput);
		dz = read_one_input("dz", finput);
		dt_max = read_one_input("dt_max", finput);
		dt_init = read_one_input("dt_init", finput);
		t_end = read_one_input("t_end", finput);
		t_itvl = read_one_input("t_itvl", finput);
		// Simulation control
		iter_max = (int) read_one_input("iter_max", finput);
		eps_min = read_one_input("eps_min", finput);
		// Media properties
		kx = read_one_input("kx", finput);
		ky = read_one_input("ky", finput);
		kz = read_one_input("kz", finput);
		phi = read_one_input("phi", finput);
		wcr = read_one_input("wcr", finput);
		ss = read_one_input("ss", finput);
		vga = read_one_input("vga", finput);
		vgn = read_one_input("vgn", finput);
		rho = read_one_input("rho", finput);
		mu = read_one_input("mu", finput);
		if (nx == 1)	{kx = 0.0;}
		if (ny == 1)	{ky = 0.0;}
		if (nz == 1)	{kz = 0.0;}
		dt = dt_init;
		// Initial and boundary conditions
		bc_type_xp = (int) read_one_input("bc_type_xp", finput);
		bc_type_xm = (int) read_one_input("bc_type_xm", finput);
		bc_type_yp = (int) read_one_input("bc_type_yp", finput);
		bc_type_ym = (int) read_one_input("bc_type_ym", finput);
		bc_type_zp = (int) read_one_input("bc_type_zp", finput);
		bc_type_zm = (int) read_one_input("bc_type_zm", finput);
		bc_val_xp = read_one_input("bc_val_xp", finput);
		bc_val_xm = read_one_input("bc_val_xm", finput);
		bc_val_yp = read_one_input("bc_val_yp", finput);
		bc_val_ym = read_one_input("bc_val_ym", finput);
		bc_val_zp = read_one_input("bc_val_zp", finput);
		bc_val_zm = read_one_input("bc_val_zm", finput);
		init_file = (int) read_one_input("init_file", finput);
		h_init = read_one_input("h_init", finput);
		wc_init = read_one_input("wc_init", finput);
		wt_init = read_one_input("wt_init", finput);
		// Build connection map
    	ndom = nx * ny * nz;
		//nall = (nx+2) * (ny+2) * (nz+2);
		nbcell = 2.0*(nx*ny + nx*nz + ny*nz);
		nall = ndom + nbcell;
        Ct = IntArr("ct", ndom);
        iP = IntArr("iP", ndom);	iM = IntArr("iM", ndom);
        jP = IntArr("jP", ndom);	jM = IntArr("jM", ndom);
        kP = IntArr("kP", ndom);	kM = IntArr("kM", ndom);
        i3d = IntArr("i3d", ndom);	j3d = IntArr("j3d", ndom);
        k3d = IntArr("k3d", ndom);
        bcell = IntArr2("bc", nbcell, 4);	bcval = DblArr("bcval", nbcell);
        for (int idx = 0; idx < ndom; idx++)	{
            j3d(idx) = floor(idx / (nx*nz));
            k3d(idx) = idx % nz;
            i3d(idx) = floor((idx - j3d(idx)*nx*nz) / nz);
            Ct(idx) = idx;
            if (k3d(idx) == 0)	{kM(idx) = j3d(idx)*nx + i3d(idx) + ndom + 2*(nx*nz+ny*nz) + nx*ny;}
            else	{kM(idx) = idx - 1;}
            if (k3d(idx) == nz-1)	{kP(idx) = j3d(idx)*nx + i3d(idx) + ndom + 2*(nx*nz+ny*nz);}
            else	{kP(idx) = idx + 1;}
            if (i3d(idx) == 0)   {iM(idx) = j3d(idx)*nz + k3d(idx) + ndom + 2*nx*nz + ny*nz;}
            else    {iM(idx) = idx - nz;}
            if (i3d(idx) == nx-1)	{iP(idx) = j3d(idx)*nz + k3d(idx) + ndom + 2*nx*nz;}
            else	{iP(idx) = idx + nz;}
            if (j3d(idx) == 0)	{jM(idx) = i3d(idx)*nz + k3d(idx) + ndom + nx*nz;}
            else	{jM(idx) = idx - nx*nz;}
            if (j3d(idx) == ny-1)	{jP(idx) = i3d(idx)*nz + k3d(idx) + ndom;}
            else	{jP(idx) = idx + nx*nz;}
            // store bcells (bcell = [bctype, interior cell id, exterior cell id, axis])
            if (i3d(idx) == 0)	{
            	bcell(kk,0) = bc_type_xm;	bcell(kk,3) = -1;
            	bcell(kk,1) = idx;	bcell(kk,2) = iM(idx);
            	bcval(kk) = bc_val_xm;	kk += 1;
        	}
            if (i3d(idx) == nx-1)	{
            	bcell(kk,0) = bc_type_xp;	bcell(kk,3) = 1;
            	bcell(kk,1) = idx;	bcell(kk,2) = iP(idx);
            	bcval(kk) = bc_val_xp;	kk += 1;
        	}
            if (j3d(idx) == 0)	{
            	bcell(kk,0) = bc_type_ym;	bcell(kk,3) = -2;
            	bcell(kk,1) = idx;	bcell(kk,2) = jM(idx);
            	bcval(kk) = bc_val_ym;	kk += 1;
        	}
            if (j3d(idx) == ny-1)	{
            	bcell(kk,0) = bc_type_yp;	bcell(kk,3) = 2;
            	bcell(kk,1) = idx;	bcell(kk,2) = jP(idx);
            	bcval(kk) = bc_val_yp;	kk += 1;
        	}
			if (k3d(idx) == 0)	{
				bcell(kk,0) = bc_type_zm;	bcell(kk,3) = -3;
				bcell(kk,1) = idx;	bcell(kk,2) = kM(idx);
				bcval(kk) = bc_val_zm;	kk += 1;
			}
            if (k3d(idx) == nz-1)	{
            	bcell(kk,0) = bc_type_zp;	bcell(kk,3) = 3;
            	bcell(kk,1) = idx;	bcell(kk,2) = kP(idx);
            	bcval(kk) = bc_val_zp;	kk += 1;
        	}
        }
    }

    /*
    	Some useful functions
    */
    // print out the first n elements of a view
    inline void print_view(dualDbl vec, int n, int col, double scale)	{
    	dualDbl::t_host h_vec = vec.h_view;
    	for (int ii = 0; ii < n; ii++)	{
    		printf(" PRINT : -%d- val=%f\n",ii,scale * h_vec(ii,col));
    	}
    	printf(" ----- \n");
    }
    inline void printi_view(dualInt vec, int n, int col)	{
    	dualInt::t_host h_vec = vec.h_view;
    	for (int ii = 0; ii < n; ii++)	{
    		printf(" PRINT : -%d- val=%d\n",ii,h_vec(ii,col));
    	}
    	printf(" ----- \n");
    }

    // sync
    void sync(dualDbl vector)	{vector.sync<dualDbl::host_mirror_space> ();}
    void synci(dualInt vector)	{vector.sync<dualInt::host_mirror_space> ();}


};

#endif
