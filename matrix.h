/* 
	The sparse matrix 
*/
#ifndef _MATRIX_H_
#define _MATRIX_H_

#include "config.h"
#include "state.h"

class Matrix {

public:
	int nrow, ncol, nnz, nx, ny, nz;
	dualInt ptr, ind, ptrT, indT;
	dualDbl val, diag, diagN, rhs, x, valT, lt, ut;
	// initialize 
	Matrix(Config config)	{
		int ndom = config.ndom;
		nx = config.nx;	ny = config.ny;	nz = config.nz;
		nrow = ndom;
		ncol = ndom;
		// Number of non-zeros for 1D-z, 2D-xz, 3D-xyz simulations
		if (nx == 1 & ny == 1 & nz > 1)	{
			nnz = nrow + (nz-2)*2 + 2;
		}
		else if (nx > 1 & ny == 1 & nz > 1)	{
			nnz = nrow + 4*2 + ((nx-2)*2 + (nz-2)*2)*3 + (nx-2)*(nz-2)*4;
		}
		else if (nx > 1 & ny > 1 & nz > 1)	{
			nnz = nrow + 3*8 + 4*((nx-2)*4 + (ny-2)*4 + (nz-2)*4) +
		    	5*((nx-2)*(nz-2)*2 + (nx-2)*(ny-2)*2 + (ny-2)*(nz-2)*2) +
		    	6*((nx-2)*(ny-2)*(nz-2));
		}
		else	{printf("ERROR : Domain must be 1D-z, 2D-xz or 3D-xyz!\n");}
		
		ptr = dualInt("ptr", nrow+1, 1);	ind = dualInt("ind", nnz, 1);
		val = dualDbl("val", nnz, 1);		diag = dualDbl("diag", nnz, 3);
		rhs = dualDbl("rhs", nrow, 1);		x = dualDbl("x", nrow, 1);
		diagN = dualDbl("diagN", nrow, 1);	ut = dualDbl("ut", nnz, 1);
		valT = dualDbl("valT", nnz, 1);		lt = dualDbl("lt", nnz, 1);
		
	}
	
	// insert matrix entries
	inline void build_matrix(State state, Config config)	{
		int nx, ny, nz;
		nx = config.nx;	ny = config.ny;	nz = config.nz;
		// from device to host
		// modified views
		dualInt::t_host h_ptr = ptr.h_view;
		for (int idx = 0; idx < nrow+1; idx++)	{
			if (idx == nrow) {h_ptr(idx,0) = nnz;}
			else	{h_ptr(idx,0) = get_irow(config.i3d(idx), config.j3d(idx), config.k3d(idx), nx, ny, nz);}
		}
		ptr.modify<dualInt::host_mirror_space> ();
		
		Kokkos::parallel_for(nrow, Insert<dspace>(ptr, ind, val, rhs, state.matcoef, config));
		config.sync(rhs);
		config.sync(val);
		config.synci(ptr);
		config.synci(ind);
		Kokkos::fence();
	}
	
	// get starting index 
	inline int get_irow(int i, int j, int k, int nx, int ny, int nz) {
		int nrowk = 0, nlayer1 = 0, nlayerj = 0, ncol1 = 0, ncoli = 0, irow;
		// Assume nz > 0 by default
		if (j == 0)	{
			if (i == 0)	{
				// 1D
				if (nx == 1)	{if (k > 0)	{nrowk = 2 + (k-1)*3;}}
				else	{
					// 2D
					if (ny == 1)	{if (k > 0)	{nrowk = 3 + (k-1)*4;}}
					// 3D
					else	{if (k > 0)	{nrowk = 4 + (k-1)*5;}}
				}	
			}
			else	{
				// 2D
				if (ny == 1)	{
					ncol1 = 3*2 + 4*(nz-2);	ncoli = (i-1)*(4*2 + 5*(nz-2));
					if (i == nx-1)	{if (k > 0)	{nrowk = 3 + (k-1)*4;}}
					else	{if (k > 0)	{nrowk = 4 + (k-1)*5;}}
				}
				// 3D
				else	{
					ncol1 = 4*2 + 5*(nz-2);	ncoli = (i-1)*(5*2 + 6*(nz-2));
					if (i == nx-1)	{if (k > 0)	{nrowk = 4 + (k-1)*5;}}
					else	{if (k > 0)	{nrowk = 5 + (k-1)*6;}}
				}
			}
		}
		else	{
			nlayer1 = 4*4 + 5*2*(nx-2+nz-2) + 6*(nx-2)*(nz-2);
            nlayerj = (j-1)*(5*4 + 6*2*(nx-2+nz-2) + 7*(nx-2)*(nz-2));
            if (j == ny-1)	{
            	if (i == 0)	{
            		if (k > 0)	{nrowk = 4 + (k-1)*5;}
            	}
            	else	{
            		ncol1 = 4*2 + 5*(nz-2);	
		        	ncoli = (i-1)*(5*2 + 6*(nz-2));
		        	if (i == nx-1)	{if (k > 0)	{nrowk = 4 + (k-1)*5;}}
					else	{if (k > 0)	{nrowk = 5 + (k-1)*6;}}
            	}
            }
            else	{
            	if (i == 0)	{
            		if (k > 0)	{nrowk = 5 + (k-1)*6;}
            	}
            	else	{
            		ncol1 = 5*2 + 6*(nz-2);	
					ncoli = (i-1)*(6*2 + 7*(nz-2));
					if (i == nx-1)	{if (k > 0)	{nrowk = 5 + (k-1)*6;}}
					else	{if (k > 0)	{nrowk = 6 + (k-1)*7;}}
            	}
            }
		}
		irow = nlayer1 + nlayerj + ncol1 + ncoli + nrowk;
		return irow;
	}
	
	// LU destateosition
	inline void decompose(Config config)	{
		Kokkos::parallel_for(nrow, Decomp<dspace>(ptr, ind, val, diag, diagN, ut, lt));
		config.sync(diag);
		config.sync(diagN);
	}
	
				
	template<class ExecutionSpace>
    struct Insert {
    	typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value, dualInt::memory_space, dualInt::host_mirror_space>::type msi;
    	typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value, dualDbl::memory_space, dualDbl::host_mirror_space>::type ms;
    	Kokkos::View<dualInt::scalar_array_type, dualInt::array_layout, msi> ptr;
    	Kokkos::View<dualInt::scalar_array_type, dualInt::array_layout, msi> ind;
		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> val;
		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> rhs;
		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> coef;
		IntArr ii, jj, kk;
		int nx, ny, nz;
   		Insert(dualInt d_ptr, dualInt d_ind, dualDbl d_val, dualDbl d_rhs, dualDbl d_coef, Config config)	{
   			ptr = d_ptr.template view<msi> ();	d_ptr.sync<msi> ();	
   			ind = d_ind.template view<msi> ();	d_ind.sync<msi> ();	d_ind.modify<msi> ();
   			val = d_val.template view<ms> ();	d_val.sync<ms> ();	d_val.modify<ms> ();
   			rhs = d_rhs.template view<ms> ();	d_rhs.sync<ms> ();	d_rhs.modify<ms> ();
   			coef = d_coef.template view<ms> ();	d_coef.sync<ms> ();
   			nx = config.nx;	ny = config.ny;	nz = config.nz;
   			ii = config.i3d;	jj = config.j3d;	kk = config.k3d;
   		}
    	KOKKOS_INLINE_FUNCTION	void operator() (const int idx) const {
    		int irow = ptr(idx,0);
        	if (jj(idx) > 0)	{ind(irow,0) = idx - nx*nz;	val(irow,0) = coef(idx,4);  irow++;}
        	if (ii(idx) > 0)	{ind(irow,0) = idx - nz;	val(irow,0) = coef(idx,2);  irow++;}
        	if (kk(idx) > 0)	{ind(irow,0) = idx - 1;		val(irow,0) = coef(idx,6);  irow++;}
        	ind(irow,0) = idx;	val(irow,0) = coef(idx,0);	irow++;
        	if (kk(idx) < nz-1)	{ind(irow,0) = idx + 1;		val(irow,0) = coef(idx,5);  irow++;}
        	if (ii(idx) < nx-1)	{ind(irow,0) = idx + nz;	val(irow,0) = coef(idx,1);  irow++;}
        	if (jj(idx) < ny-1)	{ind(irow,0) = idx + nx*nz;	val(irow,0) = coef(idx,3);  irow++;}
        	rhs(idx,0) = coef(idx,7);
    	}
    };
    
    template<class ExecutionSpace>
    struct Decomp {
    	typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value, dualInt::memory_space, dualInt::host_mirror_space>::type msi;
    	typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value, dualDbl::memory_space, dualDbl::host_mirror_space>::type ms;
    	Kokkos::View<dualInt::scalar_array_type, dualInt::array_layout, msi> ptr;
    	Kokkos::View<dualInt::scalar_array_type, dualInt::array_layout, msi> ind;
		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> val;
		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> diag;
		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> diagN;
		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> ut;
		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> lt;
   		Decomp(dualInt d_ptr, dualInt d_ind, dualDbl d_val, dualDbl d_diag, dualDbl d_diagN,
   			dualDbl d_ut, dualDbl d_lt)	{
   			ptr = d_ptr.template view<msi> ();	d_ptr.sync<msi> ();	
   			ind = d_ind.template view<msi> ();	d_ind.sync<msi> ();	
   			val = d_val.template view<ms> ();	d_val.sync<ms> ();	
   			diag = d_diag.template view<ms> ();	d_diag.sync<ms> ();	d_diag.modify<ms> ();
   			diagN = d_diagN.template view<ms> ();	d_diagN.sync<ms> ();	d_diagN.modify<ms> ();
   			ut = d_ut.template view<ms> ();	d_ut.sync<ms> ();	d_ut.modify<ms> ();
   			lt= d_lt.template view<ms> ();	d_lt.sync<ms> ();	d_lt.modify<ms> ();
   		}
    	KOKKOS_INLINE_FUNCTION	void operator() (const int idx) const {
    		int icol;
    		diagN(idx,0) = 0.0;
    		for (icol = ptr(idx,0); icol < ptr(idx+1,0); icol++)	{
    			if (ind(icol,0) == idx)	{
    				diag(icol,0) = val(icol,0);	diag(icol,1) = 0.0;	diag(icol,2) = 0.0;
    				ut(icol,0) = val(icol,0);	lt(icol,0) = val(icol,0);
    				diagN(idx,0) = val(icol,0);
    			}
    			else if (ind(icol,0) < idx)	{
    				diag(icol,0) = 0.0;	diag(icol,1) = val(icol,0);	diag(icol,2) = 0.0;
    				lt(icol,0) = val(icol,0);
    			}
    			else if (ind(icol,0) > idx)	{
    				diag(icol,0) = 0.0;	diag(icol,1) = 0.0;	diag(icol,2) = val(icol,0);
    				ut(icol,0) = val(icol,0);
    			}
    		}
    	}
    };
    

};

#endif
