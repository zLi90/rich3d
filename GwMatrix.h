/*
	The sparse matrix
*/
#ifndef _GW_MATRIX_H_
#define _GW_MATRIX_H_

#include "config.h"

class GwMatrix {

public:
	int nrow, ncol, nnz, nx, ny, nz, gsteps;
	double iter_time, matvec_time, precond_time;
	IntArr ptr, ind, ptrT, indT;
	DblArr val, diag, rhs, x, valT, lt, ut;
	// views for cg solver
	DblArr r, z, p, q;
	// views for gmres solver
	DblArr s, y, c1, c2;
	DblArr2 v, h;

	// initialize
	GwMatrix(Config &config)	{
		int ii, jj, kk, ndom = config.ndom;
		nx = config.nx;	ny = config.ny;	nz = config.nz;
		nrow = ndom;
		ncol = ndom;
		gsteps = 40;
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
		ptr = IntArr("ptr", nrow+1);		ind = IntArr("ind", nnz);
		val = DblArr("val", nnz);			diag = DblArr("diag", nrow);
		rhs = DblArr("rhs", nrow);			x = DblArr("x", nrow);
		r = DblArr("r", nrow);				z = DblArr("z", nrow);
		p = DblArr("p", nrow);				q = DblArr("q", nrow);
		if (config.pcg_solve == 0)	{
			v = DblArr2("v", nrow, gsteps+2);
			h = DblArr2("v", gsteps+1, gsteps+2);
			s = DblArr("s", gsteps+2);	y = DblArr("y", gsteps+1);
			c1 = DblArr("c1", gsteps+1);	c2 = DblArr("c2", gsteps+1);
		}
		
		// Get ptr for the CRS matrix
		for (int idx = 0; idx < nrow+1; idx++)	{
			ii = config.i3d(idx);
			jj = config.j3d(idx);
			kk = config.k3d(idx);
			if (idx == nrow) {ptr(idx) = nnz;}
			else	{ptr(idx) = get_irow(ii, jj, kk, nx, ny, nz);}
		}
		for (int idx = 0; idx < nrow; idx++)	{
			x(idx) = 0.0;	diag(idx) = 0.0;
			r(idx) = 0.0;	z(idx) = 0.0;
			p(idx) = 0.0;	q(idx) = 0.0;
		}

	}

	// get starting index
	inline int get_irow(int i, int j, int k, int nx, int ny, int nz) {
		int nrowk=0, ncoli=0, ncol1=0, nlayer1=0, nlayerj=0, irow=0;
		if (j == 0)	{
			if (i == 0)	{
				if (nx == 1)	{if (k > 0)	{nrowk = 2 + (k-1)*3;}}	// 1D
				else	{
					if (ny == 1)	{if (k > 0)	{nrowk = 3 + (k-1)*4;}}	//2D
					else	{if (k > 0)	{nrowk = 4 + (k-1)*5;}}	//3D
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
		    	if (i == 0)	{if (k > 0)	{nrowk = 4 + (k-1)*5;}}
		    	else	{
		    		ncol1 = 4*2 + 5*(nz-2);	
					ncoli = (i-1)*(5*2 + 6*(nz-2));
					if (i == nx-1)	{if (k > 0)	{nrowk = 4 + (k-1)*5;}}
					else	{if (k > 0)	{nrowk = 5 + (k-1)*6;}}
		    	}
		    }
		    else	{
		    	if (i == 0)	{if (k > 0)	{nrowk = 5 + (k-1)*6;}}
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


};

#endif
