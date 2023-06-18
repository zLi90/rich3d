/*
	Solvers for the Ax=b system
*/
#ifndef _GW_SOLVER_H_
#define _GW_SOLVER_H_

#include "config.h"
#include "GwMatrix.h"

class GwSolver {

public:
	int iter, iter_max, gsteps, nrow, nnz;
	double eps, eps_min;

	inline void init(GwMatrix &A, Config &config)	{
		nrow = A.nrow;	nnz = A.nnz;
		iter_max = 40;
		gsteps = A.gsteps;
		if (config.scheme < 4)	{iter_max = iter_max * gsteps;}
		eps_min = 1e-7;
	}

	/*
		----------------------------------------------------------
		----------------------------------------------------------
							Iterative Solvers
		----------------------------------------------------------
		----------------------------------------------------------
	*/

	/*
    	CG Solver
    */
    int cg(GwMatrix &A, Config &config)	{
		double rho, rhoOld, alpha, beta;
		rhoOld = 0.0;
		decompose(A);
		// initialize residual
		get_residual(A);
		iter = 0;	eps = 1.0;
		while (iter < iter_max & eps > eps_min)	{
			precJACO(A);
			rho = dot(A.r, A.z, A.nrow);
			if (iter == 0)	{
				beta = 0.0;
				Kokkos::deep_copy(A.p, A.z);
			}
			else {
				beta = rho / rhoOld;
				kxpy(A.p, beta, A.p, A.z, A.nrow);
			}
			mul_MV(A.q, A, A.p);
			alpha = rho / dot(A.q, A.p, A.nrow);
			update_X(A, alpha);
			kxpy(A.r, -alpha, A.q, A.r, A.nrow);
			rhoOld = rho;
			eps = pow(dot(A.r, A.r, A.nrow),0.5);
			iter += 1;
    	}
    	printf("  >> CG solver converges in %d iterations! eps = %f\n",iter,eps);
    	return iter;
    }
    
    /*
    	GMRES Solver
    */
    int gmres(GwMatrix &A, Config &config)	{
    	int ii, iter_tot = 0;
    	reset2(A.v, A.nrow, gsteps+2);
    	reset2(A.h, gsteps+1, gsteps+2);
    	reset(A.s, gsteps+1);
    	reset(A.y, gsteps+1);
    	reset(A.c1, gsteps+1);
    	reset(A.c2, gsteps+1);
   
    	decompose(A);
    	get_residual(A);
    	copy_residual(A.v, A.r, A.nrow);
    	iter = 0;	eps = 1.0;
    	while (iter < iter_max & eps > eps_min)	{
			precJACO2(A,0);
			A.s(0) = pow(dot2(A.v, A.nrow, 0, 0), 0.5);
			kx2(A.v, 0, 1/A.s(0), A.v, 0, A.nrow);
			ii = 0; eps = 1.0;
			while (ii < gsteps & eps > eps_min)	{
				mul_MV2(A.v, ii+1, A, A.v, ii);
				precJACO2(A,ii+1);
				for (int kk = 0; kk < ii+1; kk++)	{
					A.h(ii,kk) = dot2(A.v, A.nrow, kk, ii+1);
					kxpy2(A.v, ii+1, -A.h(ii,kk), A.v, kk, A.v, ii+1, A.nrow);
				}
				A.h(ii,ii+1) = pow(dot2(A.v, A.nrow, ii+1, ii+1), 0.5);
				kx2(A.v, ii+1, 1.0/A.h(ii,ii+1), A.v, ii+1, A.nrow);
				update_S(A, ii);
				ii += 1;
				iter_tot += 1;
				eps = fabs(A.s(ii));
			}
			update_Y(A, ii);
			for (int jj = ii-1; jj >= 0; jj--)	{
				update_X2(A, A.y(jj), jj);
			}
			get_residual(A);
			copy_residual(A.v, A.r, A.nrow);
			eps = pow(dot2(A.v, A.nrow, 0, 0), 0.5);
			iter_tot += 1;
    	}
    	printf("  >> GMRES solver converges in %d iterations! eps = %f\n",iter_tot,eps);
    	return iter_tot;
    }

	/*
    	Jacobi Preconditioner
    */
    void precJACO(GwMatrix A)	{
		Kokkos::parallel_for( A.nrow , KOKKOS_LAMBDA(int idom) {
			A.z(idom) = A.r(idom) / A.diag(idom);
		});
    }
    void precJACO2(GwMatrix A, int col)	{
		Kokkos::parallel_for( A.nrow , KOKKOS_LAMBDA(int idom) {
			A.v(idom, col) = A.v(idom, col) / A.diag(idom);
		});
    }

	/*
		Reset A.x
	*/
	void reset(DblArr x, int n)	{
		Kokkos::parallel_for(n , KOKKOS_LAMBDA(int idom) {x(idom) = 0.0;});
	}
	void reset2(DblArr2 x, int n1, int n2)	{
		for (int ii = 0; ii < n2; ii++)	{
			Kokkos::parallel_for(n1 , KOKKOS_LAMBDA(int idom) {x(idom,ii) = 0.0;});
		}
		
	}

	/*
		Get residual
	*/
	void get_residual(GwMatrix &A)	{
		Kokkos::parallel_for( A.nrow , KOKKOS_LAMBDA(int idom) {
			int icol;
			A.r(idom) = A.rhs(idom);
			for (icol = A.ptr(idom); icol < A.ptr(idom+1); icol++)	{
				A.r(idom) -= A.val(icol) * A.x(A.ind(icol));
			}
		});
	}
	void copy_residual(DblArr2 v, DblArr r, int n)	{
		Kokkos::parallel_for( n , KOKKOS_LAMBDA(int idx) {
			v(idx,0) = r(idx);
		});
	}

	/*
		Dot product of two vectors
	*/
	double dot(DblArr v1, DblArr v2, int n)	{
		double out;
		Kokkos::parallel_reduce( n , KOKKOS_LAMBDA (int idx, double &prod) {
			prod += v1(idx) * v2(idx);
		} , out);
		return out;
	}
	double dot2(DblArr2 v, int n, int i, int j)	{
		double out;
		Kokkos::parallel_reduce( n , KOKKOS_LAMBDA (int idx, double &prod) {
			prod += v(idx,i) * v(idx,j);
		} , out);
		return out;
	}
	
	/*
		kX
	*/
	void kx2(DblArr2 out, int col1, double k, DblArr2 x, int col2, int n)	{
		Kokkos::parallel_for( n , KOKKOS_LAMBDA(int idx) {
			out(idx,col1) = k * x(idx,col2);
		});
	}

	/*
		kX+Y
	*/
	void kxpy(DblArr out, double k, DblArr x, DblArr y, int n)	{
		Kokkos::parallel_for( n , KOKKOS_LAMBDA(int idx) {
			out(idx) = k * x(idx) + y(idx);
		});
	}
	void kxpy2(DblArr2 out, int col1, double k, DblArr2 x, int col2, DblArr2 y, int col3, int n)	{
		Kokkos::parallel_for( n , KOKKOS_LAMBDA(int idx) {
			out(idx,col1) = k * x(idx,col2) + y(idx,col3);
		});
	}

	/*
		Matrix - Vector Multiplication
	*/
	void mul_MV(DblArr out, GwMatrix A, DblArr x)	{
		Kokkos::parallel_for( A.nrow , KOKKOS_LAMBDA(int idx) {
			int icol;
			out(idx) = 0.0;
			for (icol = A.ptr(idx); icol < A.ptr(idx+1); icol++)	{
				out(idx) += A.val(icol) * x(A.ind(icol));
			}
		});
	}
	void mul_MV2(DblArr2 out, int col1, GwMatrix A, DblArr2 x, int col2)	{
		Kokkos::parallel_for( A.nrow , KOKKOS_LAMBDA(int idx) {
			int icol;
			out(idx,col1) = 0.0;
			for (icol = A.ptr(idx); icol < A.ptr(idx+1); icol++)	{
				out(idx,col1) += A.val(icol) * x(A.ind(icol),col2);
			}
		});
	}
	
	/*
		Update s and y for GMRES
	*/
	void update_S(GwMatrix A, int col)	{
		double r;
		/*Kokkos::parallel_for( col , KOKKOS_LAMBDA(int kk) {
			double h1, h2;
			h1 = A.c1(kk)*A.h(col,kk) + A.c2(kk)*A.h(col,kk+1);
			h2 = -A.c2(kk)*A.h(col,kk) + A.c1(kk)*A.h(col,kk+1);
			A.h(col,kk) = h1;
			A.h(col,kk+1) = h2;
		});*/
		double h1, h2;
		for (int kk = 0; kk < col; kk++)	{
			h1 = A.c1(kk)*A.h(col,kk) + A.c2(kk)*A.h(col,kk+1);
			h2 = -A.c2(kk)*A.h(col,kk) + A.c1(kk)*A.h(col,kk+1);
			A.h(col,kk) = h1;
			A.h(col,kk+1) = h2;
		}
		r = pow(A.h(col,col)*A.h(col,col) + A.h(col,col+1)*A.h(col,col+1), 0.5);
		A.c1(col) = A.h(col,col) / r;
		A.c2(col) = A.h(col,col+1) / r;
		A.h(col,col) = r;
		A.h(col,col+1) = 0.0;
		A.s(col+1) = -A.c2(col)*A.s(col);
		A.s(col) = A.c1(col)*A.s(col);
	}
	void update_Y(GwMatrix A, int col)	{
		int jj, kk;
		for (jj = col-1; jj >= 0; jj--)	{
			A.y(jj) = A.s(jj) / A.h(jj,jj);
			for (kk = jj-1; kk > 0; kk--)	{
				A.s(kk) -= A.h(jj,kk)*A.y(jj);
			}
		}
	}

	/*
		Update solution
	*/
	void update_X(GwMatrix A, double alpha)	{
		Kokkos::parallel_for( A.nrow , KOKKOS_LAMBDA(int idx) {
			A.x(idx) += alpha * A.p(idx);
		});
	}
	void update_X2(GwMatrix A, double alpha, int col)	{
		Kokkos::parallel_for( A.nrow , KOKKOS_LAMBDA(int idx) {
			A.x(idx) += alpha * A.v(idx, col);
		});
	}

	/*
		Get diagonal of Matrix
	*/
	void decompose(GwMatrix A)	{
		Kokkos::parallel_for(A.nrow , KOKKOS_LAMBDA(int idx) {
			int icol;
			A.diag(idx) = 0.0;
			for (icol = A.ptr(idx); icol < A.ptr(idx+1); icol++)	{
				if (A.ind(icol) == idx)	{A.diag(idx) = A.val(icol);}
			}
		});
	}

};
#endif
