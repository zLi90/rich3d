/*
	Solvers for the Ax=b system
*/
#ifndef _SOLVER_H_
#define _SOLVER_H_

#include "config.h"
#include "matrix.h"

template<class ExecutionSpace>
class Solver {

public:
	int iter, iter_max, gsteps, nrow, nnz;
	double eps, eps_min;
	dualDbl lodi, updi;
	// vectors for pcg
	dualDbl v, z, p, q, tmp11, tmp12, tmp13, tmp14;
	// vectors for gmres
	dualDbl s, y, c1, c2, h;
	typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value, dualDbl::memory_space, dualDbl::host_mirror_space>::type ms;
	typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value, dualInt::memory_space, dualInt::host_mirror_space>::type msi;
	
	inline void init(Matrix A, Config config)	{
		nrow = A.nrow;	nnz = A.nnz;
		iter_max = 40;
		gsteps = 40;
		eps_min = 1e-7;
		if (config.pcg_solve == 1)	{v = dualDbl("v", nrow, 1);}
		else	{v = dualDbl("v", nrow, gsteps+2);}
		z = dualDbl("z", nrow, 1);
		lodi = dualDbl("lodi", nnz, 1);	updi = dualDbl("updi", nnz, 1);
		p = dualDbl("p", nrow, 1);	q = dualDbl("q", nrow, 1);
		tmp11 = dualDbl("tmp11", nnz, 1);
    	tmp12 = dualDbl("tmp12", nnz, 1);
    	tmp13 = dualDbl("tmp13", nnz, 1);
    	tmp14 = dualDbl("tmp14", nrow, 1);
    	s = dualDbl("s", gsteps+2, 1);		y = dualDbl("y", gsteps+1, 1);
		c1 = dualDbl("c1", gsteps+1, 1);	c2 = dualDbl("c2", gsteps+1, 1);
		h = dualDbl("h", gsteps+1, gsteps+2);
	}
	
	/* 
		Synchronization
	*/	
	void sync1(dualDbl vector)	{vector.sync<dualDbl::host_mirror_space> ();}
	void synci(dualInt vector)	{vector.sync<dualInt::host_mirror_space> ();}
	
	
	/*
		----------------------------------------------------------
		----------------------------------------------------------
							Iterative Solvers
		----------------------------------------------------------
		----------------------------------------------------------
	*/
	
	/*
    	Approximate SSOR Preconditioner
    */
    void precSSOR(Matrix A, Matrix K, dualDbl res, dualDbl c, double omega)	{ 	
    	Kokkos::parallel_for(nrow, Reset(res));
    	
		Kokkos::parallel_for(nrow, Mul_MV(tmp14, K, c));
		Kokkos::parallel_for(nnz, Swap_VV(K.valT, K.val));
    	Kokkos::parallel_for(nrow, Swap_VVI(K.ptrT, K.ptr));
    	Kokkos::parallel_for(nnz, Swap_VVI(K.indT, K.ind));
    	
		Kokkos::parallel_for(nrow, Mul_MV(res, K, tmp14));
		Kokkos::parallel_for(nnz, Swap_VV(K.val, K.valT));
    	Kokkos::parallel_for(nrow, Swap_VVI(K.ptr, K.ptrT));
    	Kokkos::parallel_for(nnz, Swap_VVI(K.ind, K.indT));
    }
    
    /*
    	GMRES Solver
    */
    int gmres(Matrix A, Config config)	{ 
    	int ii, jj, kk, iter_tot = 0;
		double norm;
		// reset 
		for (ii = 0; ii < gsteps+2; ii++)	{
			reset(v, ii);
			Kokkos::parallel_for(gsteps+1, Reset(h, ii));
			config.sync(v); config.sync(h); 
		}	
		Kokkos::parallel_for(gsteps+1, Reset(s, 0));
		Kokkos::parallel_for(gsteps+1, Reset(y, 0));
		Kokkos::parallel_for(gsteps+1, Reset(c1, 0));
		Kokkos::parallel_for(gsteps+1, Reset(c2, 0));
		config.sync(s); config.sync(y); config.sync(c1); config.sync(c2);
		
		// initialize residual
		Kokkos::parallel_for(nrow, Residual(v, A, A.x, A.rhs));
		config.sync(v);
		iter = 0;	eps = 1.0;
		while (iter < iter_max & eps > eps_min)	{
			//precILU(A, v, v);	sync(v);
			Kokkos::parallel_reduce(nrow, Mul_VV(v, v, 0, 0), norm);
			Kokkos::parallel_for(1, Update_V(s, pow(norm,0.5), 0, 0));
			Kokkos::parallel_for(nrow, KXPY(v, 0, 1/pow(norm,0.5), v, 0, 0.0, v, 0));
			config.sync(v);	config.sync(s);
			ii = 0; eps = 1.0;
			while (ii < gsteps & eps > eps_min)	{
				Kokkos::parallel_for(nrow, Mul_MV(v, ii+1, A, v, ii));
				//config.sync(v);
				for (kk = 0; kk < ii+1; kk++)	{
					Kokkos::parallel_reduce(nrow, Mul_VV(v, v, kk, ii+1), norm);
					Kokkos::parallel_for(1, Update_V(h, norm, ii, kk));
					Kokkos::parallel_for(nrow, KXPY(v, ii+1, -norm, v, kk, 1.0, v, ii+1));
					//Kokkos::parallel_for(nrow, KXPY(v, kk, norm, v, ii+1, v, kk)); 
					
				}
				//config.sync(v);	config.sync(h);
				Kokkos::parallel_reduce(nrow, Mul_VV(v, v, ii+1, ii+1), norm);
				Kokkos::parallel_for(1, Update_V(h, pow(norm, 0.5), ii, ii+1));
				Kokkos::parallel_for(nrow, KXPY(v, ii+1, 1.0/pow(norm, 0.5), v, ii+1, 0.0, v, ii+1));
				config.sync(v);
				Kokkos::parallel_for(1, Update_S(h, s, c1, c2, ii)); 
				config.sync(s);	config.sync(h);	config.sync(c1);	config.sync(c2);
				ii += 1;
				iter_tot += 1;
				dualDbl::t_host h_s = s.h_view;
				eps = fabs(h_s(ii,0));
			}
			Kokkos::parallel_for(1, Update_Y(h, s, y, ii));
			config.sync(s);	config.sync(y);
			// update solution
			dualDbl::t_host h_y = y.h_view;
			for (jj = ii-1; jj >= 0; jj--)	{
				Kokkos::parallel_for(nrow, Update_X(A.x, h_y(jj,0), v, jj)); 
			}
			A.x.sync<dualDbl::host_mirror_space> (); 
			// update residual
			Kokkos::parallel_for(nrow, Residual(v, A, A.x, A.rhs));
			config.sync(v);	
			Kokkos::parallel_reduce(nrow, Mul_VV(v, v, 0, 0), norm); 
			eps = pow(norm, 0.5);
			iter += 1;
		}
    	printf("        >>> GMRES completed with iter=%d, eps=%f\n",iter_tot,eps);
    	return iter_tot;
    }
    
    
    /*
    	CG Solver
    */
    void cg(Matrix A, Matrix K, Config config)	{ 
		double norm, rho, rhoOld, deno, alpha, beta, omega = 1.0;
		rhoOld = 0.0;
		// initialize residual
		//Kokkos::parallel_for(nrow, Copy_VVI(A.ptr, K.ptr));
    	//Kokkos::parallel_for(nnz, Copy_VVI(A.ind, K.ind));
    	
		Kokkos::parallel_for(nrow, Residual(v, A, A.x, A.rhs));
		config.sync(v);
		
		/*config.printi_view(A.ptr, 10, 0);
		config.printi_view(A.ind, 10, 0);
		config.print_view(A.val, 10, 0);
		config.print_view(A.rhs, 10, 0);
		 config.print_view(v, 10, 0);*/
		iter = 0;	eps = 1.0;
		// preconditioning
		/*Kokkos::parallel_for(nrow, ImLD(tmp12, A, omega));	sync1(tmp12);
    	Kokkos::parallel_for(nnz, Scal_M(tmp11, A.diag, omega));	sync1(tmp11);
    	Kokkos::parallel_for(nrow, Mul_MM(tmp13, A.ptr, A.ind, tmp11, tmp12, omega));
    	sync1(tmp13);
    	Kokkos::parallel_for(nrow, Copy_VVI(A.ptr, K.ptr));
    	Kokkos::parallel_for(nnz, Copy_VVI(A.ind, K.ind));
    	Kokkos::parallel_for(nnz, Copy_VV(tmp13, K.val));
    	synci(K.ptr);	synci(K.ind);	sync1(K.val);
		K.transpose();*/
		
		while (iter < iter_max & eps > eps_min)	{
			//precSSOR(A, K, z, v, omega);
			Kokkos::parallel_for(nrow, Copy_VV(v, 0, z, 0));
			config.sync(z);
    		Kokkos::parallel_reduce(nrow, Mul_VV(v, z, 0, 0), rho);
    		if (iter == 0)	{
    			beta = 0.0;
    			Kokkos::parallel_for(nrow, Copy_VV(z, 0, p, 0));
			}
    		else 	{
    			beta = rho / rhoOld;
    			Kokkos::parallel_for(nrow, KXPY(p, 0, beta, p, 0, 1.0, z, 0));
    		}
    		config.sync(p);
    		Kokkos::parallel_for(nrow, Mul_MV(q, 0, A, p, 0));
    		config.sync(q);
    		Kokkos::parallel_reduce(nrow, Mul_VV(q, p, 0, 0), deno);
    		alpha = rho / deno;
    		Kokkos::parallel_for(nrow, Update_X(A.x, alpha, p, 0));
    		config.sync(A.x);
    		//A.x.sync<dualDbl::host_mirror_space> ();
    		Kokkos::parallel_for(nrow, KXPY(v, 0, -alpha, q, 0, 1.0, v, 0));
    		config.sync(v);
    		rhoOld = rho;
    		Kokkos::parallel_reduce(nrow, Mul_VV(v, v, 0, 0), norm);
			eps = pow(norm, 0.5);
    		iter += 1;
    	}
    	
    	printf("       >> CG completed with iter=%d, eps=%f\n",iter,eps);
    }
    
    /*
		----------------------------------------------------------
		----------------------------------------------------------
					    Matrix-Vector Operators
		----------------------------------------------------------
		----------------------------------------------------------
	*/
	
	/*
		Update one element in the view
	*/
    struct Update_V {
		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> res;
		double s;
		int ii, jj;
   		Update_V(dualDbl d_res, double coef, int row, int col)	{
   			res = d_res.template view<ms> ();	d_res.sync<ms> ();	d_res.modify<ms> ();
   			s = coef;	ii = row; jj = col;
   		}
    	KOKKOS_INLINE_FUNCTION	void operator() (const int idx) const {res(ii,jj) = s;}
    };
    
    /*
    	Basic Matrix-Vector Operations
    */
    
    // Vector-Vector Multiplication
    struct Mul_VV {
   		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> vec1;
   		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> vec2;
   		int j1, j2;
   		Mul_VV(dualDbl d_vec1, dualDbl d_vec2, int col1, int col2)	{
   			vec1 = d_vec1.template view<ms> ();	d_vec1.sync<ms> ();
   			vec2 = d_vec2.template view<ms> ();	d_vec2.sync<ms> ();
   			j1 = col1;	j2 = col2;
   		}
   		KOKKOS_INLINE_FUNCTION void
		operator() (const int idx, double& out) const	
		{out += vec1(idx,j1) * vec2(idx,j2);}

		KOKKOS_INLINE_FUNCTION void
		join (volatile double& dst, const volatile double& src) const	{dst += src;}
    
    	KOKKOS_INLINE_FUNCTION void	init (double& dst) const	{dst = 0.0;}
    };
    
    // Matrix-Vector Multiplication
    struct Mul_MV {
   		Kokkos::View<dualInt::scalar_array_type, dualInt::array_layout, msi> ptr;
   		Kokkos::View<dualInt::scalar_array_type, dualInt::array_layout, msi> ind;
   		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> val;
   		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> vec;
   		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> res;
   		int j1, j2;
   		Mul_MV(dualDbl d_res, int col1, Matrix A, dualDbl d_vec, int col2)	{
   			dualInt d_ptr = A.ptr;	dualInt d_ind = A.ind;	dualDbl d_val = A.val;
   			ptr = d_ptr.template view<msi> ();	d_ptr.sync<msi> ();
   			ind = d_ind.template view<msi> ();	d_ind.sync<msi> ();
   			val = d_val.template view<ms> ();	d_val.sync<ms> ();
   			vec = d_vec.template view<ms> ();	d_vec.sync<ms> ();
   			res = d_res.template view<ms> ();	d_res.sync<ms> ();	d_res.modify<ms> ();
   			j1 = col1;	j2 = col2;
   		}
   		
   		KOKKOS_INLINE_FUNCTION	void operator() (const int idx) const {
    		int icol;
    		res(idx,j1) = 0.0;
    		for (icol = ptr(idx,0); icol < ptr(idx+1,0); icol++)	{
				res(idx,j1) += val(icol,0) * vec(ind(icol,0),j2);
			}
    	}
    };
    
    // Scalar * Vector + Vector
    struct KXPY {
        Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> res;
   		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> vec1;
   		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> vec2;
   		double s1, s2;
   		int j1, j2, j3;
   		KXPY(dualDbl d_res, int col1, double c1, dualDbl d_vec1, int col2, double c2, dualDbl d_vec2, int col3)	{
   			res = d_res.template view<ms> ();	d_res.sync<ms> ();	d_res.modify<ms> ();
   			vec1 = d_vec1.template view<ms> ();	d_vec1.sync<ms> ();
   			vec2 = d_vec2.template view<ms> ();	d_vec2.sync<ms> ();
   			s1 = c1; s2 = c2;	j1 = col1;	j2 = col2;	j3 = col3;
   		}
    	KOKKOS_INLINE_FUNCTION	void operator() (const int idx) const {
    		res(idx,j1) = s1 * vec1(idx,j2) + s2 * vec2(idx,j3);
    	}
    };
	
	// Residual = Matrix * Vector + Vector
	struct Residual {
   		Kokkos::View<dualInt::scalar_array_type, dualInt::array_layout, msi> ptr;
   		Kokkos::View<dualInt::scalar_array_type, dualInt::array_layout, msi> ind;
   		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> val;
   		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> x;
   		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> y;
   		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> res;

   		Residual(dualDbl d_res, Matrix A, dualDbl d_x, dualDbl d_y)	{
   			dualInt d_ptr = A.ptr;	dualInt d_ind = A.ind;	dualDbl d_val = A.val;
   			ptr = d_ptr.template view<msi> ();	d_ptr.sync<msi> ();
   			ind = d_ind.template view<msi> ();	d_ind.sync<msi> ();
   			val = d_val.template view<ms> ();	d_val.sync<ms> ();
   			x = d_x.template view<ms> ();	d_x.sync<ms> ();
   			y = d_y.template view<ms> ();	d_y.sync<ms> ();
   			res = d_res.template view<ms> ();	d_res.sync<ms> ();	d_res.modify<ms> ();
   		}
   		
   		KOKKOS_INLINE_FUNCTION	void operator() (const int idx) const {
    		int icol;
    		res(idx,0) = y(idx,0);
    		for (icol = ptr(idx,0); icol < ptr(idx+1,0); icol++)	{
				res(idx,0) -= val(icol,0) * x(ind(icol,0),0);
			}
    	}
    };
    
    // Update solution A.x
    struct Update_X {
   		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> res;
   		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> vec;
   		double s;
   		int jj;
   		Update_X(dualDbl d_res, double coef, dualDbl d_vec, int col)	{
   			res = d_res.template view<ms> ();	d_res.sync<ms> ();	d_res.modify<ms> ();
   			vec = d_vec.template view<ms> ();	d_vec.sync<ms> ();
   			s = coef;	jj = col;
   		}
    	KOKKOS_INLINE_FUNCTION	void operator() (const int idx) const {
    		res(idx,0) += s * vec(idx,jj);
    	}
    };
    
    
   
    struct ImLD {
    	Kokkos::View<dualInt::scalar_array_type, dualInt::array_layout, msi> ptr;
    	Kokkos::View<dualInt::scalar_array_type, dualInt::array_layout, msi> ind;
        Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> lotr;
   		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> diag;
		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> res;
   		double w;
   		ImLD(dualDbl d_res, Matrix A, double omega)	{
   			dualInt d_ptr = A.ptr;	dualInt d_ind = A.ind;
   			dualDbl d_lotr = A.diag;	dualDbl d_diag = A.diagN;
   			ptr = d_ptr.template view<msi> ();	d_ptr.sync<msi> ();	
   			ind = d_ind.template view<msi> ();	d_ind.sync<msi> ();	
   			lotr = d_lotr.template view<ms> ();	d_lotr.sync<ms> ();
   			diag = d_diag.template view<ms> ();	d_diag.sync<ms> ();
   			res = d_res.template view<ms> ();	d_res.sync<ms> ();	d_res.modify<ms> ();
			w = omega;
   		}
    	KOKKOS_INLINE_FUNCTION	void operator() (const int idx) const {
    		for (int icol = ptr(idx,0); icol < ptr(idx+1,0); icol++)	{
    			res(icol,0) = - lotr(icol,1) * w / diag(ind(icol,0),0);
    			if (ind(icol,0) == idx)	{res(icol,0) += 1.0;}
    		}
    	}
    };
    
    struct Scal_M	{
    	Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> res;
    	Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> vec;
    	double w;
    	Scal_M(dualDbl d_res, dualDbl d_vec,double omega)	{
    		vec = d_vec.template view<ms> ();	d_vec.sync<ms> ();
    		res = d_res.template view<ms> ();	d_res.sync<ms> ();	d_res.modify<ms> ();
    		w = omega;
    	}
    	KOKKOS_INLINE_FUNCTION	void operator() (const int idx) const {
    		res(idx,0) = 0.0;
    		if (vec(idx,0) != 0.0)	{
    			res(idx,0) = pow(2.0-w, 0.5) * pow(vec(idx,0)/w, -0.5);
    		}
    	}
    };
    
    struct Mul_MM {
    	Kokkos::View<dualInt::scalar_array_type, dualInt::array_layout, msi> ptr;
    	Kokkos::View<dualInt::scalar_array_type, dualInt::array_layout, msi> ind;
        Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> m1;
   		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> m2;
		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> res;
   		double w;
   		Mul_MM(dualDbl d_res, dualInt d_ptr, dualInt d_ind, dualDbl d_m1, dualDbl d_m2, double omega)	{
   			ptr = d_ptr.template view<msi> ();	d_ptr.sync<msi> ();	
   			ind = d_ind.template view<msi> ();	d_ind.sync<msi> ();	
   			m1 = d_m1.template view<ms> ();	d_m1.sync<ms> ();
   			m2 = d_m2.template view<ms> ();	d_m2.sync<ms> ();
   			res = d_res.template view<ms> ();	d_res.sync<ms> ();	d_res.modify<ms> ();
			w = omega;
   		}
    	KOKKOS_INLINE_FUNCTION	void operator() (const int idx) const {
    		double diag = 0.0;
    		for (int jj = ptr(idx,0); jj < ptr(idx+1,0); jj++)	{
    			if (idx == ind(jj,0))	{diag = m1(jj,0);}
    		}
    		for (int jj = ptr(idx,0); jj < ptr(idx+1,0); jj++)	{
    			res(jj,0) = m2(jj,0) * diag;
    		}
    	}
    };
    
    struct Mul_MMT	{
    	Kokkos::View<dualInt::scalar_array_type, dualInt::array_layout, msi> ptr;
    	Kokkos::View<dualInt::scalar_array_type, dualInt::array_layout, msi> ind;
        Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> vec1;
   		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> vec2;
		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> res;
   		Mul_MMT(dualDbl d_res, dualInt d_ptr, dualInt d_ind, dualDbl d_vec1, dualDbl d_vec2)	{
   			ptr = d_ptr.template view<msi> ();	d_ptr.sync<msi> ();	
   			ind = d_ind.template view<msi> ();	d_ind.sync<msi> ();	
   			vec1 = d_vec1.template view<ms> ();	d_vec1.sync<ms> ();
   			vec2 = d_vec2.template view<ms> ();	d_vec2.sync<ms> ();
   			res = d_res.template view<ms> ();	d_res.sync<ms> ();	d_res.modify<ms> ();
   		}
    	KOKKOS_INLINE_FUNCTION	void operator() (const int idx) const {
    		for (int ii = ptr(idx,0); ii < ptr(idx+1,0); ii++)	{
    			res(idx,0) += vec1(ii,0) * vec2(ii,0);
    		}
    	}
    };
   
    // Copy view
    struct Copy_VV {
   		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> v1;
   		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> v2;
   		int ii, jj;
   		Copy_VV(dualDbl d_v1, int col1, dualDbl d_v2, int col2)	{
   			v1 = d_v1.template view<ms> ();	d_v1.sync<ms> ();
   			v2 = d_v2.template view<ms> ();	d_v2.sync<ms> ();	d_v2.modify<ms> ();
   			ii = col1;	jj = col2;
   		}
   		KOKKOS_INLINE_FUNCTION	void operator() (const int idx) const {v2(idx,jj) = v1(idx,ii);}
    };
    struct Copy_VVI {
   		Kokkos::View<dualInt::scalar_array_type, dualInt::array_layout, msi> v1;
   		Kokkos::View<dualInt::scalar_array_type, dualInt::array_layout, msi> v2;
   		Copy_VVI(dualInt d_v1, dualInt d_v2)	{
   			v1 = d_v1.template view<msi> ();	d_v1.sync<msi> ();
   			v2 = d_v2.template view<msi> ();	d_v2.sync<msi> ();	d_v2.modify<msi> ();
   		}
   		KOKKOS_INLINE_FUNCTION	void operator() (const int idx) const {v2(idx,0) = v1(idx,0);}
    };
    void inline copy(dualDbl v1, int col1, dualDbl v2, int col2)	
    {Kokkos::parallel_for(nrow, Copy_VV(v1, col1, v2, col2));}
    
    // Swap view
    struct Swap_VV {
   		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> v1;
   		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> v2;
   		Swap_VV(dualDbl d_v1, dualDbl d_v2)	{
   			v1 = d_v1.template view<ms> ();	d_v1.sync<ms> ();	d_v1.modify<ms> ();
   			v2 = d_v2.template view<ms> ();	d_v2.sync<ms> ();	d_v2.modify<ms> ();
   		}
   		KOKKOS_INLINE_FUNCTION	void operator() (const int idx) const {
   			double tmp;
   			tmp = v2(idx,0);	v2(idx,0) = v1(idx,0);	v1(idx,0) = tmp;
		}
    };
    struct Swap_VVI {
   		Kokkos::View<dualInt::scalar_array_type, dualInt::array_layout, msi> v1;
   		Kokkos::View<dualInt::scalar_array_type, dualInt::array_layout, msi> v2;
   		Swap_VVI(dualInt d_v1, dualInt d_v2)	{
   			v1 = d_v1.template view<msi> ();	d_v1.sync<msi> ();	d_v1.modify<msi> ();
   			v2 = d_v2.template view<msi> ();	d_v2.sync<msi> ();	d_v2.modify<msi> ();
   		}
   		KOKKOS_INLINE_FUNCTION	void operator() (const int idx) const {
   			int tmp;
   			tmp = v2(idx,0);	v2(idx,0) = v1(idx,0);	v1(idx,0) = tmp;
		}
    };
    
    // Reset view
    struct Reset	{
    	Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> res;
    	int icol;
    	Reset(dualDbl d_res, int col)	{
    		res = d_res.template view<ms> ();	d_res.sync<ms> ();	d_res.modify<ms> ();
    		icol = col;
    	}
    	KOKKOS_INLINE_FUNCTION	void operator() (const int idx) const {res(idx,icol) = 0.0;}
    };
    struct ResetI	{
    	Kokkos::View<dualInt::scalar_array_type, dualInt::array_layout, msi> res;
    	ResetI(dualInt d_res)	{
    		res = d_res.template view<msi> ();	d_res.sync<msi> ();	d_res.modify<msi> ();
    	}
    	KOKKOS_INLINE_FUNCTION	void operator() (const int idx) const {res(idx,0) = 0;}
    };
    void inline reset(dualDbl v, int col)	
    {Kokkos::parallel_for(nrow, Reset(v, col));	}
    
    
    // Functions specifically for GMRES
    struct Update_S {
        Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> h_h;
   		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> h_s;
   		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> h_c1;
   		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> h_c2;
   		int ii;
   		Update_S(dualDbl d_h, dualDbl d_s, dualDbl d_c1, dualDbl d_c2, int icol)	{
   			h_h = d_h.template view<ms> ();	d_h.sync<ms> ();	d_h.modify<ms> ();
   			h_s = d_s.template view<ms> ();	d_s.sync<ms> ();	d_s.modify<ms> ();
   			h_c1 = d_c1.template view<ms> ();	d_c1.sync<ms> ();	d_c1.modify<ms> ();
   			h_c2 = d_c2.template view<ms> ();	d_c2.sync<ms> ();	d_c2.modify<ms> ();
			ii = icol;
   		}
    	KOKKOS_INLINE_FUNCTION	void operator() (const int idx) const {
    		int kk;
    		double r, h1, h2;
			for (kk = 0; kk < ii; kk++)	{
				h1 = h_c1(kk,0)*h_h(ii,kk) + h_c2(kk,0)*h_h(ii,kk+1);
				h2 = -h_c2(kk,0)*h_h(ii,kk) + h_c1(kk,0)*h_h(ii,kk+1);
				h_h(ii,kk) = h1;
				h_h(ii,kk+1) = h2;
			}
			r = pow(h_h(ii,ii)*h_h(ii,ii)+h_h(ii,ii+1)*h_h(ii,ii+1), 0.5);
			h_c1(ii,0) = h_h(ii,ii) / r;
			h_c2(ii,0) = h_h(ii,ii+1) / r;
			h_h(ii,ii) = r;
			h_h(ii,ii+1) = 0.0;
			h_s(ii+1,0) = -h_c2(ii,0)*h_s(ii,0);
			h_s(ii,0) = h_c1(ii,0)*h_s(ii,0);
    	}
    };
    
    struct Update_Y {
        Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> h_h;
   		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> h_s;
   		Kokkos::View<dualDbl::scalar_array_type, dualDbl::array_layout, ms> h_y;
   		int ii;
   		Update_Y(dualDbl d_h, dualDbl d_s, dualDbl d_y, int icol)	{
   			h_h = d_h.template view<ms> ();	d_h.sync<ms> ();	
   			h_s = d_s.template view<ms> ();	d_s.sync<ms> ();	d_s.modify<ms> ();
   			h_y = d_y.template view<ms> ();	d_y.sync<ms> ();	d_y.modify<ms> ();
			ii = icol;
   		}
    
    	KOKKOS_INLINE_FUNCTION	void operator() (const int idx) const {
    		int jj, kk;
    		for (jj = ii-1; jj >= 0; jj--)	{
				h_y(jj,0) = h_s(jj,0) / h_h(jj,jj);
				for (kk = jj-1; kk > 0; kk--)	{
					h_s(kk,0) -= h_h(jj,kk)*h_y(jj,0);
				}
			}
    	}
    };
    
    
    
    	
	
};
#endif
