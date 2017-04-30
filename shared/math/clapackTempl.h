#ifndef CLAPACKTEMPL_H
#define CLAPACKTEMPL_H

#include <f2c.h>
#include <clapack.h>

/// res = dot(x, y)
template <typename T> T clapack_dot(integer n, T *dx, integer incx, T *dy, integer incy);

/// x <- a * x
template <typename T> int clapack_scal(integer n, T da, T *dx, integer incx);

template <typename T> void clapack_axpy(integer N,
       T alpha,  T *X,  integer ldx,
       T* Y,  integer ldy);

template <typename T> void clapack_gemv(char * trans, integer M,  integer N,
       T alpha,  T *A,  integer lda,
       T* X,  integer incX,
	   T beta, T * Y, integer incY);

/// y <- alpha * a * x + beta * y   
template <typename T> int clapack_symv(char *uplo, integer n, T alpha, 
	T *a, integer lda, T *x, integer incx, T beta, T *y, integer incy);
	   
template <typename T> int clapack_gesvd(char *jobu, char *jobvt, integer m, integer n, 
	T *a, integer lda, T *s, T *u, integer ldu, T *vt, 
	integer ldvt, T *work, integer *lwork, integer *info);
	
template <typename T> int clapack_syevr(char *jobz, char *range, char *uplo, integer n, 
	T *a, integer lda,
	T *vl, T *vu, integer il, integer iu,
	T abstol, integer *m,
	T *w, T *z__, integer ldz, integer *isuppz, T *work, 
	integer *lwork, integer *iwork, integer *liwork, integer *info);
	
template <typename T> int clapack_gemm(char *transa, char *transb, integer m, integer n, integer k, 
	T alpha, T *a, integer lda, 
	T *b, integer ldb, T beta, T *c__, integer ldc);

/// http://scc.qibebt.cas.cn/docs/library/Intel%20MKL/2011/mkl_manual/lau/functn_c_z_syr.htm
/// symmetric rank-1 update	
/// a <- alpha * x * x' + a
template <typename T> int clapack_syr(char *uplo, integer n, T alpha, 
	T *x, integer incx, T *a, integer lda);

template <typename T> int clapack_syrk(char *uplo, char *trans, integer n, integer k, 
	T alpha, T *a, integer lda, T beta, 
	T *c__, integer ldc);
	
/// http://www.math.utah.edu/software/lapack/lapack-d/dgetrf.html
/// LU factorization
template <typename T> int clapack_getrf(integer m, integer n, T *a, integer lda, 
	integer *ipiv, integer *info);
	
/// http://www.math.utah.edu/software/lapack/lapack-d/dgetri.html
/// inverse using the LU factorization computed by getrf
template <typename T> int clapack_getri(integer n, T *a, integer lda, 
	integer *ipiv, T *work, integer *lwork, integer *info);
	
/// http://physics.oregonstate.edu/~landaur/nacphy/lapack/routines/sgetrs.html
/// solve a system of linear equations  A * X = B or A' * X = B with a
/// general N-by-N matrix	A using	the LU factorization computed by SGETRF
template <typename T> int clapack_getrs(char * trans, integer n, integer nrhs,
	T * a, integer lda, integer * ipiv, T * b, integer ldb, integer *info);
	
template <typename T> int clapack_sytrf(char *uplo, integer n, T *a, integer lda, 
	integer *ipiv, T *work, integer *lwork, integer *info);
	
template <typename T> int clapack_sytri(char *uplo, integer n, T *a, integer lda, 
	integer *ipiv, T *work, integer *info);
	
/// http://www.math.utah.edu/software/lapack/lapack-d/dsyev.html
/// compute all eigenvalues and, optionally, eigenvectors of a real symmetric matrix A

template <typename T> int clapack_syev(char *jobz, char *uplo, 
									integer n, T *a, 
									integer lda, T *w,
									T *work, integer *ipiv, integer *info);

/// http://www.math.utah.edu/software/lapack/lapack-d/dpotrf.html
/// compute the Cholesky factorization of a real symmetric positive definite matrix A
/// A = U**T * U if UPLO = 'U'
/// A = L * L**T if UPLO = 'L'

template <typename T> int clapack_potrf(char *uplo, integer n, T *a, integer lda, integer *info);
		
/// http://www.math.utah.edu/software/lapack/lapack-d/dsysv.html
/// compute the solution to a real system of linear equations
/// A * X = B

template <typename T> int clapack_sysv(char *uplo, integer n, integer nrhs,
									T *a, integer lda, 
									integer *ipiv,
									T *b, integer ldb,
									T *work,
									integer *lwork,
									integer *info);
									
/// http://physics.oregonstate.edu/~landaur/nacphy/lapack/routines/spptrs.html
/// solve a system of linear equations A*X = B with a symmetric positive definite matrix A

template <typename T> int clapack_pptrs(char *uplo, integer n, integer nrhs,
									T *a, T *b, integer ldb,
									integer *info);
									
/// http://physics.oregonstate.edu/~landaur/nacphy/lapack/routines/dsytrs.html
/// solve a system of linear equations A*X = B with a real symmetric
/// matrix A using the factorization A = U*D*U**T	or A = L*D*L**T	computed by
/// SYTRF

template <typename T> int clapack_sytrs(char *uplo, integer n, integer nrhs,
									T *a, integer lda, 
									integer * ipiv,
									T *b, integer ldb,
									integer *info);
									
template <> inline double clapack_dot<double>(integer n, double *dx, integer incx, double *dy, integer incy)
{
	return ddot_(&n, dx, &incx, dy, &incy);
}

template <> inline float clapack_dot<float>(integer n, float *dx, integer incx, float *dy, integer incy)
{
	return sdot_(&n, dx, &incx, dy, &incy);
}

template <> inline int clapack_scal<double>(integer n, double da, double *dx, integer incx)
{
	return dscal_(&n, &da, dx, &incx);
}

template <> inline int clapack_scal<float>(integer n, float da, float *dx, integer incx)
{
	return sscal_(&n, &da, dx, &incx);
}

template <> inline void clapack_axpy<double>(integer N,
       double alpha,  double* X,  integer incX,
       double* Y,  integer incY) {
	 daxpy_(&N, &alpha, X, &incX, Y, &incY);
}

template <> inline void clapack_axpy<float>(integer N,
       float alpha,  float* X,  integer incX,
       float* Y,  integer incY) {
	 saxpy_(&N, &alpha, X, &incX, Y, &incY);
}

template <> inline void clapack_gemv<double>(char * trans, integer M,  integer N,
       double alpha,  double *A,  integer lda,
       double *X,  integer incX,
	   double beta, double * Y, integer incY)
{
    dgemv_(trans, &M, &N, 
           &alpha, A, &lda, 
           X, &incX, 
           &beta, Y, &incY);
}

template <> inline void clapack_gemv<float>(char * trans, integer M,  integer N,
       float alpha,  float *A,  integer lda,
       float *X,  integer incX,
	   float beta, float * Y, integer incY)
{
    sgemv_(trans, &M, &N, 
           &alpha, A, &lda, 
           X, &incX, 
           &beta, Y, &incY);
}

template <> inline int clapack_symv<double>(char *uplo, integer n, double alpha, 
	double *a, integer lda, double *x, integer incx, double beta, double *y, integer incy)
{
	return dsymv_(uplo, &n, &alpha, 
		a, &lda, x, &incx, &beta, y, &incy);
}

template <> inline int clapack_symv<float>(char *uplo, integer n, float alpha, 
	float *a, integer lda, float *x, integer incx, float beta, float *y, integer incy)
{
	return ssymv_(uplo, &n, &alpha, 
		a, &lda, x, &incx, &beta, y, &incy);
}

template <> inline int clapack_gesvd<float>(char *jobu, char *jobvt, integer m, integer n, 
	float *a, integer lda, float *s, float *u, integer ldu, float *vt, 
	integer ldvt, float *work, integer *lwork, integer *info)
{
	return sgesvd_(jobu, jobvt, &m, &n, 
		a,	&lda,	s,	u,	&ldu,	vt, 
		&ldvt,	work,	lwork,	info);
}

template <> inline int clapack_gesvd<double>(char *jobu, char *jobvt, integer m, integer n, 
	double *a, integer lda, double *s, double *u, integer ldu, double *vt, 
	integer ldvt, double *work, integer *lwork, integer *info)
{ return dgesvd_(jobu, jobvt, &m, &n, 
		a,	&lda,	s,	u,	&ldu,	vt, 
		&ldvt,	work,	lwork,	info); }
		
template <> inline int clapack_syevr<double>(char *jobz, char *range, char *uplo, integer n, 
	double *a, integer lda,
	double *vl, double *vu, integer il, integer iu,
	double abstol, integer *m,
	double *w, double *z__, integer ldz, integer *isuppz, double *work, 
	integer *lwork, integer *iwork, integer *liwork, integer *info)
{
	return dsyevr_(jobz, range, uplo, &n, 
	a, &lda, vl, vu, &il, &iu, 
	&abstol, m, 
	w, z__, &ldz, isuppz, work, 
	lwork, iwork, liwork, info);
}

template <> inline int clapack_syevr<float>(char *jobz, char *range, char *uplo, integer n, 
	float *a, integer lda, 
	float *vl, float *vu, integer il, integer iu, 
	float abstol, integer *m, 
	float *w, float *z__, integer ldz, integer *isuppz, float *work, 
	integer *lwork, integer *iwork, integer *liwork, integer *info)
{
	return ssyevr_(jobz, range, uplo, &n, 
	a, &lda, vl, vu, &il, &iu, 
	&abstol, m, 
	w, z__, &ldz, isuppz, work, 
	lwork, iwork, liwork, info);
}

template <> inline int clapack_gemm<double>(char *transa, char *transb, integer m, integer n, 
	integer k, double alpha, double *a, integer lda, 
	double *b, integer ldb, double beta, double *c__, integer ldc)
{
	return dgemm_(transa, transb, &m, &n, &k, 
	&alpha, a, &lda, 
	b, &ldb, &beta, c__, &ldc);
}

template <> inline int clapack_gemm<float>(char *transa, char *transb, integer m, integer n, 
	integer k, float alpha, float *a, integer lda, 
	float *b, integer ldb, float beta, float *c__, integer ldc)
{
	return sgemm_(transa, transb, &m, &n, &k, 
	&alpha, a, &lda, 
	b, &ldb,&beta, c__, &ldc);
}

template <> inline int clapack_syr<double>(char *uplo, integer n, double alpha, 
	double *x, integer incx, double *a, integer lda)
{
	return dsyr_(uplo, &n, &alpha, x, &incx, a, &lda);
}

template <> inline int clapack_syr<float>(char *uplo, integer n, float alpha, 
	float *x, integer incx, float *a, integer lda)
{
	return ssyr_(uplo, &n, &alpha, x, &incx, a, &lda);
}

template <> inline int clapack_syrk<double>(char *uplo, char *trans, integer n, integer k, 
	double alpha, double *a, integer lda, double beta, 
	double *c__, integer ldc)
{ 
	return dsyrk_(uplo, trans, &n, &k, 
	&alpha, a, &lda, &beta, 
	c__, &ldc);
}

template <> inline int clapack_syrk<float>(char *uplo, char *trans, integer n, integer k, 
	float alpha, float *a, integer lda, float beta, 
	float *c__, integer ldc)
{ 
	return ssyrk_(uplo, trans, &n, &k, 
	&alpha, a, &lda, &beta, 
	c__, &ldc);
}

template <> inline int clapack_getrf<double>(integer m, integer n, double *a, integer lda, 
	integer *ipiv, integer *info)
{ return dgetrf_(&m, &n, a, &lda, ipiv, info); }

template <> inline int clapack_getrf<float>(integer m, integer n, float *a, integer lda, 
	integer *ipiv, integer *info)
{ return sgetrf_(&m, &n, a, &lda, ipiv, info); }

template <> inline int clapack_sytrf<double>(char *uplo, integer n, double *a, integer lda, 
	integer *ipiv, double *work, integer *lwork, integer *info)
{
	return dsytrf_(uplo,  &n, a, &lda, 
		ipiv, work, lwork, info);
}

template <> inline int clapack_sytrf<float>(char *uplo, integer n, float *a, integer lda, 
	integer *ipiv, float *work, integer *lwork, integer *info)
{
	return ssytrf_(uplo,  &n, a, &lda, 
		ipiv, work, lwork, info);
}

template <> inline int clapack_getri<double>(integer n, double *a, integer lda, 
	integer *ipiv, double *work, integer *lwork, integer *info)
{
	return dgetri_(&n, a, &lda,
		ipiv, work, lwork, info);
}

template <> inline int clapack_getri<float>(integer n, float *a, integer lda, 
	integer *ipiv, float *work, integer *lwork, integer *info)
{
	return sgetri_(&n, a, &lda,
		ipiv, work, lwork, info);
}

template <> inline int clapack_sytri<double>(char *uplo, integer n, double *a, integer lda, 
	integer *ipiv, double *work, integer *info)
{
	return dsytri_(uplo, &n, a, &lda, ipiv, work, info);
}

template <> inline int clapack_sytri<float>(char *uplo, integer n, float *a, integer lda, 
	integer *ipiv, float *work, integer *info)
{
	return ssytri_(uplo, &n, a, &lda, ipiv, work, info);
}

template <> inline int clapack_syev<double>(char *jobz, char *uplo, 
									integer n, double *a, 
									integer lda, double *w,
									double *work, integer *ipiv, integer *info)
{
	return dsyev_(jobz, uplo, &n, a, &lda, w, work, ipiv, info);
}

template <> inline int clapack_syev<float>(char *jobz, char *uplo, 
									integer n, float *a, 
									integer lda, float *w,
									float *work, integer *ipiv, integer *info)
{
	return ssyev_(jobz, uplo, &n, a, &lda, w, work, ipiv, info);
}

template <> inline int clapack_potrf<double>(char *uplo, integer n, double *a, integer lda, integer *info)
{
	return dpotrf_(uplo, &n, a, &lda, info);
}

template <> inline int clapack_potrf<float>(char *uplo, integer n, float *a, integer lda, integer *info)
{
	return spotrf_(uplo, &n, a, &lda, info);
}

template <> inline int clapack_sysv<double>(char *uplo, integer n, integer nrhs,
									double *a, integer lda, 
									integer *ipiv,
									double *b, integer ldb,
									double *work,
									integer *lwork,
									integer *info)
{
	return dsysv_(uplo, &n, &nrhs, a, &lda, ipiv, b, &ldb, work, lwork, info);
}

template <> inline int clapack_sysv<float>(char *uplo, integer n, integer nrhs,
									float *a, integer lda, 
									integer *ipiv,
									float *b, integer ldb,
									float *work,
									integer *lwork,
									integer *info)
{
	return ssysv_(uplo, &n, &nrhs, a, &lda, ipiv, b, &ldb, work, lwork, info);
}

template <> inline int clapack_pptrs<double>(char *uplo, integer n, integer nrhs,
									double *a, double *b, integer ldb,
									integer *info)
{
	return dpptrs_(uplo, &n, &nrhs, a, b, &ldb, info);
}

template <> inline int clapack_pptrs<float>(char *uplo, integer n, integer nrhs,
									float *a, float *b, integer ldb,
									integer *info)
{
	return spptrs_(uplo, &n, &nrhs, a, b, &ldb, info);
}

template <> inline int clapack_sytrs<double>(char *uplo, integer n, integer nrhs,
									double *a, integer lda, 
									integer * ipiv,
									double *b, integer ldb,
									integer *info)
{
	return dsytrs_(uplo, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}

template <> inline int clapack_sytrs<float>(char *uplo, integer n, integer nrhs,
									float *a, integer lda, 
									integer * ipiv,
									float *b, integer ldb,
									integer *info)
{
	return ssytrs_(uplo, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}

template <> inline int clapack_getrs<double>(char * trans, integer n, integer nrhs,
		double* a, integer lda, integer * ipiv, double* b, integer ldb, integer *info)
{
	return dgetrs_(trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}

template <> inline int clapack_getrs<float>(char * trans, integer n, integer nrhs,
		float* a, integer lda, integer * ipiv, float* b, integer ldb, integer *info)
{
	return sgetrs_(trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}
#endif        //  #ifndef CLAPACKTEMPL_H

