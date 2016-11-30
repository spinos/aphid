#ifndef CLAPACKTEMPL_H
#define CLAPACKTEMPL_H

#include <f2c.h>
#include <clapack.h>

template<typename T>
T absoluteValue(T a)
{ return a > 0 ? a : -a; }

template <typename T, typename I>
void sort(I* irOut, T* prOut,I beg, I end) {
   I i;
   if (end <= beg) return;
   I pivot=beg;
   for (i = beg+1; i<=end; ++i) {
      if (irOut[i] < irOut[pivot]) {
         if (i == pivot+1) {
            I tmp = irOut[i];
            T tmpd = prOut[i];
            irOut[i]=irOut[pivot];
            prOut[i]=prOut[pivot];
            irOut[pivot]=tmp;
            prOut[pivot]=tmpd;
         } else {
            I tmp = irOut[pivot+1];
            T tmpd = prOut[pivot+1];
            irOut[pivot+1]=irOut[pivot];
            prOut[pivot+1]=prOut[pivot];
            irOut[pivot]=irOut[i];
            prOut[pivot]=prOut[i];
            irOut[i]=tmp;
            prOut[i]=tmpd;
         }
         ++pivot;
      }
   }
   sort(irOut,prOut,beg,pivot-1);
   sort(irOut,prOut,pivot+1,end);
}

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
	
template <typename T> int clapack_sytrf(char *uplo, integer n, T *a, integer lda, 
	integer *ipiv, T *work, integer *lwork, integer *info);
	
template <typename T> int clapack_sytri(char *uplo, integer n, T *a, integer lda, 
	integer *ipiv, T *work, integer *info);

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
#endif        //  #ifndef CLAPACKTEMPL_H

