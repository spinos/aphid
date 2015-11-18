#include <f2c.h>
#include <clapack.h>

template <typename T> void clapack_axpy(integer N,
       T alpha,  T *X,  integer ldx,
       T* Y,  integer ldy);

template <typename T> void clapack_gemv(char * trans, integer M,  integer N,
       T alpha,  T *A,  integer lda,
       T* X,  integer incX,
	   T beta, T * Y, integer incY);
	   
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

template <> void clapack_axpy<double>(integer N,
       double alpha,  double* X,  integer incX,
       double* Y,  integer incY) {
	 daxpy_(&N, &alpha, X, &incX, Y, &incY);
}

template <> void clapack_axpy<float>(integer N,
       float alpha,  float* X,  integer incX,
       float* Y,  integer incY) {
	 saxpy_(&N, &alpha, X, &incX, Y, &incY);
}

template <> void clapack_gemv<double>(char * trans, integer M,  integer N,
       double alpha,  double *A,  integer lda,
       double *X,  integer incX,
	   double beta, double * Y, integer incY)
{
    dgemv_(trans, &M, &N, 
           &alpha, A, &lda, 
           X, &incX, 
           &beta, Y, &incY);
}

template <> void clapack_gemv<float>(char * trans, integer M,  integer N,
       float alpha,  float *A,  integer lda,
       float *X,  integer incX,
	   float beta, float * Y, integer incY)
{
    sgemv_(trans, &M, &N, 
           &alpha, A, &lda, 
           X, &incX, 
           &beta, Y, &incY);
}

template <> int clapack_gesvd<float>(char *jobu, char *jobvt, integer m, integer n, 
	float *a, integer lda, float *s, float *u, integer ldu, float *vt, 
	integer ldvt, float *work, integer *lwork, integer *info)
{
	return sgesvd_(jobu, jobvt, &m, &n, 
		a,	&lda,	s,	u,	&ldu,	vt, 
		&ldvt,	work,	lwork,	info);
}

template <> int clapack_gesvd<double>(char *jobu, char *jobvt, integer m, integer n, 
	double *a, integer lda, double *s, double *u, integer ldu, double *vt, 
	integer ldvt, double *work, integer *lwork, integer *info)
{ return dgesvd_(jobu, jobvt, &m, &n, 
		a,	&lda,	s,	u,	&ldu,	vt, 
		&ldvt,	work,	lwork,	info); }
		
template <> int clapack_syevr<double>(char *jobz, char *range, char *uplo, integer n, 
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

template <> int clapack_syevr<float>(char *jobz, char *range, char *uplo, integer n, 
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

template <> int clapack_gemm<double>(char *transa, char *transb, integer m, integer n, 
	integer k, double alpha, double *a, integer lda, 
	double *b, integer ldb, double beta, double *c__, integer ldc)
{
	return dgemm_(transa, transb, &m, &n, &k, 
	&alpha, a, &lda, 
	b, &ldb, &beta, c__, &ldc);
}

template <> int clapack_gemm<float>(char *transa, char *transb, integer m, integer n, 
	integer k, float alpha, float *a, integer lda, 
	float *b, integer ldb, float beta, float *c__, integer ldc)
{
	return sgemm_(transa, transb, &m, &n, &k, 
	&alpha, a, &lda, 
	b, &ldb,&beta, c__, &ldc);
}