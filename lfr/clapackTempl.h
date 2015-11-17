#include <f2c.h>
#include <clapack.h>

template <typename T> void clapack_axpy(integer N,
       T alpha,  T *X,  integer ldx,
       T* Y,  integer ldy);

template <typename T> void clapack_gemv(char * trans, integer M,  integer N,
       T alpha,  T *A,  integer lda,
       T* X,  integer incX,
	   T beta, T * Y, integer incY);

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
