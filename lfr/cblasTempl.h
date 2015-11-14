/*
 *  cblasTempl.h
 *  
 *
 *  Created by jian zhang on 11/14/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <cblas.h>

template <typename T> void cblas_gemv( CBLAS_ORDER Order, CBLAS_TRANSPOSE Trans,  int M,  int N,
       T alpha,  T *A,  int lda,
       T* X,  int incX,
	   T beta, T * Y, int incY);
	   
template <typename T> void cblas_syrk( CBLAS_ORDER order, 
       CBLAS_UPLO Uplo,  CBLAS_TRANSPOSE Trans,  int N,  int K,
       T alpha,  T *A,  int lda,
       T beta, T*C,  int ldc);
	   
template <> inline void cblas_gemv<double>( CBLAS_ORDER Order, CBLAS_TRANSPOSE Trans,  int M,  int N,
       double alpha,  double *A,  int lda,
       double* X,  int incX,
	   double beta, double * Y, int incY) {
	cblas_dgemv(Order, Trans, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}

template <> inline void cblas_gemv<float>( CBLAS_ORDER Order, CBLAS_TRANSPOSE Trans,  int M,  int N,
       float alpha,  float *A,  int lda,
       float* X,  int incX,
	   float beta, float * Y, int incY) {
	cblas_sgemv(Order, Trans, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}

template <> inline void cblas_syrk<double>( CBLAS_ORDER Order, 
       CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans,  int N,  int K,
       double alpha,  double *A,  int lda,
       double beta, double *C,  int ldc) {
   cblas_dsyrk(Order,Uplo,Trans,N,K,alpha,A,lda,beta,C,ldc);
};

template <> inline void cblas_syrk<float>( CBLAS_ORDER Order, 
       CBLAS_UPLO Uplo,  CBLAS_TRANSPOSE Trans,  int N,  int K,
       float alpha,  float *A,  int lda,
       float beta, float *C,  int ldc) {
   cblas_ssyrk(Order,Uplo,Trans,N,K,alpha,A,lda,beta,C,ldc);
};
