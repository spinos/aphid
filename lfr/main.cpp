/* cblas_example.c */


#include <stdio.h>
#include <stdlib.h>
#include "linearMath.h"

void testblas()
{
   enum CBLAS_ORDER order;
   enum CBLAS_TRANSPOSE transa;

   double *a, *x, *y;
   double alpha, beta;
   int m, n, lda, incx, incy, i;

   order = CblasColMajor;
   transa = CblasNoTrans;

   m = 4; /* Size of Column ( the number of rows ) */
   n = 4; /* Size of Row ( the number of columns ) */
   lda = 4; /* Leading dimension of 5 * 4 matrix is 5 */
   incx = 1;
   incy = 1;
   alpha = 1;
   beta = 0;

   a = (double *)malloc(sizeof(double)*m*n);
   x = (double *)malloc(sizeof(double)*n);
   y = (double *)malloc(sizeof(double)*n);
   /* The elements of the first column */
   a[0] = 1;
   a[1] = 2;
   a[2] = 3;
   a[3] = 4;
   /* The elements of the second column */
   a[m] = 1;
   a[m+1] = 1;
   a[m+2] = 1;
   a[m+3] = 1;
   /* The elements of the third column */
   a[m*2] = 3;
   a[m*2+1] = 4;  
   a[m*2+2] = 5;
   a[m*2+3] = 6;
   /* The elements of the fourth column */
   a[m*3] = 5;
   a[m*3+1] = 6;
   a[m*3+2] = 7;
   a[m*3+3] = 8;
   /* The elemetns of x and y */ 
   x[0] = 1;
   x[1] = 2;
   x[2] = 1;
   x[3] = 1;
   y[0] = 0;
   y[1] = 0;
   y[2] = 0;
   y[3] = 0;
   
   cblas_dgemv( order, transa, m, n, alpha, a, lda, x, incx, beta,
                y, incy );
   /* Print y */
   for( i = 0; i < n; i++ ) 
      printf(" y%d = %f\n", i, y[i]);
   free(a);
   free(x);
   free(y);
}

int main ( )
{ 
	testblas();
	
	lfr::DenseMatrix<float> A;
	A.create(4, 8);
	
	float * c0 = A.column(0);
	c0[0] = 1;
	c0[1] = 2;
	c0[2] = 3;
	c0[3] = 4;
	c0[4] = 5;
	c0[5] = 6;
	c0[6] = 7;
	c0[7] = 8;
	
	float * c1 = A.column(1);
	c1[0] = 1;
	c1[1] = 3;
	c1[2] = 3;
	c1[3] = 1;
	c1[4] = 1;
	c1[5] = 2;
	c1[6] = 2;
	c1[7] = 2;
	
	float * c2 = A.column(2);
	c2[0] = 2;
	c2[1] = 2;
	c2[2] = 2;
	c2[3] = 1;
	c2[4] = 1;
	c2[5] = 1;
	c2[6] = 3;
	c2[7] = 3;
	
	float * c3 = A.column(3);
	c3[0] = 2;
	c3[1] = 2;
	c3[2] = -1;
	c3[3] = 1;
	c3[4] = 4;
	c3[5] = 4;
	c3[6] = 4;
	c3[7] = 1;

	A.normalize();
	
	std::cout<<" A "<<A;
	
	lfr::DenseMatrix<float> B;
	
	A.AtA(B);
	
	B.addDiagonal(1.f);
	std::cout<<" B = A' * A "<<B;
	
	lfr::DenseVector<float> L(B.column(1), B.numRows());
	
	std::cout<<" L "<<L;
	
	lfr::DenseVector<float> b;
	b.create(B.numRows());
	
	B.multTrans(b, L);
	
	std::cout<<" b "<<b;
	return 1;
}
