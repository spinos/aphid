#include <iostream>
#include "linearMath.h"
#include <cmath>

void printMatrix( char* desc, int m, int n, double* a ) 
{
	int i, j;
	std::cout<<"\n "<<desc;
	for(i=0; i< m; i++) {
        std::cout<<"\n| ";
        for(j=0; j< n; j++) {
            std::cout<<" "<<a[j*m + i];
        }
        std::cout<<" |";
    }
	std::cout<<"\n";
}

void testAdd()
{
	std::cout<<"\n test axpy";
	int m = 5, n = 4;
    double * A = new double[m*n];
    double * X = new double[m*n];
    int i, j;
    for(i=0; i< m; i++) {
        for(j=0; j< n; j++) {
            if(i==j) A[j*m + i] = 1.0;
            else A[j*m + i] = 0.0;
            X[j*m + i] = j*m + i;
        }
        
    }
    
    printMatrix("A", m, n, A);
    
	printMatrix("X", m, n, X);
    
    clapack_axpy<double>(m*n, .5, X, 1, A, 1);
    
	printMatrix("A <- A + 0.5 X", m, n, A);
}

#define M 6
#define N 5
#define LDA M
#define LDU M
#define LDVT N

/// A = U*SIGMA*VT
/// where SIGMA is an m-by-n matrix which is zero except for its min(m,n)
/// diagonal elements, U is an m-by-m orthogonal matrix 
/// and VT (V transposed) is an n-by-n orthogonal matrix. 
/// The diagonal elements of SIGMA are the singular values of A; 
/// they are real and non-negative, and are returned in descending order. The first min(m, n) columns of U and V are
/// the left and right singular vectors of A.

void testSVD()
{
	std::cout<<"\n test svd";
	int m = M, n = N, lda = LDA, ldu = LDU, ldvt = LDVT;
        double wkopt;
        double* work;
        /* Local arrays */
        double s[N];
		double u[LDU*M];
		double vt[LDVT*N];
        double a[LDA*N]= {
            8.79,  6.11, -9.15,  9.57, -3.49,  9.84,
            9.93,  6.91, -7.93,  1.64,  4.02,  0.15,
            9.83,  5.04,  4.86,  8.83,  9.80, -8.99,
            5.45, -0.27,  4.85,  0.74, 10.00, -6.02,
            3.16,  7.98,  3.01,  5.80,  4.27, -5.31
        };
		/* Query and allocate the optimal workspace */
        integer lwork = -1, info;
        clapack_gesvd<double>( "All", "All", m, n, a, lda, s, u, ldu, vt, ldvt, &wkopt, &lwork, &info );
        lwork = (int)wkopt;
		std::cout<<"\n work l "<<lwork;
        work = (double*)malloc( lwork*sizeof(double) );
        /* Compute SVD */
		clapack_gesvd<double>( "All", "All", m, n, a, lda, s, u, ldu, vt, ldvt, work, &lwork, &info );
        /* Check for convergence */
        if( info > 0 ) {
                printf( "The algorithm computing SVD failed to converge.\n" );
                exit( 1 );
        }
        /* Print singular values */
        printMatrix( "Singular values", 1, n, s );
        /* Print left singular vectors */
        printMatrix( "Left singular vectors (stored columnwise)", m, m, u );
        /* Print right singular vectors */
        printMatrix( "Right singular vectors (stored rowwise)", n, n, vt );
        /* Free workspace */
        free( (void*)work );
}

#define NSELECT 5
const double VA[N*N] = {
            0.67,  0.00,  0.00,  0.00,  0.00,
           -0.21,  3.82,  0.00,  0.00,  0.00,
            0.29, -0.13,  3.27,  0.00,  0.00,
           -1.06,  1.06,  0.11,  5.89,  0.00,
            0.46, -0.48,  1.10, -0.98,  4.54
        };
		
void testSqrt()
{
	std::cout<<"\n test sqrt Relatively Robust Representations";

#if 0
	double A[N*N] = {
			0.67,  0.00,  0.00,  0.00,  0.00,
           -0.21,  3.82,  0.00,  0.00,  0.00,
            0.29, -0.13,  3.27,  0.00,  0.00,
           -1.06,  1.06,  0.11,  5.89,  0.00,
            0.46, -0.48,  1.10, -0.98,  4.54
        };
		
	printMatrix("A", N, N, A);

	double W[N];
	double Z[N*NSELECT];
	integer ISUPPZ[2*N];
	int i, j;
	double * work;
	integer * iwork;
	double abstol = -1.0;
	double vl, vu;
	int il = 1;
	int iu = NSELECT;
	integer m;
	integer info;
	integer lwork = -1;
	integer liwork = -1;
	double queryWork; work = &queryWork;
	integer queryIwork; iwork = &queryIwork;
	
	clapack_syevr<double>("V", "A", "U", N, A, N, 
		&vl, &vu, il, iu, abstol, &m,
         W, Z, N, ISUPPZ, 
		 work, &lwork, iwork, &liwork, &info);
		 
	std::cout<<"\n m "<<m;
	std::cout<<"\n info "<<info;
	
	lwork = queryWork;
	liwork = queryIwork;
	std::cout<<"\n lwork "<<lwork;
	std::cout<<"\n liwork "<<liwork;
	
	work = (double *)malloc(lwork*sizeof(double));
	iwork = (integer *)malloc(liwork*sizeof(integer));
	
	clapack_syevr<double>("V", "A", "U", N, A, N, 
		&vl, &vu, il, iu, abstol, &m,
         W, Z, N, ISUPPZ, 
		 work, &lwork, iwork, &liwork, &info);
	if( info > 0 ) {
                std::cout<<"\nThe algorithm failed to compute eigenvalues.\n";
                exit( 1 );
        }
	std::cout<<"\n info "<<info;
	std::cout<<"\n The total number of eigenvalues found: "<<m;
	printMatrix("A", N, N, A);
	printMatrix( "W", 1, m, W);
	printMatrix( "Z", N, m, Z);
	
	double B[N*N];
	
    for(i=0; i< N; i++) {
		double  lambda=sqrt(W[i]);
        for(j=0; j< N; j++) {
            B[i*N + j] = Z[i*N + j] * lambda;
        }
    }
	
	printMatrix("B = Z * D", N, N, B);
	
	clapack_gemm<double>("N", "T", N, N, N,
						1.0, B, N, Z, N, 0.0, A, N);
						
	printMatrix("A' = B * Zt", N, N, A);
	
	memcpy(B, A, N*N*sizeof(double));
	
	clapack_gemm<double>("N", "N", N, N, N,
						1.0, A, N, B, N, 0.0, Z, N);
						
	printMatrix("A = A' * A'", N, N, Z);
#else
	lfr::DenseMatrix<double> A;
	A.create(N, N);
	double *va = A.raw();
	int i, j;
	for(i=0;i<N*N;i++) va[i] = VA[i];
	//A.fillSymmetric();
	
	std::cout<<"A"<<A;
	
	lfr::DenseMatrix<double> b;
	b.create(N, N);
	
	A.sqrtRRR(b);
	
	std::cout<<"b"<<b;
	
	double Y[N*N];
	
	clapack_gemm<double>("N", "N", N, N, N,
						1.0, b.column(0), N, b.column(0), N, 0.0, Y, N);
						
	printMatrix("A = A' * A'", N, N, Y);
#endif
}

int main()
{ 
    // testAdd();
    // testSVD();
	testSqrt();
    return 1;
}
