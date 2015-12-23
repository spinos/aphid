#include <iostream>
#include "linearMath.h"
#include <cmath>
#include "regr.h"
#include <MersenneTwister.h>

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

void testSVD3()
{
	double a[3*8]= {
            -0.008275, 0.01513, 0.0228, -0.02298, 0.007532, -0.04152, -0.01905, -0.01169,
0.064, 0.06322, 0.06318, 0.0007675, -4.113e-05, -0.06099, -0.06174, -0.06178,
0.1215, 0.002071, -0.1193, 0.1173, -0.1192, 0.1121, -0.002552, -0.1191
		};
		
	lfr::DenseMatrix<double> P(3, 8);
	int i, j;
	for(i=0;i<3;++i) {
		for(j=0;j<8;++j) {
			P.column(j)[i] = a[i*8+j];
		}
	}
	lfr::DenseMatrix<double> Q(3, 8);
	Q.copyData(P.raw());
	
	lfr::DenseMatrix<double> S(3, 3);
	P.multTrans(S, Q);
	std::cout<<" S "<<S;

	lfr::SvdSolver<double> slv;
	slv.compute(S);
	
	std::cout<<" s"<<*slv.S();
	std::cout<<" u"<<*slv.U();
	std::cout<<" v"<<*slv.V();
	
	lfr::DenseMatrix<double> M(3, 3);
	lfr::DenseMatrix<double> D(3, 3); D.setZero(); D.addDiagonal(1.0);
	slv.V()->transMult(M, D);
	M.multTrans(D, *slv.U());
	std::cout<<" M"<<D;
}

void testTM()
{
	lfr::DenseMatrix<double> Q(4, 3);
	Q.column(0)[0] = 1; Q.column(1)[0] = 5; Q.column(2)[0] = 9;
	Q.column(0)[1] = 2; Q.column(1)[1] = 6; Q.column(2)[1] = 10;
	Q.column(0)[2] = 3; Q.column(1)[2] = 7; Q.column(2)[2] = 11;
	Q.column(0)[3] = 4; Q.column(1)[3] = 8; Q.column(2)[3] = 12;
	
	lfr::DenseMatrix<double> P(4, 4);
	P.setZero();
	P.addDiagonal(2.0);
	
	std::cout<<"Q"<<Q;
	std::cout<<"P"<<P;

	lfr::DenseMatrix<double> R(3, 4);
	
	Q.transMultTrans(R, P);
	
	
	std::cout<<"R"<<R;
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

void testInv()
{
	std::cout<<"\n test symmetric inverse";
	lfr::DenseMatrix<double> A;
	A.create(N, N);
	memcpy(A.raw(), VA, N*N*sizeof(double));
	
	A.fillSymmetric();
	A.normalize();
	
	lfr::DenseMatrix<double> G;
	G.create(N, N);
	A.AtA(G);
	double X[N*N];
	memcpy(X, G.raw(), N*N*sizeof(double));
	
	std::cout<<"A"<<G;
	G.inverseSymmetric();
	std::cout<<"A^-1"<<G;
	
	double Y[N*N];
	
	clapack_gemm<double>("N", "N", N, N, N,
						1.0, G.column(0), N, X, N, 0.0, Y, N);
	
	printMatrix("A * A^-1", N, N, Y);					
}

#define XjM 6
#define NXj 4
const double VAs[XjM*NXj] = { .6, 3.27, 2.3, 3.2, 4.1, 4.4,
						-15.7, -.1, -2.1, .3, 4.5, 5.9,
						-6.3, -1.9, 2.1, 2.4, 4.1, 7.1,
						-4.1, 3.9, 3.93, 3.9, 2.2, .9 };
void testEquiangularVector()
{
	std::cout<<"\n test equiangular vector";
	lfr::DenseMatrix<double> XA(XjM, NXj);
	memcpy(XA.raw(), VAs, XjM*NXj*sizeof(double));
	XA.normalize();
	std::cout<<"\n XA"<<XA;
	lfr::DenseMatrix<double> GA;
	XA.AtA(GA);
	// std::cout<<"\n GA"<<GA;
	GA.inverseSymmetric();
	std::cout<<"\n GA^-1"<<GA;
	
	lfr::DenseVector<double> IA(NXj);
	IA.setOne();
	// std::cout<<"\n IA"<<IA;
	
	lfr::DenseVector<double> IGAI(NXj);
	GA.lefthandMult(IGAI, IA);
							
	std::cout<<"\n IGA"<<IGAI;
	double AA = 1.0 / sqrt(IGAI.sumVal());
	std::cout<<"\n AA "<<AA;
	
	lfr::DenseVector<double> AGI(NXj);
	GA.mult(AGI, IA);
	
	AGI.scale(AA);					
	std::cout<<"\n AGI "<<AGI;
	
	lfr::DenseVector<double> UA(XjM);
	XA.mult(UA, AGI);
	std::cout<<"\n UA "<<UA;
	std::cout<<"\n length "<<UA.norm();
	
	lfr::DenseVector<double> corr(NXj);
	XA.multTrans(corr, UA);
							
	std::cout<<"\n corr "<<corr;
}

void testLAR()
{
	std::cout<<"\n test least angle regression";
	MersenneTwister twist(99);
	
	const int p = 200;
	const int m = 11;
	lfr::DenseMatrix<double> A(m, p);
	lfr::DenseMatrix<double> G(p, p);
	
	double * c0 = A.raw();
	int i, j;
	for(i=0; i<p; i++) {
		for(j=0;j<m;j++) {
			c0[i*m+j] = twist.random() - 0.5;
		}
	}
	
	A.normalize();
	A.AtA(G);
	
	lfr::DenseVector<double> y(m);
	y.copyData(A.column(23));
	for(i=0;i<m;i++)
		y.raw()[i] += .03 * (twist.random() - 0.5);
	
	y.scale(10.0);
	
	
	lfr::DenseVector<double> beta(p);
	lfr::DenseVector<int> ind(p);
	lfr::LAR<double> lar(&A, &G);
	lar.lars(y, beta, ind);
}

int main()
{ 
    // testAdd();
	//testSVD();
    testSVD3();
	testTM();
	// testSqrt();
	// testInv();
	// testEquiangularVector();
	// testLAR();
    return 1;
}
