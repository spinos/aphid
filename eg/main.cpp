// minimal linear system Ax=b example 
// numbers from http://people.fh-landshut.de/~maurer/femeth/node24.html
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/Sparse>
#include <Eigen/SVD>
using namespace Eigen;
typedef SparseMatrix<double,Eigen::RowMajor> SparseMatrixType;

void testSimple()
{
    std::cout<<"LU\n";
    
    MatrixXf m(3,3);
    m(0,0) = 4;
    m(0,1) = 1;
    m(0,2) = 0;
    m(1,0) = 2;
    m(1,1) = 4;
    m(1,2) = 1;
    m(2,0) = 1;
    m(2,1) = 2;
    m(2,2) = 2;
    
    VectorXf b(3);
    b(0) = 6;
    b(1) = 13;
    b(2) = 11;
    std::cout << "m\n" << m << std::endl;
    std::cout << "b\n" << b << std::endl;
    
    
    LU<MatrixXf> lu(m);
    VectorXf x;
    lu.solve(b, &x);
    std::cout << "x\n" << x << std::endl;
    
}

void testLLT()
{
    std::cout<<"LLT\n";
    MatrixXf D = MatrixXf::Random(8,6);
    D = D * D.adjoint();
    MatrixXf m = MatrixXf::Random(8,6);
    D += m * m.adjoint();
    D(3, 0) = 0.f;
    D(3, 1) = -0.1f;
    D(3, 2) = -0.2f;
    D(3, 3) = 1.f;
    D(3, 4) = -0.7f;
    D(3, 5) = 0.f;

    std::cout<<"D\n"<<D<<std::endl;
    
    VectorXf b = VectorXf::Random(8);
    std::cout<<"b\n"<<b<<std::endl;
    
    MatrixXf A = D.transpose() * D;
    
    VectorXf b1 = D.transpose() * b;
    
    LLT<MatrixXf> llt(A);
    
    VectorXf x;
    llt.solve(b1,&x);
    std::cout << "x\n" << x << std::endl;
    std::cout << "check Ax\n" << D * x << std::endl;
}

void resizeTest()
{
    std::cout<<"Sizing row-major matrix\n";
    Matrix<float, Dynamic, Dynamic, RowMajor> m;
    m.resize(3, 3);
    m(0,0) = 4;
    m(0,1) = 1;
    m(0,2) = 0;
    m(1,0) = 2;
    m(1,1) = 4;
    m(1,2) = 1;
    m(2,0) = 1; 
    m(2,1) = 2;
    m(2,2) = 2;
    
    std::cout << "m before\n" << m << std::endl;
    
    m.resize(5, 3);
    std::cout << "m after\n" << m << std::endl << "coefficients changed\n";
}

void testSparse()
{
    std::cout<<"test sparse\n";
    int n = 16;
    SparseMatrixType Rs(n, n);
    for( int i = 0; i < n; i++ ) {
        if(i>0)
            Rs.fill(i, i-1) = .4f;
        Rs.fill(i, i) = -1.f;
        if(i<n-1)
            Rs.fill(i, i+1) = .6f;
    }
    std::cout << "Rs \n" << Rs << std::endl;
    
    SparseMatrixType RsT = Rs.transpose();
    
    std::cout << "RsT \n" << RsT << std::endl;
    
    SparseMatrixType RsTRs = RsT * Rs;
    
    std::cout << "RsTRs \n" << RsTRs << std::endl;
}

void testSVD()
{
	Matrix<float, 4, 2> A;
	A.setZero();
	A(0,0) = 2;
	A(0,1) = 4;
	A(1,0) = 1;
	A(1,1) = 3;
	
	std::cout<<"SVD A\n"<<A<<"\n";
	
	SVD<Matrix<float, 4, 2> > solver(A);
	
	std::cout<<"U\n"<<solver.matrixU()<<"\n";
	std::cout<<"V\n"<<solver.matrixV()<<"\n";
	std::cout<<"S\n"<<solver.singularValues()<<"\n";
}

void valuePtr()
{
    Matrix<double, 4, 4, Eigen::RowMajor> Rs;
    Rs.setZero();
    
    Rs(0, 1) = 0.1;
    Rs(0, 2) = 0.2;
    Rs(1, 1) = 1.1;
    Rs(1, 3) = 1.3;
    Rs(2, 3) = 2.3;
    Rs(3, 1) = 3.1;
    Rs(3, 3) = 1;

    double *p = Rs.data();
    std::cout<<"retrieve p\n";
    for(int i=0; i< 16; i++) {
        std::cout<<" "<<*p;
        p++;
        if((i+1)%4 == 0) 
            std::cout<<"\n";
    }
}

/*
// based on Least-Squares Rigid Motion Using SVD 
//      0    1
//      |  /
// 5 -  x  -  2
//   /  |
// 4    3
*/
void localRotation()
{
    float x[6] = {0, 1, 2, 0, -1, -1};
    float y[6] = {0, 0, 0, 0, 0, 0};
    float z[6] = {-1, -1, 0, 1, 1, 0};
    float xt[6] = {-0.3, -0.2, -0.15, 0.1, -1, -1};
    float yt[6] = {0, 1, 2, 0, 0, 0};
    float zt[6] = {-1, -1, 0, 1, 1, 0};
    
    float cx, cy, cz, cxt, cyt, czt;
    cx = 0.f;
    cy = 0.f;
    cz = 0.f;
    cxt = 0.f;
    cyt = 0.f;
    czt = 0.f;
    for(int i = 0; i < 6; i++) {
        cx += x[i] / 6.0;
        cy += y[i] / 6.0;
        cz += z[i] / 6.0;
        cxt += xt[i] / 6.0;
        cyt += yt[i] / 6.0;
        czt += zt[i] / 6.0;
    }
    
    float dx[6], dy[6], dz[6], dxt[6], dyt[6], dzt[6];
    for(int i = 0; i < 6; i++) {
        dx[i] = x[i] - cx;
        dy[i] = y[i] - cy;
        dz[i] = z[i] - cz;
        dxt[i] = xt[i] - cxt;
        dyt[i] = yt[i] - cyt;       
        dzt[i] = zt[i] - czt;
    }
    
    MatrixXf X(3, 6);
    MatrixXf Y(3, 6);
    for(int i = 0; i < 6; i++) {
        X(0, i) = dx[i];
        X(1, i) = dy[i];
        X(2, i) = dz[i];
        Y(0, i) = dxt[i];
        Y(1, i) = dyt[i];
        Y(2, i) = dzt[i];
    }
    MatrixXf W(6, 6);
    W.setZero();
    for(int i = 0; i < 6; i++) {
        W(i,i) = 1.f/6.f;
    }
    MatrixXf S = X * W * Y.transpose();
    
    SVD<MatrixXf > solver(S);
    
    float d = (solver.matrixV() * solver.matrixU().transpose()).determinant();
    std::cout<<"d\n"<<d<<"\n";
    MatrixXf D(3, 3);
    D.setIdentity();
    D(2,2) = d;
    
    std::cout<<"U\n"<<solver.matrixU()<<"\n";
    std::cout<<"V\n"<<solver.matrixV()<<"\n";
    
    MatrixXf R = solver.matrixV() * D * solver.matrixU().transpose();
    std::cout<<"R\n"<<R<<"\n";
}

int main()
{
    //testSimple();
    //testLLT();
    //resizeTest();
    //testSparse();
    //valuePtr();
	//testSVD();
	localRotation();
    return 0;
}

