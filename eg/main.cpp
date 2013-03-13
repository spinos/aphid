// minimal linear system Ax=b example 
// numbers from http://people.fh-landshut.de/~maurer/femeth/node24.html
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/Sparse>
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

int main()
{
    testSimple();
    testLLT();
    resizeTest();
    testSparse();
    valuePtr();
    return 0;
}

