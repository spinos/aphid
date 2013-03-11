// minimal linear system Ax=b example 
// numbers from http://people.fh-landshut.de/~maurer/femeth/node24.html
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/LU>
using namespace Eigen;
int main()
{
    MatrixXd m(3,3);
    m(0,0) = 4;
    m(0,1) = 1;
    m(0,2) = 0;
    m(1,0) = 2;
    m(1,1) = 4;
    m(1,2) = 1;
    m(2,0) = 1;
    m(2,1) = 2;
    m(2,2) = 2;
    
    std::cout << "m\n" << m << std::endl;
    
    VectorXd b(3);
    b(0) = 6;
    b(1) = 13;
    b(2) = 11;
    
    std::cout << "b\n" << b << std::endl;
    
    LU<MatrixXd> lu(m);
    VectorXd x;
    lu.solve(b, &x);
    std::cout << "x\n" << x << std::endl;

    return 0;
}

