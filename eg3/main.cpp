#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
using Eigen::MatrixXd;
typedef Eigen::SparseMatrix<double> SparseMatrixType;

void testSparse()
{
    std::cout<<"test sparse\n";
    int n = 16;
    SparseMatrixType Rs(n, n);
    Rs.reserve(Eigen::VectorXi::Constant(n,3));
    for( int i = 0; i < n; i++ ) {
        Rs.insert(i,i) = 1.f;
        
        if(i> 0)Rs.insert(i,i-1) = -.5f;
        else Rs.insert(i,4) = -.5f;
        
        if(i< n-1)Rs.insert(i,i+1) = -.5f;
        else Rs.insert(i,3) = -.5f;
    }
    
    Rs.makeCompressed(); 
    std::cout << "Rs \n" << Rs << std::endl;
    
    Eigen::VectorXd b(n);
    b.setRandom();
    std::cout << "b \n" <<b<<"\n";
    
    Eigen::SimplicialLDLT<SparseMatrixType> chol(Rs);
    
    std::cout<<"U\n"<<chol.matrixU()<<"\n";
    std::cout<<"L\n"<<chol.matrixL()<<"\n";
    
    Eigen::VectorXd x = chol.solve(b); 
    std::cout << "x \n" <<x<<"\n";
}

int main()
{
    testSparse();
    return 0;
}
