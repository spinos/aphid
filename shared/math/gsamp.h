#ifndef APH_MATH_GSAMP_H
#define APH_MATH_GSAMP_H
/*
 * X = gsamp(covar, nsamp) Sample from a D-dimensional 
 * Gaussian distribution
 * mean vector is zero
 * X has nsamp rows in which each row represents a D-dimensional 
 * sample vector
 * 
 * https://github.com/sods/netlab/blob/master/gsamp.m
 */
#include <math/linearMath.h>

namespace aphid {
   
template<typename T>
inline void gsamp(DenseMatrix<T> & X,
    const DenseMatrix<T> & covar, int nsamp,
                    SvdSolver<T> * solver)
{
    int d = covar.numRows();
    if(!solver->compute(covar)) {
        std::cout<<"\n gamp cannot SVD";
        return;
    }
    
    DenseVector<T> sqs(d);
    
    for(int j=0;j<d;++j)
        sqs[j] = sqrt(solver->S()[j]);
    
    DenseMatrix<T> coeffs(nsamp, d);
    for(int j=0;j<d;++j) {
        T * cc = coeffs.column(j);
        for(int i=0;i<nsamp;++i) {
            cc[i] = sqs[j] * GenerateGaussianNoise(0.0, 1.0);
        }
    }
    
    X.resize(nsamp, d);
    coeffs.mult(X, solver->Vt() );
}

}
#endif
