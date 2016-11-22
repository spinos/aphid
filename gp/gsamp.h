#ifndef APH_GPR_GSAMP_H
#define APH_GPR_GSAMP_H
/*
 * X = gsamp(covar, nsamp) Sample from a D-dimensional 
 * Gaussian distribution
 * mean vector is zero
 * X has nsamp rows in which each row represents a D-dimensional 
 * sample vector
 * 
 * https://github.com/sods/netlab/blob/master/gsamp.m
 */
#include <linearMath.h>

namespace aphid {
namespace gpr {
    
template<typename T>
inline void gsamp(lfr::DenseMatrix<T> & X,
    const lfr::DenseMatrix<T> & covar, int nsamp,
                    lfr::SvdSolver<T> * solver)
{
    int d = covar.numRows();
    if(!solver->compute(covar)) {
        std::cout<<"\n gamp cannot SVD";
        return;
    }
    
    lfr::DenseVector<T> sqs(d);
    
    for(int j=0;j<d;++j)
        sqs[j] = sqrt(solver->S()[j]);
    
    lfr::DenseMatrix<T> coeffs(nsamp, d);
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
}
#endif
