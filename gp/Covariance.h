#ifndef APH_GPR_COVARIANCE_H
#define APH_GPR_COVARIANCE_H
/*
 * Covariance function, all possible combinations of points
 * K = [k(x1,x1) k(x1,x2) ... k(x1,xn)]
 *     [k(x2,x1) k(x2,x2) ... k(x2,xn)]
 *     [    .       .     .      .    ]
 *     [    .       .      .     .    ]
 *     [    .       .       .    .    ]
 *     [k(x1,x1) k(x1,x2) ... k(x1,xn)]
 */
#include <linearMath.h>
#include "dist2.h"

namespace aphid {
namespace gpr {
 
template <typename TScalar, typename TKernel>
class Covariance {
   
public:
    
    typedef lfr::DenseMatrix<TScalar> TMatrix;
    
    Covariance();
    virtual ~Covariance();
    
    bool create(const TMatrix & x,
                const TKernel & k);
    
    const TMatrix & K() const;
    
private:
    TMatrix m_K;
    
};

template <class TScalar, typename TKernel>
Covariance<TScalar, TKernel>::Covariance()
{}

template <class TScalar, typename TKernel>
Covariance<TScalar, TKernel>::~Covariance()
{}

template <class TScalar, typename TKernel>
bool Covariance<TScalar, TKernel>::create(const TMatrix & x,
                                        const TKernel & k)
{
    dist2(m_K, x, x);
    int nc = m_K.numColumns();
    int nr = m_K.numRows();
    for(int j=0;j<nc;++j) {
        TScalar * kc = m_K.column(j);
        for(int i=0;i<nr;++i) {
            kc[i] = k(kc[i]);
        }
    }
    
    return 1;
}

template <class TScalar, typename TKernel>
const lfr::DenseMatrix<TScalar> & Covariance<TScalar, TKernel>::K() const
{ return m_K; }

}
}
#endif
