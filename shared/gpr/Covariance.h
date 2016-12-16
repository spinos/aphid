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
#include <math/dist2.h>

namespace aphid {
namespace gpr {
 
template <typename TScalar, typename TKernel>
class Covariance {
   
public:
    
    typedef DenseMatrix<TScalar> TMatrix;
    
    Covariance();
    virtual ~Covariance();
    
    bool create(const TMatrix & x,
                const TKernel & k);
				
	bool create(const TMatrix & x1,
				const TMatrix & x2,
                const TKernel & k);
    
    const TMatrix & K() const;
	const TMatrix & Kinv() const;
    
private:
    TMatrix m_K;
	TMatrix m_invK;
    
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

	m_invK.resize(m_K.numRows(), m_K.numCols() );
	m_invK.copy(m_K);
/// A * A^-1 is not I
/// make it inversable ?
	//m_invK.addDiagonal(TScalar(0.01) );
	
	if(!m_invK.inverseSymmetric() ) {
		std::cout<<"\n ERROR Covariance K cannot inverse!";
		return 0;
	}
    return 1;
}

template <class TScalar, typename TKernel>
bool Covariance<TScalar, TKernel>::create(const TMatrix & x1,
				const TMatrix & x2,
                const TKernel & k)
{
	dist2(m_K, x1, x2);
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
const DenseMatrix<TScalar> & Covariance<TScalar, TKernel>::K() const
{ return m_K; }

template <class TScalar, typename TKernel>
const DenseMatrix<TScalar> & Covariance<TScalar, TKernel>::Kinv() const
{ return m_invK; }

}
}
#endif
