#ifndef APH_GPR_RBF_KERNEL_H
#define APH_GPR_RBF_KERNEL_H
/* 
 * Squared Exponential, also known as Radial Basis Function, kernel
 * k(x,x')=sigmaf^2 * exp(-(x-x')^2/(2 * l^2))
 */
#include <math/linearMath.h>

namespace aphid {
namespace gpr {
    
template <typename TScalar>
class RbfKernel {
    
    TScalar m_sigma; // deltaf
    TScalar m_lengthScale; // l
    TScalar m_sigma2;
    TScalar m_invlengthScale2;
    
public:
    
    typedef DenseVector<TScalar> TVector;
    
    RbfKernel();
    RbfKernel(TScalar lengthScale, TScalar sigma=1);
    virtual ~RbfKernel();
    
	void setParameter(TScalar lengthScale, TScalar sigma);
	
    virtual TScalar operator()(const TScalar & d) const;
    
private:
    
};

template <class TScalar>
RbfKernel<TScalar>::RbfKernel()
{
    setParameter(0.5, 1.0);
}

template <class TScalar>
RbfKernel<TScalar>::RbfKernel(TScalar lengthScale, TScalar sigma)
{
	setParameter(lengthScale, sigma);
}
    
template <class TScalar>
RbfKernel<TScalar>::~RbfKernel()
{}

template <class TScalar>
void RbfKernel<TScalar>::setParameter(TScalar lengthScale, TScalar sigma)
{
	m_lengthScale = lengthScale;
	m_sigma = sigma;
	m_sigma2 = sigma * sigma;
    m_invlengthScale2 = -0.5 / lengthScale / lengthScale;
}

template <class TScalar>
TScalar RbfKernel<TScalar>::operator()(const TScalar & d) const
{
	return m_sigma2 * std::exp(m_invlengthScale2 * d );
}

}
}
#endif

