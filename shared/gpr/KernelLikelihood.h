/*
 *  KernelLikelihood.h
 *  
 *
 *  Created by jian zhang on 12/24/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include <math/cholesky_inverse.h>
#include <math/logdet.h>

namespace aphid {
namespace gpr {

template <typename TScalar, typename TCovariance, typename TKernel>
class KernelLikelihood {

	TCovariance * m_cov;
	TKernel * m_kern;
	DenseMatrix<TScalar> * m_x;
	DenseMatrix<TScalar> * m_y;
	
public:
	KernelLikelihood(TCovariance * cov, TKernel * kern, 
					DenseMatrix<TScalar> * x, DenseMatrix<TScalar> * y);
	virtual ~KernelLikelihood();
	
	TScalar optimise(TScalar lmin, TScalar lmax);
	
protected:

private:

};

template <typename TScalar, typename TCovariance, typename TKernel>
KernelLikelihood<TScalar, TCovariance, TKernel>::KernelLikelihood(TCovariance * cov, TKernel * kern, 
							DenseMatrix<TScalar> * x, DenseMatrix<TScalar> * y)
{
	m_cov = cov;
	m_kern = kern;
	m_x = x;
	m_y = y;
}

template <typename TScalar, typename TCovariance, typename TKernel>
KernelLikelihood<TScalar, TCovariance, TKernel>::~KernelLikelihood()
{}

template <typename TScalar, typename TCovariance, typename TKernel>
TScalar KernelLikelihood<TScalar, TCovariance, TKernel>::optimise(TScalar lmin, TScalar lmax)
{
	const int dim = m_cov->K().numRows();
	TScalar logDetK;
	TScalar thetai;
	TScalar lli;
	TScalar llmax = -1.0e24;
	TScalar thetamax;
	DenseMatrix<TScalar> U(dim, dim);
	DenseMatrix<TScalar> YtKinv(1, dim);
	DenseMatrix<TScalar> YtKinvY(1, 1);
	
	std::cout<<"\n estimate theta within ("<<lmin<<", "<<lmax<<") by brute force";
	const TScalar deltal = (lmax - lmin) / 39;
	for(int i=0;i<40;++i) {
	
		thetai = lmin + deltal * i;
		m_kern->setParameter(thetai, (TScalar)1.0);
		
		m_cov->create(*m_x, *m_kern);

/// complexity	
		if(!cholesky_fac<TScalar>(U, m_cov->K() ) ) {
			continue;
		}
 	
		logDetK = logdet<TScalar>(U);
/// model fit	
		m_y->transMult(YtKinv,  m_cov->Kinv() );
		YtKinv.mult(YtKinvY, *m_y);	
			
/// n * log(2 * PI)	not added	
		lli = -0.5 * (logDetK + YtKinvY.column(0)[0]);
		
		if(lli > llmax) {
			llmax = lli;
			thetamax = thetai;
		}
	}

	std::cout<<"\n use theta "<<thetamax
		<<"\n max likelihood "<<llmax;
	std::cout.flush();
	
	m_kern->setParameter(thetamax, (TScalar)1.0);
	m_cov->create(*m_x, *m_kern);

	return llmax;
}


}
}