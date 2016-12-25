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
	std::cout<<"\n Y = "<<*m_y;
	
	const int dim = m_cov->K().numRows();
	TScalar logDetK;
	TScalar thetai;
	TScalar lli;
	TScalar llmax = -1.0e24;
	TScalar thetamax;
	
	std::cout<<"\n estimate theta within ("<<lmin<<", "<<lmax<<")";
	const TScalar deltal = (lmax - lmin) / 49;
	for(int i=0;i<50;++i) {
	
		thetai = lmin + deltal * i;
		m_kern->setParameter(thetai, (TScalar)1.0);
		
		m_cov->create(*m_x, *m_kern);
		
		DenseMatrix<TScalar> Kinv(dim, dim);
	DenseMatrix<TScalar> U(dim, dim);
	DenseMatrix<TScalar> YtKinv(1, dim);
	DenseMatrix<TScalar> YtKinvY(1, 1);
	
/// pdinv
		if(!cholesky_inv<TScalar>(U, Kinv, m_cov->K() ) ) {
			continue;
		}
	
		logDetK = logdet<TScalar>(U);
	
		m_y->transMult(YtKinv, Kinv );
	
		YtKinv.mult(YtKinvY, *m_y);
		
		std::cout<<"\n log|Kinv|"<<logDetK
			<<"\n YtKinv"<<YtKinv
			<<"\n model fit "<<YtKinvY.column(0)[0];
		
/// n * log(2 * PI)		
		lli = -0.5 * (logDetK + YtKinvY.column(0)[0] + (TScalar)dim * 1.8378770664093453);
		
		if(lli > llmax) {
			llmax = lli;
			thetamax = thetai;
		}
		//std::cout<<"\n ll["<<i<<"] = "<<lli
		//		<<" theta["<<i<<"] = "<<thetai;
		
	}

	std::cout<<"\n use theat "<<thetamax;
	std::cout.flush();
	
	m_kern->setParameter(thetamax, (TScalar)1.0);
	m_cov->create(*m_x, *m_kern);

	return llmax;
}


}
}