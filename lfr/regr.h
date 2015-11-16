/*
 *  regr.h
 *  
 *
 *  Created by jian zhang on 11/17/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "linearMath.h"

namespace lfr {

/// Least Angle Regression
/// Y^ = x1b1 + x2b2 + ... xpbp
/// find b
/// minimize ||Y - Y^||
/// Y is measurement/signal/respounce
/// X matrix of predictors, normalized

template<typename T>
class LAR {
/// coefficients 
	DenseVector<T> m_beta;
	DenseVector<T> m_correl;
	DenseVector<T> m_Y;
	DenseVector<T> m_residual;
	const DenseMatrix<T> * m_X;
	
/// num predictors
	int m_p;
/// step size	
	T m_epsilon;
	
public:
	LAR(const DenseMatrix<T> * X);
	virtual ~LAR();
	
	void lars(const DenseVector<T> & Y);
protected:
	void findMaxCorrelated(int & dst);
private:
};

template<typename T>
LAR<T>::LAR(const DenseMatrix<T> * X) 
{
	m_X = X;
	m_p = X->numColumns();
	m_beta.create(m_p);
	m_correl.create(m_p);
	m_epsilon = 0.001;
	if(m_p > 1000) m_epsilon =  1.0/m_p;
}

template<typename T>
LAR<T>::~LAR() {}

template<typename T>
void LAR<T>::lars(const DenseVector<T> & Y)
{
	m_Y.copy(Y);
/// bj <- 0
	m_beta.setZero();
/// R <- Y
	m_residual.copy(Y);
	std::cout<<"\n err "<<m_residual.norm();
	
	int niter = 2000;
	if(m_p > 1000) niter = m_p * 2;
	std::cout<<"\n eps "<<m_epsilon<<" n iter "<<niter;
	
	DenseVector<T> dxj;
	dxj.create(m_X->numRows());
	
	int i, j = -1;
	int jpre = -1;
	for(i=0; i< niter; i++) {
		
		findMaxCorrelated(j);
		
		if(j != jpre) {
			std::cout<<"\n step "<<i;
			std::cout<<"\n max correlated ["<<j<<"] "<<m_correl.v()[j];
			if(jpre >= 0) {
				std::cout<<"\n pre max correlated ["<<jpre<<"] "<<m_correl.v()[jpre]
				<<"\n encounter new predictor, todo advance along least square direction.";
				break;
			}
			jpre = j;
		}

/// increase b
		T delta = m_epsilon * SIGN(m_correl.v()[j]);
		m_beta.v()[j] += delta;
/// advance along xj and reduce R
		m_X->getColumn(dxj, j);
		dxj.scale(delta);
		m_residual.minus(dxj);

	}
	std::cout<<"\n err "<<m_residual.norm();
}

template<typename T>
void LAR<T>::findMaxCorrelated(int & dst)
{
/// RtY correlation between current R and Y
	m_X->multTrans(m_correl, m_residual);
	dst = m_correl.maxAbsInd();
}

}