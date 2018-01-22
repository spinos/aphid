/*
 *  ConjugateGradient.h
 *  
 *  solve ax = b by conjugate gradient method
 *
 *  Created by jian zhang on 1/12/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_MATH_ConjugateGradient_H
#define APH_MATH_ConjugateGradient_H

#include "sparseLinearMath.h"
#include "miscfuncs.h"

namespace aphid {

template<typename T>
class ConjugateGradient {

/// left-hand-side row-major x3
	SparseMatrix<T>* m_A;
	DenseVector<T> m_residual;
	DenseVector<T> m_Ap;
	DenseVector<T> m_prev;
	int m_numRows;
	
public:
	ConjugateGradient(SparseMatrix<T>* a);
	virtual ~ConjugateGradient();
	
	void solve(DenseVector<T>& x, const DenseVector<T>& b);
	
protected:

};

template<typename T>
ConjugateGradient<T>::ConjugateGradient(SparseMatrix<T>* a)
{
	m_A = a;
	m_numRows = a->numRows();
	m_residual.create(m_numRows);
	m_Ap.create(m_numRows);
	m_prev.create(m_numRows);
	
}

template<typename T>
ConjugateGradient<T>::~ConjugateGradient()
{}

template<typename T>
void ConjugateGradient<T>::solve(DenseVector<T>& x, const DenseVector<T>& b)
{
/// r_0 <- b - Ax_0
/// p_0 <- r_0
	for(unsigned i=0;i<m_numRows;++i) {
		
		m_residual[i] = b[i];
		
		SparseVector<T>& ai = m_A->row(i);
		SparseIterator<T> iter = ai.begin();
		SparseIterator<T> itEnd = ai.end();
		for(;iter != itEnd;++iter) {
			int j = iter.index();
/// Ax_0
			m_residual[i] -= iter.value() * x[j];
			
		}
		
		m_prev[i] = m_residual[i];
	}
	
	for(int k=0;k<20;++k) {
		T r_kTr_k =0;
		T d2=0;
		
	 	for(int i=0;i<m_numRows;++i) {

			m_Ap[i] = 0;
			 
			SparseVector<T>& ai = m_A->row(i);
			SparseIterator<T> iter = ai.begin();
			SparseIterator<T> itEnd = ai.end();
			for(;iter != itEnd;++iter) {
				int j = iter.index();
/// Ap_k				
				m_Ap[i] += iter.value() * m_prev[j];
				 
			}
/// r_k^Tr_k
			r_kTr_k += m_residual[i] * m_residual[i];
/// p_k^TAp_k			
			d2 += m_prev[i] * m_Ap[i];
		}
		
		if(Absolute<T>(d2)< 1e-6f)
			d2 = 1e-6f;
			
/// alpha <- r_k^Tr_k / p_k^TAp_k
		T alpha = r_kTr_k / d2;
		T r_k1Tr_k1 = 0;

/// x_k+1 <- x_k + alpha p_k
/// r_k+1 <- r_k - alpha A p_k
		for(int i=0;i<m_numRows;i++) {
			
			x[i] += m_prev[i] * alpha;
			m_residual[i] -= m_Ap[i] * alpha;
			
			r_k1Tr_k1 += m_residual[i] * m_residual[i];
		}
		
		//std::cout<<"\n step"<<k<<" residual "<<r_k1Tr_k1;
		
		if(r_k1Tr_k1 < 1e-3f)
			break;

		if(Absolute<T>(r_kTr_k)<1e-6f)
			r_kTr_k = 1e-6f;
			
/// beta <- r_k+1^T r_k+1 / r_k^Tr_k
		T beta = r_k1Tr_k1 / r_kTr_k;
/// p_k+1 <- r_k+1 + beta p_k		
		for(int i=0;i<m_numRows;++i) {
			m_prev[i] = m_residual[i] + m_prev[i] * beta;
		}		
	}
}
	

}

#endif
