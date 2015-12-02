#ifndef REGR_H
#define REGR_H

/*
 *  regr.h
 *  
 *
 *  Created by jian zhang on 11/17/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "linearMath.h"
#include <map>

namespace lfr {

#define sign(x) ((x)>=0 ? 1 : -1)

/// Least Angle Regression
/// Y^ = x1 * b1 + x2 * b2 + ... xp * bp
/// find b
/// minimize ||Y - Y^||
/// Y is measurement/signal/respounce
/// X matrix of predictors/patterns/atoms, normalized
/// Y^ is projected of Y

template<typename T>
class LAR {
/// D predictors
	const DenseMatrix<T> * m_X;
/// D^t * D gram of D
	const DenseMatrix<T> * m_G;
	
/// D^t * R correlation to residual
	DenseVector<T> m_correl;
/// gram of selected
	DenseMatrix<T> m_Ga;
	DenseMatrix<T> m_Gs;
	DenseMatrix<T> m_invGs;
	DenseVector<T> m_u;
	DenseVector<T> m_work;
/// num predictors
	int m_p;
	int m_numSel;
/// step size	
	T m_epsilon;
	
public:
	LAR(const DenseMatrix<T> * X, const DenseMatrix<T> * G);
	virtual ~LAR();
	
/// Y is signal
/// beta is coefficients 
/// ind is indices to selections
/// lambda is penalty constraint of sparsity, 
/// large lambda value induces less non-zero coefficients and less accurate projection
	void lars(const DenseVector<T> & Y, DenseVector<T> & beta, DenseVector<int> & ind, const T lambda = 0.1);
	
protected:
	
private:
	void showProof(const DenseVector<T> & Y, const DenseVector<T> & beta, const DenseVector<int> & ind);
};

template<typename T>
LAR<T>::LAR(const DenseMatrix<T> * X, const DenseMatrix<T> * G) 
{
	m_X = X;
	m_G = G;
	m_p = X->numColumns();
	m_Ga.create(m_p, m_p);
	m_Gs.create(m_p, m_p);
	m_invGs.create(m_p, m_p);
	m_u.create(m_p);
	m_correl.create(m_p);
	m_work.create(m_p * 3);
	m_epsilon = 0.001;
}

template<typename T>
LAR<T>::~LAR() {}

template<typename T>
void LAR<T>::lars(const DenseVector<T> & Y, DenseVector<T> & beta, DenseVector<int> & ind, const T lambda)
{
/// bj <- 0
	beta.setZero();
	m_u.setZero();

/// clear indices
	ind.setValue(-1);
	
/// ||Y||^2
	T normX = Y.normSq();
		
	if(normX < 1e-5) return;
	// std::cout<<"\n in err "<<normX;
	
/// R <- Y	
/// c <- D^t * R
	m_X->multTrans(m_correl, Y);
/// find most correlated predictor
	int currentInd = m_correl.maxAbsInd();
	
/// T thrs = 0;
	m_numSel = 0;
	bool toSelect = true;
	int i, j;
	int numIter = 0;
	const int maxNumIter = m_p<<2;
/// for each predictor
	for(i=0; i< m_p; ++i) {
		if(toSelect) {
/// begin select
		ind.raw()[i] = currentInd;
		m_numSel++;
/// Ga[i] <- G[selected]
		m_Ga.copyColumn(i, m_G->column(currentInd));
/// fill upper part of Gs by Ga to (i,i)
		for (j = 0; j<=i; ++j)
            m_Gs.column(i)[j] = m_Ga.column(i)[ind.v()[j]];
			
/// fill inverse Gs
		if(i==0) {
			m_invGs.raw()[0] = T(1.0) / m_Gs.column(0)[0];
		}
		else {
/// u <- invGs * Gs 
			clapack_symv<T>("U",i,T(1.0),
                  m_invGs.column(0), m_p, m_Gs.column(i),1,T(0.0), m_u.raw(), 1);
/// i column of invGs
			T lower = ( m_Gs.column(i)[i] 
									- clapack_dot<T>(i, m_u.raw(), 1, m_Gs.column(i), 1) );
			if(lower == 0) {
				// std::cout<<"\n divided by zero "<<lower;
				lower = 1.0;
			}
			const T schur = T(1.0) / lower;
			
			m_invGs.column(i)[i] = schur;
			memcpy(m_invGs.column(i), m_u.v(), i*sizeof(T));
			clapack_scal(i, -schur, m_invGs.column(i), 1);
/// rand 1 the part to (i-1,i-1)
			clapack_syr<T>("U",i, schur,  m_u.v(),1, m_invGs.raw(), m_p);
		}
/// end select
		} 
/// direction of work
		for (j = 0; j<=i; ++j)
			m_work[j] = m_correl[ind[j]] > 0 ? T(1.0) : T(-1.0);
			
/// u <- invGs * work part to (i,i)
		clapack_symv<T>("U",i+1, T(1.0), m_invGs.column(0), m_p,
            m_work.v(), 1, T(0.0), m_u.raw(), 1);
			
/// for each predictor done, find any one has sign(bj) != sign(uj)
/// ratio of beta and u to be max step ?
		 T maxStep = INFINITY;
		 int firstPassZero = -1;
		 for (j = 0; j<=i; ++j) {
			T ratio = -beta[j] / m_u[j]; 
			if (ratio > 0 && ratio <= maxStep) {
				maxStep = ratio;
				firstPassZero = j;
			}
		}
		// std::cout<<"\n first zero"<<firstPassZero;
/// absolute max of <D, R>, first one selected
		T currentCorrelation = absoluteValue<T>(m_correl[ind[0]]);
		
/// all 3 parts of work <- Ga * u
		clapack_gemv<T>("N", m_p, i+1, T(1.0), m_Ga.column(0),
            m_p, m_u.v(), 1, T(0.0), &m_work.raw()[2*m_p], 1);
		
		memcpy(&m_work.raw()[m_p], &m_work.raw()[2*m_p], m_p*sizeof(T));
		memcpy(m_work.raw(), &m_work.raw()[m_p], m_p*sizeof(T));

/// exclude predictors done in 1st and 2nd part
		for (j = 0; j<=i; ++j) {
			 m_work[ind[j]]=INFINITY;
			 m_work[ind[j]+m_p]=INFINITY;
		}

/// for each predictors not entered model
/// in both directions
/// work <- ( C - c ) / ( 1 - Ga * u)
/// C is max correlation
/// c is correlation of each predictor
		for (j = 0; j< m_p; ++j)
			m_work[j] = ((m_work[j] < INFINITY) && (m_work[j] > T(-1.0))) ? (m_correl[j] + currentCorrelation)/(T(1.0) + m_work[j]) : INFINITY;
		
		for (j = 0; j< m_p; ++j)
			m_work[j+m_p] = ((m_work[j+m_p] < INFINITY) && (m_work[j+m_p] < T(1.0))) ? (currentCorrelation - m_correl[j])/(T(1.0) - m_work[j+m_p]) : INFINITY;

/// select by smallest work 
		int index = m_work.minAbsInd(2*m_p);
		
		T step = m_work[index];
		
		currentInd = index % m_p;
		
/// sum of uj
		T coeff1 = 0.0;
		for (j = 0; j<=i; ++j)
			coeff1 += m_correl[ind[j]] > 0 ? m_u[j] : -m_u[j];
/// sum of cj * uj	
		T coeff2 = 0.0;
		for (j = 0; j<=i; ++j)
			coeff2 += m_correl[ind[j]] * m_u[j];
		
		// std::cout<<"\n maxStep "<<maxStep;
		// std::cout<<"\n c "<<currentCorrelation;
		// std::cout<<" step max "<<maxStep;
/// step_max2 = current_correlation - constraint(lambda)
		step = min(min(step, currentCorrelation - lambda), maxStep);
		// std::cout<<" step "<<step;
		if(step == INFINITY) break;
		
/// coefficients
/// beta <- beta + step * u
		clapack_axpy<T>(i+1, step, m_u.v(),1, beta.raw(),1);
		
/// correlations
/// c <- c - step * work
		clapack_axpy<T>(m_p, -step, &m_work[2*m_p], 1, m_correl.raw(), 1);

		// std::cout<<"\n coeff1 "<<coeff1<<" 2 "<<coeff2;
/// reduce normX
		normX += coeff1*step*step - 2*coeff2*step;
		// std::cout<<" normX "<<normX;
/// add thrs
///		thrs += step * coeff1;
		
		if (step == maxStep) {
			// std::cout<<"\n downdate "<<ind[firstPassZero];
			
			for( j = firstPassZero; j<i; ++j) {
				m_Ga.copyColumn(j, m_Ga.column(j+1));
				ind[j] = ind[j+1];
				beta[j] = beta[j+1];
			}
			ind[i] = -1;
			beta[i] = 0.0;
			
			for (j = firstPassZero; j<i; ++j) {
				memcpy(m_Gs.column(j), m_Gs.column(j+1), firstPassZero*sizeof(T));
				memcpy(&m_Gs.column(j)[firstPassZero], 
						&m_Gs.column(j+1)[firstPassZero+1], (i-firstPassZero)*sizeof(T));
			}
			
			const T schur = m_invGs.column(firstPassZero)[firstPassZero];
			memcpy(m_u.raw(), m_invGs.column(firstPassZero), firstPassZero*sizeof(T));
			memcpy(&m_invGs.column(firstPassZero)[firstPassZero],
					&m_invGs.column(firstPassZero+1)[firstPassZero], (i-firstPassZero)*sizeof(T));
					
			for (j = firstPassZero; j<i; ++j) {
				memcpy(m_invGs.column(j), m_invGs.column(j+1), firstPassZero*sizeof(T));
				memcpy(&m_invGs.column(j)[firstPassZero], 
						&m_invGs.column(j+1)[firstPassZero+1], (i-firstPassZero)*sizeof(T));
			}
			
			clapack_syr<T>("U", i, T(-1.0)/schur, m_u.v(),1, m_invGs.raw(), m_p);
			
			i -= 2;
			m_numSel--;
			toSelect = false;
		}
		else 
			toSelect = true;
			
/// exit condition
		if(numIter++ >= maxNumIter || absoluteValue<T>(step) < 1e-4
				|| step == (currentCorrelation - lambda)
				|| normX < 1e-8) break;
	}
}
	
template<typename T>
void LAR<T>::showProof(const DenseVector<T> & Y, const DenseVector<T> & beta, const DenseVector<int> & ind)
{
	std::cout<<"\n n sel "<<m_numSel<<" |";
	int i, j;
	for(j=0;j<m_numSel;j++) {
		std::cout<<" "<<ind[j];
	}
	std::cout<<" |\n b |";
	for(j=0;j<m_numSel;j++) {
		std::cout<<" "<<beta[j];
	}

	std::cout<<" |\n ";
	
	DenseVector<T> proof(Y.numElements());
	proof.setZero();
	for(i=0;i<Y.numElements();i++) {
		for(j=0;j<m_numSel;j++) {
			proof[i] += m_X->column(ind[j])[i] * beta[j];
		}
	}
	// std::cout<<"\n y^ "<<proof;
	
	proof.minus(Y);
	// std::cout<<"\n y^ - y "<<proof;
	
	std::cout<<"\n sum |y^ - y| "<<proof.sumAbsVal();
	std::cout<<"\n norm y "<<Y.norm()
		<<"\n sum |b| "<<beta.sumAbsVal();
}

}
#endif        //  #ifndef REGR_H
