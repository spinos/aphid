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
/// coefficients 
	DenseVector<T> m_beta;
/// R
	DenseVector<T> m_residual;
/// D^t * R correlation to residual
	DenseVector<T> m_correl;
/// D predictors
	const DenseMatrix<T> * m_X;
/// D^t * D gram of D
	DenseMatrix<T> m_G;
/// gram of selected
	DenseMatrix<T> m_Ga;
	DenseMatrix<T> m_Gs;
	DenseMatrix<T> m_invGs;
/// indices to selections
	DenseVector<int> m_ind;
	DenseVector<T> m_u;
	DenseVector<T> m_work;
/// num predictors
	int m_p;
	int m_numSel;
/// step size	
	T m_epsilon;
	
public:
	LAR(const DenseMatrix<T> * X);
	virtual ~LAR();
	
	void lars(const DenseVector<T> & Y);
	const DenseVector<T> * coefficients() const;
protected:
	
private:
	void showProof(const DenseVector<T> & Y);
};

template<typename T>
LAR<T>::LAR(const DenseMatrix<T> * X) 
{
	m_X = X;
	m_p = X->numColumns();
	m_G.create(m_p, m_p);
	X->AtA(m_G);
	m_Ga.create(m_p, m_p);
	m_Gs.create(m_p, m_p);
	m_invGs.create(m_p, m_p);
	m_ind.create(m_p);
	m_u.create(m_p);
	m_beta.create(m_p);
	m_correl.create(m_p);
	m_work.create(m_p * 3);
	m_epsilon = 0.001;
}

template<typename T>
LAR<T>::~LAR() {}

template<typename T>
void LAR<T>::lars(const DenseVector<T> & Y)
{
/// ||Y||^2
	T normX = Y.normSq();
/// bj <- 0
	m_beta.setZero();
	m_u.setZero();
/// R <- Y
	m_residual.copy(Y);
	std::cout<<"\n in err "<<normX;
	
/// c <- D^t * R
	m_X->multTrans(m_correl, m_residual);
/// find most correlated predictor
	int currentInd = m_correl.maxAbsInd();
	
	T thrs = 0;
	m_numSel = 0;
	bool toSelect = true;
	int i, j;
/// for each predictor
	for(i=0; i< m_p; ++i) {
		if(toSelect) {
/// begin select
		m_ind.raw()[i] = currentInd;
		m_numSel++;
/// Ga[i] <- G[selected]
		m_Ga.copyColumn(i, m_G.column(currentInd));
/// fill upper part of Gs by Ga to (i,i)
		for (j = 0; j<=i; ++j)
            m_Gs.column(i)[j] = m_Ga.column(i)[m_ind.v()[j]];
			
/// fill inverse Gs
		if(i==0) {
			m_invGs.raw()[0] = T(1.0) / m_Gs.column(0)[0];
		}
		else {
/// u <- invGs * Gs 
			clapack_symv<T>("U",i,T(1.0),
                  m_invGs.column(0), m_p, m_Gs.column(i),1,T(0.0), m_u.raw(), 1);
/// i column of invGs
			const T schur = T(1.0) / ( m_Gs.column(i)[i] 
									- clapack_dot<T>(i, m_u.raw(), 1, m_Gs.column(i), 1) );
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
			m_work[j] = m_correl[m_ind[j]] > 0 ? T(1.0) : T(-1.0);
			
/// u <- invGs * work part to (i,i)
		clapack_symv<T>("U",i+1, T(1.0), m_invGs.column(0), m_p,
            m_work.v(), 1, T(0.0), m_u.raw(), 1);
			
/// for each predictor done, find any one has sign(bj) != sign(uj)
/// ratio of beta and u to be max step ?
		 T maxStep = INFINITY;
		 int firstPassZero = -1;
		 for (j = 0; j<=i; ++j) {
			T ratio = -m_beta[j] / m_u[j]; 
			if (ratio > 0 && ratio <= maxStep) {
				maxStep = ratio;
				firstPassZero = j;
			}
		}
		// std::cout<<"\n first zero"<<firstPassZero;
/// absolute max of <D, R>, first one selected
		T currentCorrelation = abs(m_correl[m_ind[0]]);
		
/// all 3 parts of work <- Ga * u
		clapack_gemv<T>("N", m_p, i+1, T(1.0), m_Ga.column(0),
            m_p, m_u.v(), 1, T(0.0), &m_work.raw()[2*m_p], 1);
		
		memcpy(&m_work.raw()[m_p], &m_work.raw()[2*m_p], m_p*sizeof(T));
		memcpy(m_work.raw(), &m_work.raw()[m_p], m_p*sizeof(T));

/// exclude predictors done in 1st and 2nd part
		for (j = 0; j<=i; ++j) {
			 m_work[m_ind[j]]=INFINITY;
			 m_work[m_ind[j]+m_p]=INFINITY;
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
			coeff1 += m_correl[m_ind[j]] > 0 ? m_u[j] : -m_u[j];
/// sum of cj * uj	
		T coeff2 = 0.0;
		for (j = 0; j<=i; ++j)
			coeff2 += m_correl[m_ind[j]] * m_u[j];
		
		// std::cout<<"\n maxStep "<<maxStep;
		std::cout<<"\n C "<<currentCorrelation;
		std::cout<<"\n step "<<step<<" max "<<maxStep;
/// step_max2 = current_correlation-constraint(lambda)
		step = min(min(step, currentCorrelation - 0.0), maxStep);
		
		if(step == INFINITY) break;
		
/// coefficients
/// beta <- beta + step * u
		clapack_axpy<T>(i+1, step, m_u.v(),1, m_beta.raw(),1);
		
/// correlations
/// c <- c - step * work
		clapack_axpy<T>(m_p, -step, &m_work[2*m_p], 1, m_correl.raw(), 1);

		std::cout<<"\n coeff1 "<<coeff1<<" 2 "<<coeff2;
/// reduce normX
		normX += coeff1*step*step - 2*coeff2*step;
		std::cout<<"\n normX "<<normX;
/// add thrs
		thrs += step * coeff1;
		
/// exit condition
		if(abs(step) < 1e-13 || normX < 1e-13) break;
		
		if (step == maxStep) {
			std::cout<<"\n step < maxStep\n downdate"
			<<" first pass zero "<<m_ind[firstPassZero]; // break;
			
			for( j = firstPassZero; j<i; ++j) {
				m_Ga.copyColumn(j, m_Ga.column(j+1));
				m_ind[j] = m_ind[j+1];
				m_beta[j] = m_beta[j+1];
			}
			m_ind[i] = -1;
			m_beta[i] = 0.0;
			
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
	}
	
	showProof(Y);
}
	
template<typename T>
void LAR<T>::showProof(const DenseVector<T> & Y)
{
	std::cout<<"\n n sel "<<m_numSel<<" |";
	int i, j;
	for(j=0;j<m_numSel;j++) {
		std::cout<<" "<<m_ind[j];
	}
	std::cout<<" |\n b |";
	for(j=0;j<m_numSel;j++) {
		std::cout<<" "<<m_beta[j];
	}

	std::cout<<" |\n y "<<Y;
	
	DenseVector<T> proof(Y.numElements());
	proof.setZero();
	for(i=0;i<Y.numElements();i++) {
		for(j=0;j<m_numSel;j++) {
			proof[i] += m_X->column(m_ind[j])[i] * m_beta[j];
		}
	}
	std::cout<<"\n y^ "<<proof;
	
	proof.minus(Y);
	std::cout<<"\n ||y - y^|| "<<proof.norm();
}

template<typename T>
const DenseVector<T> * LAR<T>::coefficients() const
{ return &m_beta; }

}