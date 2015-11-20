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

template<typename T>
struct AngleProp {
/// correlation to signal
	T _corr;
/// ind in Xa
	int _colInd;
};
	
template<typename T>
class LASelector {

/// array of Xj
	DenseMatrix<T> m_Xa;
/// inverse of gram of Xa
	DenseMatrix<T> m_Gainv;
/// direction of advance
/// Ua <- Xa Wa
	DenseVector<T> m_Ua;
/// ones
	DenseVector<T> m_Ia;
///	Aa <- Ia^t Ga^-1 Ia
	DenseVector<T> m_IGaI;
///	Wa <- Aa Ga^-1 Ia
	DenseVector<T> m_Wa;
/// last selected
	int m_jLast;

public:
	LASelector();
	virtual ~LASelector() {}
/// clear and set first dim of Xj
	void create(const int m);
	bool select(int idx, const DenseMatrix<T> & X, const DenseVector<T> & correl);
	void deselect(const DenseVector<T> & correl);
	void advance(DenseVector<T> & beta, DenseVector<T> & residual, const T eps);
	int numSelected() const;
	
/// max num columns of Xa and Ga
	static int MaxNumSelections;
protected:

private:
	void calculateUa();
	void bisectUa();
	void equiangularUa();
	
private:
/// look up column index and previous correlation by j
typedef std::map<int, AngleProp<T> > LookupType;
	LookupType m_lookup;
	
};

template<typename T>
int LASelector<T>::MaxNumSelections = 25;

template<typename T>
LASelector<T>::LASelector() {}

template<typename T>
void LASelector<T>::create(const int m)
{
	m_lookup.clear();
	m_Xa.create(m, MaxNumSelections);
	m_Ua.create(m);
	m_Gainv.create(MaxNumSelections, MaxNumSelections);
	m_Ia.create(MaxNumSelections);
	m_Ia.setOne();
	m_IGaI.create(MaxNumSelections);
	m_Wa.create(MaxNumSelections);
}

template<typename T>
int LASelector<T>::numSelected() const
{ return m_lookup.size(); }

template<typename T>
bool LASelector<T>::select(int idx, const DenseMatrix<T> & X, const DenseVector<T> & correl)
{
/// already selected, skip
	if(m_lookup.find(idx) != m_lookup.end() ) return false;

	const int colInd = numSelected();
	const T corr = correl.v()[idx];
/// add sign(c) Xj to last	
	m_Xa.copyColumn(colInd, X.column(idx));
	m_Xa.scaleColumn(colInd, sign(corr));
	m_Xa.resizeNumCol(colInd+1);
/// add corr and colInd
	AngleProp<T> a;
	a._corr = corr;
	a._colInd = colInd;
	m_lookup[idx] = a;
	
	m_jLast = idx;
	
	if(numSelected() > 2) { //std::cout<<"\n Xa "<<m_Xa; 
		m_Xa.AtA(m_Gainv); //std::cout<<"\n Ga "<<m_Gainv;
		m_Gainv.addDiagonal(1e-6); //std::cout<<"\n Ga "<<m_Gainv;
		m_Gainv.inverseSymmetric();	//std::cout<<"\n Gainv "<<m_Gainv;
	}
	calculateUa();
	return true;
}

template<typename T>
void LASelector<T>::calculateUa()
{
	if(numSelected()==1)
		m_Ua.copyData(m_Xa.column(0));
	else if(numSelected()==2)
		bisectUa();
	else
		equiangularUa();
}

template<typename T>
void LASelector<T>::bisectUa()
{
	m_Ua.setZero();
	m_Ua.add(m_Xa.column(0), 0.5);
	m_Ua.add(m_Xa.column(1), 0.5);
	m_Ua.normalize();
}

template<typename T>
void LASelector<T>::equiangularUa()
{
	const int n = numSelected();
	m_Ia.resize(n);
	m_IGaI.resize(n);
	m_Wa.resize(n); 
	
	m_Gainv.lefthandMult(m_IGaI, m_Ia); //std::cout<<"\n IGaI "<<m_IGaI;
	
	const T Aa = 1.0 / sqrt(m_IGaI.sumVal()); //std::cout<<"\n scalar Aa "<<Aa;
	m_Gainv.mult(m_Wa, m_Ia);
	m_Wa.scale(Aa);
	
	m_Xa.mult(m_Ua, m_Wa);
	
	std::cout<<"\n c0 "<<m_Ua.dot(m_Xa.column(0))<<" c1 "<<m_Ua.dot(m_Xa.column(1))
			<<" c2 "<<m_Ua.dot(m_Xa.column(2));
}

template<typename T>
void LASelector<T>::deselect(const DenseVector<T> & correl)
{
	T lo = 1e8, hi = -1e8;
	typename LookupType::iterator it = m_lookup.begin();
	for(;it != m_lookup.end(); ++it) {
		//if(sign(correl.v()[it->first]) != sign(it->second._corr)) 
		//	std::cout<<" sign changed "<<it->first<<" "<<it->second._corr<<" / "<<correl.v()[it->first];
			
		//if(abs(correl.v()[it->first]) < 1e-3) 
		//	std::cout<<" drop "<<it->first<<" "<<correl.v()[it->first];
			
		if(correl.v()[it->first] < lo) lo = correl.v()[it->first];
		if(correl.v()[it->first] > hi) hi = correl.v()[it->first];
/// update corrj
		it->second._corr = correl.v()[it->first];
	}
	// std::cout<<" corr min/max "<<lo<<"/"<<hi;
}

template<typename T>
void LASelector<T>::advance(DenseVector<T> & beta, DenseVector<T> & residual, const T eps)
{
	//std::cout<<" n sel "<<numSelected()<<" Ua "<<m_Ua;
	// if(m_Ua.dot(residual) < 1e-3) 
	// std::cout<<"\n ua corr "<<m_Ua.dot(residual);
	T c = m_Ua.dot(residual) * eps; 
/// advance along Ua and reduce R
	residual.minus(m_Ua, c);
	
/// angle between Xj and Ua
	// if(numSelected()>1) c *= m_Ua.dot(m_Xa.column(0));
		
	typename LookupType::const_iterator it = m_lookup.begin();
	for(;it != m_lookup.end(); ++it) {
/// increase bj
		beta.raw()[it->first] += c * sign( it->second._corr ) / numSelected();
	}
}

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
/// step size	
	T m_epsilon;
	
public:
	LAR(const DenseMatrix<T> * X);
	virtual ~LAR();
	
	void lars(const DenseVector<T> & Y);
	const DenseVector<T> * coefficients() const;
protected:
	
private:
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
	
/// find most correlated predictor
	m_X->multTrans(m_correl, m_residual);
	int currentInd = m_correl.maxAbsInd();
	
	T thrs = 0;
	int nsel = 0;
	int i, j;
/// for each predictor
	for(i=0; i< m_p; i++) {
/// select
		std::cout<<"\n select X"<<currentInd;
		nsel++;
		m_ind.raw()[i] = currentInd;
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

/// select the smallest work
		int index = m_work.minAbsInd(2*m_p);
		
		T step = m_work[index];
		
		currentInd = index % m_p;
		std::cout<<"\n smallest work ind "<<currentInd;
		
/// sum of uj
		T coeff1 = 0;
		for (j = 0; j<=i; ++j)
			coeff1 += m_correl[m_ind[j]] > 0 ? m_u[j] : -m_u[j];
/// sum of cj * uj	
		T coeff2 = 0;
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

		// std::cout<<"\n coeff1 "<<coeff1<<" 2 "<<coeff2;
/// reduce normX
		normX += coeff1*step*step - 2*coeff2*step;
		std::cout<<"\n normX "<<normX;
/// add thrs
		thrs += step * coeff1;
		
		if (step == maxStep) {
			std::cout<<"\n step < maxStep todo downdate";
			break;
		}
		
/// exit condition
		if(abs(step) < 1e-15 || normX < 1e-15) break;
	}
	std::cout<<"\n n iter "<<i<<" out err "<<normX;
	std::cout<<"\n n sel "<<nsel;
	
	for(j=0;j<nsel;j++) {
		std::cout<<" "<<m_ind[j];
	}
	std::cout<<"\n b ";
	for(j=0;j<nsel;j++) {
		std::cout<<" "<<m_beta[j];
	}

	DenseVector<T> proof(Y.numElements());
	proof.setZero();
	for(i=0;i<Y.numElements();i++) {
		for(j=0;j<nsel;j++) {
			proof[i] += m_X->column(m_ind[j])[i] * m_beta[j];
		}
	}
	std::cout<<"\n proof "<<proof;
}

template<typename T>
const DenseVector<T> * LAR<T>::coefficients() const
{ return &m_beta; }

}