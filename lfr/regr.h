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
	DenseVector<T> m_correl;
	DenseVector<T> m_Y;
	DenseVector<T> m_residual;
	const DenseMatrix<T> * m_X;
	
/// num predictors
	int m_p;
/// step size	
	T m_epsilon;
	
	LASelector<T> m_selector;
	
public:
	LAR(const DenseMatrix<T> * X);
	virtual ~LAR();
	
	void lars(const DenseVector<T> & Y);
	const DenseVector<T> * coefficients() const;
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
	m_selector.create(X->numRows());
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
	
	int niter = 4000;
	if(m_p > 1000) niter = m_p * 4;
	std::cout<<"\n eps "<<m_epsilon<<" n iter "<<niter;
	
	DenseVector<T> dxj;
	dxj.create(m_X->numRows());
	
	T preErr = 1e10;
	int i, j;
	for(i=0; i< niter; i++) {
		findMaxCorrelated(j);
		m_selector.select(j, *m_X, m_correl);
		m_selector.deselect(m_correl);
		m_selector.advance(m_beta, m_residual, m_epsilon);
		
		if(m_residual.norm() > preErr) break;
		preErr = m_residual.norm();
	}
	std::cout<<"\n "<<i<<" err "<<m_residual.norm();
}

template<typename T>
void LAR<T>::findMaxCorrelated(int & dst)
{
/// RtY correlation between current R and Y
	m_X->multTrans(m_correl, m_residual);
	dst = m_correl.maxAbsInd();
}

template<typename T>
const DenseVector<T> * LAR<T>::coefficients() const
{ return &m_beta; }

}