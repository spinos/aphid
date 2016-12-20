/*
 *  PCAPCAFeature.h
 *  
 *	use pca to find local space
 *  Created by jian zhang on 12/20/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_PCA_PCAFeature_H
#define APH_PCA_PCAFeature_H

#include <math/center_data.h>
#include <math/sort.h>

namespace aphid {

template<typename T, int Nvar=3>
class PCAFeature {

/// np-by-nvar nvar-dimensional point stored rowwise
	DenseMatrix<T> m_pnts;
	DenseMatrix<T> m_pcSpace;
/// nvar stores mean of each dimension
	DenseVector<T> m_mean;
/// 2-by-nvar lowest and highest of each dimension
/// stored rowwise
	DenseMatrix<T> m_bound;
	
public:
	PCAFeature(const int & np);
	virtual ~PCAFeature();
	
	void setPnt(const T * p, 
				const int & idx);
	
	static int numVars();
	int numPnts() const;
	const DenseMatrix<T> & pnts() const;
	void copy(const PCAFeature & another);
	
	void toLocalSpace();
	
	void getPCSpace(T * dst) const;
	void getDataPoint(T * p, const int & idx) const;
	const DenseMatrix<T> & dataPoints() const;
/// dim = 1 stored columnwise
/// dim = 2 stored rowwise
	void getBound(T * dst, const int & dim) const;
	
protected:
	
private:

};

template<typename T, int Nvar>
PCAFeature <T, Nvar>::PCAFeature(const int & np)
{
	m_pnts.resize(np, Nvar);
}

template<typename T, int Nvar>
PCAFeature <T, Nvar>::~PCAFeature()
{}

template<typename T, int Nvar>
void PCAFeature <T, Nvar>::setPnt(const T * p, 
						const int & idx)
{
	for(int i=0;i<Nvar;++i) {
		m_pnts.column(i)[idx] = p[i];
	}
}

template<typename T, int Nvar>
int PCAFeature <T, Nvar>::numPnts() const
{
	return m_pnts.numRows();
}

template<typename T, int Nvar>
int PCAFeature <T, Nvar>::numVars()
{ return Nvar; }

template<typename T, int Nvar>
const DenseMatrix<T> & PCAFeature <T, Nvar>::pnts() const
{
	return m_pnts;
}

template<typename T, int Nvar>
void PCAFeature <T, Nvar>::copy(const PCAFeature & another)
{
	m_pnts.resize(another.numPnts(), Nvar);
	m_pnts.copy(another.pnts() );
}

template<typename T, int Nvar>
void PCAFeature <T, Nvar>::toLocalSpace()
{
	center_data(m_pnts, 1, (T)numPnts(), m_mean);

	DenseMatrix<T> cov;
	m_pnts.AtA(cov);
	
	cov.scale((T)1.0 / (T)(numPnts()-1) );
	
	EigSolver<T> eig;
	eig.computeSymmetry(cov);
	
	DenseVector<T> sortedS(Nvar);
	sortedS.copyData(eig.S().v() );
	
	DenseVector<int> sortedInd(Nvar);
	for(int i=0;i<Nvar;++i) {
		sortedInd[i] = i;
	}
	
	sort_descent<T, int>(sortedInd.raw(), sortedS.raw(), 0, Nvar-1 );
	
	m_pcSpace.resize(Nvar, Nvar);
	for(int i=0;i<Nvar;++i) {
		float * vs = eig.V().column(sortedInd[i]);
		m_pcSpace.copyColumn(i, vs);
	}
	
	DenseMatrix<T> lft(m_pnts.numRows(), m_pnts.numCols() );
	lft.copy(m_pnts);
	
	lft.mult(m_pnts, m_pcSpace);
	
	m_bound.resize(2, Nvar);
	m_pnts.getBound(m_bound);
	
}

template<typename T, int Nvar>
void PCAFeature <T, Nvar>::getPCSpace(T * dst) const
{
	const int sizecp = Nvar*sizeof(T);
	for(int i=0;i<Nvar;++i) {
		m_pcSpace.extractColumnData(&dst[(Nvar+1)*i], i);
	}
	
	m_mean.extractData(&dst[(Nvar+1)*Nvar]);
	
}

template<typename T, int Nvar>
void PCAFeature<T, Nvar>::getDataPoint(T * p, 
				const int & idx) const
{
	for(int i=0;i<Nvar;++i) {
		p[i] = m_pnts.column(i)[idx];
	}
}

template<typename T, int Nvar>
const DenseMatrix<T> & PCAFeature<T, Nvar>::dataPoints() const
{ return m_pnts; }

template<typename T, int Nvar>
void PCAFeature<T, Nvar>::getBound(T * dst, const int & dim) const
{
	if(dim==1) {
		DenseMatrix<T> tb = m_bound.transposed();
		tb.extractData(dst);
		
	} else {
		m_bound.extractData(dst);
		
	}
}

}

#endif