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

/// np-by-nvar a nvar-dimensional point stored rowwise
	DenseMatrix<T> m_pnts;
	T m_mean[Nvar];
	
public:
	PCAFeature(const int & np);
	virtual ~PCAFeature();
	
	void setPnt(const T * p, 
				const int & idx);
	
	int numPnts() const;
	int numVars() const;
	const DenseMatrix<T> & pnts() const;
	void copy(const PCAFeature & another);
	
	void toLocalSpace();
	
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
int PCAFeature <T, Nvar>::numVars() const
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
	
	DenseMatrix<T> sortedV(Nvar, Nvar);
	for(int i=0;i<Nvar;++i) {
		float * vs = eig.V().column(sortedInd[i]);
		sortedV.copyColumn(i, vs);
	}
	
}

}

#endif