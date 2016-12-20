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
#include <math/Matrix33F.h>

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
/// fix flip problem, make more points with positive values of first 2 variables
	void rotatePositiveX();
	void rotatePositiveY();
	void makeRighthanded();
	void flipX();
	
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
#if 0
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
	
#else
	SvdSolver<T> svd;
	svd.compute(m_pnts);
	
	m_pcSpace.resize(Nvar, Nvar);
	
/// Vt is pc vectors stored rowwise
	for(int i=0;i<Nvar;++i) {
		float * vs = m_pcSpace.column(i);
		for(int j=0;j<Nvar;++j) {
			vs[j] = svd.Vt().column(j)[i];
		}
	}
	
	DenseVector<T> ax(Nvar );
	DenseVector<T> vtax(Nvar );
			
	for(int i=0;i<numPnts();++i) {
		
		getDataPoint(ax.raw(), i);		
		svd.Vt().mult(vtax, ax);
		
		for(int j=0;j<Nvar;++j) {
			m_pnts.column(j)[i] = vtax[j];
		}
	}
	
#endif
 
	makeRighthanded();
	rotatePositiveX();
	rotatePositiveY();
	
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

template<typename T, int Nvar>
void PCAFeature<T, Nvar>::rotatePositiveX()
{
	int posx = 0, negx = 0;
	for(int i=0;i<numPnts();++i) {
		if(m_pnts.column(0)[i] > 0) {
			posx++;
		} else {
			negx++;
		}
	}
	if(posx > negx) {
		return;
	}
	
/// rotate around y axis
	for(int i=0;i<Nvar;++i) {
		if(i==1) continue;
		for(int j=0;j<Nvar;++j) {
			m_pcSpace.column(i)[j] *= (T)-1.0;
		}
	}
	
	for(int i=0;i<numPnts();++i) {
		for(int j=0;j<Nvar;++j) {
			if(j==1) continue;
			m_pnts.column(j)[i] *= (T)-1.0;
		}
	}
}

template<typename T, int Nvar>
void PCAFeature<T, Nvar>::rotatePositiveY()
{
	int posx = 0, negx = 0;
	for(int i=0;i<numPnts();++i) {
		if(m_pnts.column(1)[i] > 0) {
			posx++;
		} else {
			negx++;
		}
	}
	if(posx > negx) {
		return;
	}
	
/// flip around x axis
	for(int i=1;i<Nvar;++i) {
		for(int j=0;j<Nvar;++j) {
			m_pcSpace.column(i)[j] *= (T)-1.0;
		}
	}
	
	for(int i=0;i<numPnts();++i) {
		for(int j=1;j<Nvar;++j) {
			m_pnts.column(j)[i] *= (T)-1.0;
		}
	}
	
}

template<typename T, int Nvar>
void PCAFeature<T, Nvar>::makeRighthanded()
{
	Vector3F vx(m_pcSpace.column(0));
	Vector3F vy(m_pcSpace.column(1));
	Vector3F vz(m_pcSpace.column(2));
				
	if(vx.cross(vy).dot(vz) < 0.5f) {
		flipX();
	}
}

template<typename T, int Nvar>
void PCAFeature<T, Nvar>::flipX()
{
	for(int j=0;j<Nvar;++j) {
		m_pcSpace.column(0)[j] *= (T)-1.0;
	}
	
	for(int i=0;i<numPnts();++i) {
		m_pnts.column(0)[i] *= (T)-1.0;
	}
}

}

#endif