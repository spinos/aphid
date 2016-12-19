/*
 *  PCAReduction.h
 *  
 *	dimensionality reduction via principle component analysis
 *  draw n d-dimensional data
 *  x is n-by-d, each row a data
 *  reduce dimensionality from d to m 
 *  center x
 *  calculate d-by-d covariance matrix c = x^t * x
 *  calculate eigenvectors of c
 *  select m eigenvectors to be the new reduced space dimensions
 *
 *  http://deeplearning.stanford.edu/wiki/index.php/PCA
 *
 *  Created by jian zhang on 12/18/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_PCA_REDUCTION_H
#define APH_PCA_REDUCTION_H

#include <math/center_data.h>
#include <math/sort.h>

namespace aphid {

template<typename T>
class PCAReduction {

	DenseMatrix<T> m_x;
	
public:
	PCAReduction();
	virtual ~PCAReduction();
	
/// d number of variables per observation
/// n number of observations
	void createX(const int & d, 
				const int & n);
	
	void setXiCompj(const T & v,
				const int & i,
				const int & j);

	void compute(DenseMatrix<T> & reducedX,
				const int & toDim=2);

/// d	
	const int & nvars() const;
/// n
	const int & nobs() const;
	
protected:

private:
};

template<typename T>
PCAReduction<T>::PCAReduction()
{}

template<typename T>
PCAReduction<T>::~PCAReduction()
{}
	
template<typename T>
void PCAReduction<T>::createX(const int & d, 
								const int & n)
{
	m_x.resize(n, d);
}

template<typename T>
void PCAReduction<T>::setXiCompj(const T & v,
								const int & i,
								const int & j)
{
	m_x.column(j)[i] = v;
}

template<typename T>
void PCAReduction<T>::compute(DenseMatrix<T> & reducedX,
							const int & toDim)
{
/// https://cn.mathworks.com/help/matlab/ref/cov.html
/// remove mean
	center_data(m_x, 1, (float)nobs() );
	
	DenseMatrix<T> cov;
	m_x.AtA(cov);
	
/// normalized by n - 1
	cov.scale((T)1.0 / (T)(nobs()-1) );
		
	EigSolver<float> eig;
	eig.computeSymmetry(cov);
	
	DenseVector<float> sortedS(nvars());
	sortedS.copyData(eig.S().v() );
	
	DenseVector<int> sortedInd(nvars());
	for(int i=0;i<nvars();++i) {
		sortedInd[i] = i;
	}
	
	sort_descent<float, int>(sortedInd.raw(), sortedS.raw(), (int)0, (int)(nvars() - 1) );
	
	const DenseMatrix<T> & V = eig.V();

	DenseMatrix<T> reducedM(nvars(), toDim);
	
/// v is eigenvectors stored columnwise
/// select first ndim columns based on sorted eigenvalues
	for(int i=0;i<toDim;++i) {
		float * vs = V.column(sortedInd[i]);
		reducedM.copyColumn(i, vs);
	}
	
	reducedX.resize(nobs(), toDim);
	
	m_x.mult(reducedX, reducedM);
	
}

template<typename T>
const int &  PCAReduction<T>::nvars() const
{ return m_x.numCols(); }

template<typename T>
const int &  PCAReduction<T>::nobs() const
{ return m_x.numRows(); }

}
#endif