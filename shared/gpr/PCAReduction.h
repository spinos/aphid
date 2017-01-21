/*
 *  PCAReduction.h
 *  
 *	dimensionality reduction via principle component analysis
 *  draw n d-dimensional data points
 *  x is n-by-d, each row a data point
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
#include <math/deviate_mean.h>

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
/// pack all columns of v into i-th row as a data point
	void setXi(const DenseMatrix<T> & v,
				const int & idx);
				
	void setXiCompj(const T & v,
				const int & i,
				const int & j);

	bool compute(DenseMatrix<T> & reducedX,
				const int & toDim=2);

/// d	
	const int & nvars() const;
/// n
	const int & nobs() const;
/// idx-th observation
	void getXi(DenseVector<T> & xi,
				const int & idx) const;
				
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
void PCAReduction<T>::setXi(const DenseMatrix<T> & v,
				const int & idx)
{
	const int & ncv = v.numCols();
	const int & nrv = v.numRows();
	const int d = ncv * nrv;
	if(d != m_x.numCols() ) {
		throw " PCAReduction setXi has wrong input dims";
	}
	
	const T * vv = v.column(0);
	for(int i=0;i<d;++i) {
		m_x.column(i)[idx] = vv[i];
	}
	
}

template<typename T>
void PCAReduction<T>::setXiCompj(const T & v,
								const int & i,
								const int & j)
{
	m_x.column(j)[i] = v;
}

template<typename T>
bool PCAReduction<T>::compute(DenseMatrix<T> & reducedX,
							const int & toDim)
{
/// https://cn.mathworks.com/help/matlab/ref/cov.html
/// remove mean
	DenseVector<T> vmean;
	center_data(m_x, 1, (float)nobs(), vmean);
	
	const T dev = deviate_from_mean(m_x, 2);
/// fuzziness
	if(dev < 0.4) {
		std::cout<<"\n PCAReduction too small deviation "<<dev;		
		return false;
	}

#if 0
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
#else
	SvdSolver<T> svd;
	if(!svd.compute(m_x) ) {
		std::cout<<"\n PCA reduction cannot compute";
		return false;
	}

/// svd	X
/// X = USV^t
/// pc stored as rows of V^t
/// project a point (column vector x) to pc coordinate
/// V^tx	
/// take first m coordinates of V^tx
	
	reducedX.resize(nobs(), toDim);
	
	DenseVector<T> ax(nvars() );
	DenseVector<T> vtax(nvars() );
			
	for(int i=0;i<nobs();++i) {
		
		getXi(ax, i);
		
		svd.Vt().mult(vtax, ax);
		
		for(int j=0;j<toDim;++j) {
			reducedX.column(j)[i] = vtax[j];
		}
	}
	
#endif
	return true;
}

template<typename T>
const int &  PCAReduction<T>::nvars() const
{ return m_x.numCols(); }

template<typename T>
const int &  PCAReduction<T>::nobs() const
{ return m_x.numRows(); }

template<typename T>
void PCAReduction<T>::getXi(DenseVector<T> & xi,
							const int & idx) const
{ 
	const int & n = nvars();
	for(int i=0;i<n;++i) {
		xi[i] = m_x.column(i)[idx];
	}
}

}
#endif