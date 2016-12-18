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

namespace aphid {

template<typename T>
class PCAReduction {

	DenseMatrix<T> m_x;
	
public:
	PCAReduction();
	virtual ~PCAReduction();
	
/// d dimension of data
/// n number of data
	void createX(const int & d, 
				const int & n);
	
	void setXiCompj(const T & v,
				const int & i,
				const int & j);

	void compute(DenseMatrix<T> & reducedX,
				const int & toDim=2);
	
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
	center_data(m_x, 1);
	DenseMatrix<T> cov;
	
	m_x.AtA(cov);
	cov.scale((T)1.0 / (T)m_x.numRows() );
	
	std::cout<<"\n c "<<cov.numRows()<<"-by-"<<cov.numCols();
		
	SvdSolver<float> solv;
	solv.compute(cov);
	
	const DenseMatrix<T> & V = solv.Vt();//.transposed();
	DenseMatrix<T> reducedM(V.numCols(), toDim);
	
	for(int j=0;j<toDim;++j) {
		const T * vc = V.column(j);
		for(int i=0;i<V.numCols();++i) {
			reducedM.column(j)[i] = vc[i];
		}
	}
	
	std::cout<<" M "<<reducedM;
	
	reducedX.resize(m_x.numRows(), toDim);
	
	std::cout<<" x "<<m_x.numRows()<<"-by-"<<m_x.numCols();
	
	m_x.mult(reducedX, reducedM);
	std::cout<<" s "<<solv.S();
	
}

}
#endif