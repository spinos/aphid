/*
 *  center_data.h
 *  
 *  remove mean
 *	X <- X - mean(X, dim)
 *
 *  Created by jian zhang on 12/19/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_CENTER_DATA_H
#define APH_CENTER_DATA_H

#include <math/linearMath.h>

namespace aphid {

/// mean of each column
template<typename T>
inline void col_mean(DenseVector<T> & m,
					const DenseMatrix<T> & X,
					const T & nn)
{
	const int & d = X.numCols();
	m.resize(d);
	m.setZero();
	for(int i=0;i<d;++i) {
		const T * cx = X.column(i);
		for(int j=0;j<X.numRows();++j) {
			m[i] += cx[j];		
		}
	}
	m.scale(T(1.0)/nn );
}

/// mean of each row
template<typename T>
inline void row_mean(DenseVector<T> & m,
					const DenseMatrix<T> & X,
					const T & nn)
{
	const int & d = X.numRows();
	m.resize(d);
	m.setZero();
	for(int i=0;i<d;++i) {
		for(int j=0;j<X.numCols();++j) {
			m[i] += X.column(j)[i];
		
		}
	}
	m.scale(T(1.0)/nn );
}

template<typename T>
inline void center_col(DenseMatrix<T> & X,
						const DenseVector<T> & m)
{
	const int & d = X.numCols();
	for(int i=0;i<d;++i) {
		T * cx = X.column(i);
		for(int j=0;j<X.numRows();++j) {
			cx[j] -= m[i];		
		}
	}
}

template<typename T>
inline void center_row(DenseMatrix<T> & X,
						const DenseVector<T> & m)
{
	const int & d = X.numRows();
	for(int i=0;i<d;++i) {
		for(int j=0;j<X.numCols();++j) {
			X.column(i)[j] -= m[i];		
		}
	}
}

template<typename T>
inline void center_data(DenseMatrix<T> & X, int dim,
						const T & nn)
{
	DenseVector<T> vmean;
	if(dim==1) {
		col_mean(vmean, X, nn);
		center_col(X, vmean);
	} else {
		row_mean(vmean, X, nn);
		center_row(X, vmean);
	}
}

}
#endif