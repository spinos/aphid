/*
 *  deviate_mean.h
 *  
 *  deviate from mean
 *	
 *  dim = 1 data stored columnwise
 *  dim = 2 data stored rowwise
 *
 *  Created by jian zhang on 12/26/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_MATH_DEVIATE_MEAN_H
#define APH_MATH_DEVIATE_MEAN_H

#include <math/linearMath.h>

namespace aphid {

/// max distance among column vectors to mean
template<typename T>
inline T max_row_dist(const DenseMatrix<T> & X,
						const DenseVector<T> & vmean)
{
	T maxd = -1.0e20;
	T d;
	const int m = X.numRows();
	const int p = X.numCols();
	for(int i=0;i<p;++i) {
		const DenseVector<T> vcol(X.column(i), m);
		d = vcol.distanceTo(vmean);
		if(maxd < d) {
			maxd = d;
		}
	}
	return maxd;
}

/// max distance among row vectors to mean
template<typename T>
inline T max_col_dist(const DenseMatrix<T> & X,
						const DenseVector<T> & vmean)
{
	T maxd = -1.0e20;
	T d;
	const int m = X.numRows();
	const int p = X.numCols();
	
	DenseVector<T> vrow(p);
		
	for(int i=0;i<m;++i) {
		X.extractRowData(vrow.raw(), i);
		d = vrow.distanceTo(vmean);
		if(maxd < d) {
			maxd = d;
		}
	}
	return maxd;
}

template<typename T>
inline T deviate_from_mean(const DenseMatrix<T> & X, int dim)
{
	DenseVector<T> vmean;
	T nn;
	T r;
	if(dim==1) {
		nn = (T)X.numCols();
/// column vector
		row_mean(vmean, X, nn);
		r = max_row_dist(X, vmean);
		
	} else {
		nn = (T)X.numRows();
/// row vector
		col_mean(vmean, X, nn);
		r = max_col_dist(X, vmean);
	}
	return r;
}

}
#endif
