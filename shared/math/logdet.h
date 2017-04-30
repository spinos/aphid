/*
 *  logdet.h
 *  
 *  https://github.com/jluttine/matlab/blob/master/gppca/logdet.m
 *	logarithm of determinant of a matrix
 *  v = logdet(U)
 *  computes the inverse of a real symmetric positive definite matrix A
 *  U is upper triangle matrix computed by cholesky factorization of A 
 *	v = 2 * sum(log(diag(chol(A))));
 *
 *  Created by jian zhang on 12/24/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_MATH_LOG_DET_H
#define APH_MATH_LOG_DET_H

#include <math/linearMath.h>

namespace aphid {

template<typename T>
inline T logdet(const DenseMatrix<T> & U) 
{
	T s = 0;
	for(int i=0;i<U.numRows();++i) {
		s += log(U.column(i)[i]);
	}
	return s * 2;
}

}
#endif
