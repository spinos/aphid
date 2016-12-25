/*
 *  ge_axb.h
 *  
 *	solve AX=B using getrf and getrs
 *  A is general
 *
 *  Created by jian zhang on 12/24/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_MATH_GE_AXB_H
#define APH_MATH_GE_AXB_H

#include <math/linearMath.h>

namespace aphid {

template<typename T>
inline bool solve_ge_axb2(DenseMatrix<T> & A,
							DenseMatrix<T> & B)
{
	int lda = A.numRows();
	
	integer * ipiv = new integer[lda];
	integer info;

	clapack_getrf<T>(A.numRows(), A.numCols(), A.raw(), lda, ipiv, &info);
	if(info != 0) {
		std::cout<<"\n ERROR getrf returned INFO="<<info<<"\n";
		delete[] ipiv;
		return false;
	}
	
	int nrhs = B.numCols();
	int ldb = B.numRows();
	
	clapack_getrs<T>("N", A.numRows(), nrhs, A.raw(), lda, ipiv, B.raw(), ldb, &info);
	if(info != 0) {
		std::cout<<"\n ERROR getrs query returned INFO="<<info<<"\n";
		delete[] ipiv;
		return false;
	}
	
	delete[] ipiv;
	return info==0;
}

template<typename T>
inline bool solve_ge_axb(DenseMatrix<T> & X,
							DenseMatrix<T> & A,
							DenseMatrix<T> & B)
{
	X.copy(B);
	return solve_ge_axb2(A, X);
}

}
#endif
