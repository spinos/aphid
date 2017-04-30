/*
 *  symm_axb.h
 *  
 *	solve AX=B using sytrf and sytrs
 *  A is symmetric
 *
 *  Created by jian zhang on 12/24/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_MATH_SYMM_AXB_H
#define APH_MATH_SYMM_AXB_H

#include <math/linearMath.h>

namespace aphid {

template<typename T>
inline bool solve_symm_axb2(DenseMatrix<T> & A,
							DenseMatrix<T> & B)
{
	int m = A.numRows();
	int nrhs = m;
	int ldb = m;
	
	T * work;
	T queryWork;
	work = &queryWork;
	integer * ipiv = new integer[m];
	integer info;
	integer lwork = -1;
	clapack_sytrf<T>("U", m, A.raw(), m, ipiv, work, &lwork, &info);
	if(info != 0) {
		std::cout<<"\n ERROR sytrf query returned INFO="<<info<<"\n";
		return false;
	}
	
	lwork = work[0];
	work = new T[lwork];
	clapack_sytrf<T>("U", m, A.raw(), m, ipiv, work, &lwork, &info);
	if(info != 0) {
		std::cout<<"\n ERROR sytrf returned INFO="<<info<<"\n";
		return false;
	}
	
	clapack_sytrs<T>("U", m, nrhs, A.raw(), m, ipiv, B.raw(), ldb, &info);
	if(info != 0) {
		std::cout<<"\n ERROR sytri returned INFO="<<info<<"\n";
		return false;
	}
	
	delete[] work;
	delete[] ipiv;
	return info==0;
}

template<typename T>
inline bool solve_symm_axb(DenseMatrix<T> & X,
							DenseMatrix<T> & A,
							DenseMatrix<T> & B)
{
	X.copy(B);
	return solve_symm_axb2(A, X);
}

}
#endif
