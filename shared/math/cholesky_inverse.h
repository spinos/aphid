/*
 *  choleskyInverse.h
 *  
 *  https://github.com/SheffieldML/ndlutil/blob/master/pdinv.m
 *  http://freesourcecode.net/matlabprojects/5373/sourcecode/pdinv.m
 *  http://www.math.utah.edu/software/lapack/lapack-d/dsysv.html
 *  http://people.sc.fsu.edu/~jburkardt/cpp_src/clapack/clapack_prb.cpp
 *	https://software.intel.com/en-us/node/520881
 *
 *  computes the inverse of a real symmetric positive definite matrix A
 *  output U and Ainv
 *  U is the Cholesky decomposition of A, so A = U**T * U
 *  Uinv = U \ eye(size(A,1)), left division, X = A \ B is the solution to
 *  A * X = B
 *  Ainv = Uinv * Uinv**T  
 *  
 *  Created by jian zhang on 12/24/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_MATH_CHOLESKY_INV_H
#define APH_MATH_CHOLESKY_INV_H

#include <math/ge_axb.h>

namespace aphid {

template<typename T>
inline bool cholesky_inv(DenseMatrix<T> & U,
						DenseMatrix<T> & Ainv,
						const DenseMatrix<T> & A) 
{
	U.copy(A);
	const int m = A.numRows();
	integer info;
	clapack_potrf<T>("U", m, U.raw(), m, &info);
	if(info != 0) {
		std::cout<<"\n ERROR cholesky_inv potrf INFO="<<info<<"\n";
		return false;
	}
	
	U.zeroLowerTriangle();
	
#if 0
	DenseMatrix<T> UtU(m, m);
	U.AtA(UtU);
	
	std::cout<<"A input"<<A;
	std::cout<<"UtU"<<UtU;
#endif
	
	DenseMatrix<T> B(m, m);
	B.setZero();
	B.addDiagonal((T)m);
	
	DenseMatrix<T> UU(m, m);
	UU.copy(U);
	
	DenseMatrix<T> Uinv(m, m);
	solve_ge_axb(Uinv, UU, B);
	
#if 0
	DenseMatrix<T> UUinv(m, m);
	U.mult(UUinv, Uinv);
	std::cout<<"U * Uinv"<<UUinv;
#endif

	B.copy(Uinv);

	Uinv.multTrans(Ainv, B);
	return true;
}

}
#endif
