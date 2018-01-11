/*
 *  MatrixC33F.h
 *  
 *  column-major layout 3-by-3 matrix
 *
 *  Created by jian zhang on 7/28/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_MATH_MATRIXC33F_H
#define APH_MATH_MATRIXC33F_H

#include <iostream>

namespace aphid {

class Vector3F;
class Quaternion;

 class MatrixC33F
 {
	float v[9];
	
 public:
	MatrixC33F();
	MatrixC33F(const MatrixC33F & a);
	MatrixC33F(const Quaternion& q);
	~MatrixC33F();
	
	void copy(const MatrixC33F & another);
	void setIdentity();
	void setZero();
	void addDiagonal(const float& a);
	
/// ith column
	float* col(int i);
	const float* col(int i) const;
	Vector3F colV(int i) const;
	void setCol(int i, const Vector3F& b);
/// b <- Ax matrix-vector product b is 3-by-1 column vector
	Vector3F operator*(const Vector3F& other ) const;
/// C <- AB matrix-matrix product C is 3-by-3 matrix
	MatrixC33F operator*(const MatrixC33F& b ) const;
	MatrixC33F operator*(const float& scaling ) const;	
	MatrixC33F operator+( const MatrixC33F& other ) const;
	MatrixC33F operator-( const MatrixC33F& other ) const;
	void operator+=( const MatrixC33F& other );
	void operator-=( const MatrixC33F& other );
	void operator*=( const float& scaling );
/// b <- A^tx
	Vector3F transMult(const Vector3F& other ) const;
/// C <- A^tB	
	MatrixC33F transMult(const MatrixC33F& b ) const;
	
	bool inverse();
	MatrixC33F inversed() const;
	void transpose();
	MatrixC33F transposed() const;	
/// |  0 -a3  a2 |
/// | a3   0 -a1 |
/// |-a2  a1   0 |
	void asCrossProductMatrix(const Vector3F& a);
/// |a1| |a1 a2 a3|
/// |a2|
/// |a3|
	void asAAtMatrix(const Vector3F& a);
/// c <- a^T a
	MatrixC33F AtA() const;
/// m_ij <- sqrt(m_ij)
	void squareRoot();
/// A Robust Method to Extract the Rotational Part of Deformations 
	void extractRotation(Quaternion& q, const int& maxIter = 10) const;
    friend std::ostream& operator<<(std::ostream &output, const MatrixC33F & p);
    
 };

}
#endif
