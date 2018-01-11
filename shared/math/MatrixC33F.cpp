/*
 *  MatrixC33F.cpp
 *  
 *
 *  Created by jian zhang on 7/28/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "MatrixC33F.h"
#include "Quaternion.h"
#include "miscfuncs.h"
#include "clapackTempl.h"

namespace aphid {

MatrixC33F::MatrixC33F()
{}

MatrixC33F::MatrixC33F(const MatrixC33F & a)
{ copy(a); }

MatrixC33F::MatrixC33F(const Quaternion& q)
{
	float qx, qy, qz, qw, qx2, qy2, qz2, qxqx2, qyqy2, qzqz2, qxqy2, qyqz2, qzqw2, qxqz2, qyqw2, qxqw2;
    qx = q.x;
    qy = q.y;
    qz = q.z;
    qw = q.w;
    qx2 = ( qx + qx );
    qy2 = ( qy + qy );
    qz2 = ( qz + qz );
    qxqx2 = ( qx * qx2 );
    qxqy2 = ( qx * qy2 );
    qxqz2 = ( qx * qz2 );
    qxqw2 = ( qw * qx2 );
    qyqy2 = ( qy * qy2 );
    qyqz2 = ( qy * qz2 );
    qyqw2 = ( qw * qy2 );
    qzqz2 = ( qz * qz2 );
    qzqw2 = ( qw * qz2 );

	col(0)[0] = 1.0f - qyqy2 - qzqz2; 
	col(0)[1] = qxqy2 + qzqw2; 
	col(0)[2] = qxqz2 - qyqw2;
	 
	col(1)[0] = qxqy2 - qzqw2; 
	col(1)[1] = 1.0f - qxqx2 - qzqz2; 
	col(1)[2] = qyqz2 + qxqw2;
	 
	col(2)[0] = qxqz2 + qyqw2; 
	col(2)[1] = qyqz2 - qxqw2; 
	col(2)[2] = 1.0f - qxqx2 - qyqy2;

}

MatrixC33F::~MatrixC33F()
{}

void MatrixC33F::copy(const MatrixC33F & another)
{ memcpy(v, another.v, 36); }

float* MatrixC33F::col(int i)
{ return &v[i*3]; }

const float* MatrixC33F::col(int i) const
{ return &v[i*3]; }

Vector3F MatrixC33F::colV(int i) const
{ return Vector3F(v[i*3], v[i*3 + 1], v[i*3 + 2]); }

void MatrixC33F::setCol(int i, const Vector3F& b)
{ memcpy(col(i), &b, 12); }

void MatrixC33F::setIdentity()
{
	setZero();
	addDiagonal(1.f);
}

void MatrixC33F::addDiagonal(const float& a)
{ col(0)[0] = col(1)[1] = col(2)[2] = a; }

void MatrixC33F::setZero()
{ memset(v, 0, 36); }

/// http://mathinsight.org/matrix_vector_multiplication
Vector3F MatrixC33F::operator*(const Vector3F& other ) const
{ 
	Vector3F res;
	res.x = col(0)[0] * other.x +  col(1)[0] * other.y + col(2)[0] * other.z;
	res.y = col(0)[1] * other.x +  col(1)[1] * other.y + col(2)[1] * other.z;
	res.z = col(0)[2] * other.x +  col(1)[2] * other.y + col(2)[2] * other.z;	
	return res;
}

/// http://mathinsight.org/matrix_vector_multiplication
MatrixC33F MatrixC33F::operator*(const MatrixC33F& b ) const
{
	MatrixC33F C;
	C.col(0)[0] = col(0)[0] * b.col(0)[0] + col(1)[0] * b.col(0)[1] + col(2)[0] * b.col(0)[2];
	C.col(1)[0] = col(0)[0] * b.col(1)[0] + col(1)[0] * b.col(1)[1] + col(2)[0] * b.col(1)[2];
	C.col(2)[0] = col(0)[0] * b.col(2)[0] + col(1)[0] * b.col(2)[1] + col(2)[0] * b.col(2)[2];
	
	C.col(0)[1] = col(0)[1] * b.col(0)[0] + col(1)[1] * b.col(0)[1] + col(2)[1] * b.col(0)[2];
	C.col(1)[1] = col(0)[1] * b.col(1)[0] + col(1)[1] * b.col(1)[1] + col(2)[1] * b.col(1)[2];
	C.col(2)[1] = col(0)[1] * b.col(2)[0] + col(1)[1] * b.col(2)[1] + col(2)[1] * b.col(2)[2];
	
	C.col(0)[2] = col(0)[2] * b.col(0)[0] + col(1)[2] * b.col(0)[1] + col(2)[2] * b.col(0)[2];
	C.col(1)[2] = col(0)[2] * b.col(1)[0] + col(1)[2] * b.col(1)[1] + col(2)[2] * b.col(1)[2];
	C.col(2)[2] = col(0)[2] * b.col(2)[0] + col(1)[2] * b.col(2)[1] + col(2)[2] * b.col(2)[2];
	
	return C;
}

Vector3F MatrixC33F::transMult(const Vector3F& other ) const
{
	Vector3F res;
	res.x = col(0)[0] * other.x +  col(0)[1] * other.y + col(0)[2] * other.z;
	res.y = col(1)[0] * other.x +  col(1)[1] * other.y + col(1)[2] * other.z;
	res.z = col(2)[0] * other.x +  col(2)[1] * other.y + col(2)[2] * other.z;	
	return res;
}

MatrixC33F MatrixC33F::transMult(const MatrixC33F& b ) const
{
	MatrixC33F C;
	C.col(0)[0] = col(0)[0] * b.col(0)[0] + col(0)[1] * b.col(0)[1] + col(0)[2] * b.col(0)[2];
	C.col(1)[0] = col(0)[0] * b.col(1)[0] + col(0)[1] * b.col(1)[1] + col(0)[2] * b.col(1)[2];
	C.col(2)[0] = col(0)[0] * b.col(2)[0] + col(0)[1] * b.col(2)[1] + col(0)[2] * b.col(2)[2];
	
	C.col(0)[1] = col(1)[0] * b.col(0)[0] + col(1)[1] * b.col(0)[1] + col(2)[1] * b.col(0)[2];
	C.col(1)[1] = col(1)[0] * b.col(1)[0] + col(1)[1] * b.col(1)[1] + col(2)[1] * b.col(1)[2];
	C.col(2)[1] = col(1)[0] * b.col(2)[0] + col(1)[1] * b.col(2)[1] + col(2)[1] * b.col(2)[2];
	
	C.col(0)[2] = col(2)[0] * b.col(0)[0] + col(2)[1] * b.col(0)[1] + col(2)[2] * b.col(0)[2];
	C.col(1)[2] = col(2)[0] * b.col(1)[0] + col(2)[1] * b.col(1)[1] + col(2)[2] * b.col(1)[2];
	C.col(2)[2] = col(2)[0] * b.col(2)[0] + col(2)[1] * b.col(2)[1] + col(2)[2] * b.col(2)[2];
	
	return C;
}

MatrixC33F MatrixC33F::operator*(const float& scaling ) const
{
	MatrixC33F C;
	for(int i=0;i<9;++i) {
		C.v[i] = v[i] * scaling;
	}
	return C;
}

MatrixC33F MatrixC33F::operator+( const MatrixC33F& other ) const
{
	MatrixC33F C;
	for(int i=0;i<9;++i) {
		C.v[i] = v[i] + other.v[i];
	}
	return C;
}

MatrixC33F MatrixC33F::operator-( const MatrixC33F& other ) const
{
	MatrixC33F C;
	for(int i=0;i<9;++i) {
		C.v[i] = v[i] - other.v[i];
	}
	return C;
}

void MatrixC33F::operator+=( const MatrixC33F& other )
{
	for(int i=0;i<9;++i) {
		v[i] += other.v[i];
	}
}

void MatrixC33F::operator-=( const MatrixC33F& other )
{
	for(int i=0;i<9;++i) {
		v[i] -= other.v[i];
	}
}

void MatrixC33F::operator*=( const float& scaling )
{
	for(int i=0;i<9;++i) {
		v[i] *= scaling;
	}
}

bool MatrixC33F::inverse()
{
	integer * ipiv = new integer[3];
	integer info;

	clapack_getrf<float>(3, 3, v, 3, ipiv, &info);
	if(info != 0) {
		std::cout<<"\n ERROR MatrixC33F::inverse getrf returned INFO="<<info<<"\n";
		return false;
	}
	
	float * work;
	float queryWork;
	work = &queryWork;
	integer lwork = -1;
	
	clapack_getri<float>(3, v, 3, ipiv, work, &lwork, &info);
	if(info != 0) {
		std::cout<<"\n ERROR MatrixC33F::inverse sytri returned INFO="<<info<<"\n";
		return false;
	}
	
	lwork = work[0];
	work = new float[lwork];
	clapack_getri<float>(3, v, 3, ipiv, work, &lwork, &info);
	if(info != 0) {
		std::cout<<"\n ERROR MatrixC33F::inverse sytri returned INFO="<<info<<"\n";
		return false;
	}
	
	delete[] work;
	delete[] ipiv;
	return info==0;
}

MatrixC33F MatrixC33F::inversed() const
{ 
	MatrixC33F b(*this);
	b.inverse();
	return b;
}

///    10 20
/// 01    21
/// 02 12   
void MatrixC33F::transpose()
{
	float tmp;
	tmp = col(1)[0];
	col(1)[0] = col(0)[1];
	col(0)[1] = tmp;
	
	tmp = col(2)[0];
	col(2)[0] = col(0)[2];
	col(0)[2] = tmp;
	
	tmp = col(2)[1];
	col(2)[1] = col(1)[2];
	col(1)[2] = tmp;
}

MatrixC33F MatrixC33F::transposed() const
{
	MatrixC33F b(*this);
	b.transpose();
	return b;
}

void MatrixC33F::asCrossProductMatrix(const Vector3F& a)
{
	col(0)[0] =  0.f;
	col(0)[1] =  a.z;
	col(0)[2] = -a.y;
	
	col(1)[0] = -a.z;
	col(1)[1] =  0.f;
	col(1)[2] =  a.x;
	
	col(2)[0] =  a.y;
	col(2)[1] = -a.x;
	col(2)[2] =  0.f;
}

void MatrixC33F::asAAtMatrix(const Vector3F& a)
{
	setCol(0, a * a.x);
	setCol(1, a * a.y);
	setCol(2, a * a.z);
}

MatrixC33F MatrixC33F::AtA() const
{
	return transMult(*this);
}

void MatrixC33F::squareRoot()
{
	for(int i=0;i<9;++i) {
		v[i] = sqrt(v[i]);
	}
}

/// sigma ( r_i x a_i ) / ( | sigma ( r_i . a_i ) | + eta )
void  MatrixC33F::extractRotation(Quaternion& q, const int& maxIter) const
{
	for(int i=0;i<maxIter;++i) {
		MatrixC33F R(q);
		Vector3F omega = (R.colV(0).cross( colV(0) )
					+ R.colV(1).cross( colV(1) )
					+ R.colV(2).cross( colV(2) ) ) 
					* (1.f / ( Absolute<float>(R.colV(0).dot( colV(0) )
					+ R.colV(1).dot( colV(1) )
					+ R.colV(2).dot( colV(2) )) + 1e-9f ) );
					
		float w = omega.length();
		if(w < 1e-9f)
			break;
			
		// std::cout<<"\n iter"<<i<<" w "<<w;
			 
		q = Quaternion(w, omega.normal() ) * q;
		q.normalize();
	}
}

std::ostream& operator<<(std::ostream &output, const MatrixC33F & p) 
{
    output<<"\n |"<<p.v[0]<<", "<<p.v[3]<<", "<<p.v[6]
		<<"|\n |"<<p.v[1]<<", "<<p.v[4]<<", "<<p.v[7]
		<<"|\n |"<<p.v[2]<<", "<<p.v[5]<<", "<<p.v[8]<<"|";
	return output;
}

}