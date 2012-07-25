/*
 *  Matrix44F.cpp
 *  easymodel
 *
 *  Created by jian zhang on 10/30/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include "Vector3F.h"
#include "Matrix44F.h"

Matrix44F::Matrix44F() 
{
	setIdentity();
}

Matrix44F::~Matrix44F() {}

float Matrix44F::operator() (int i, int j)
{
	return v[i * 4 + j];
}

float Matrix44F::operator() (int i, int j) const
{
	return v[i * 4 + j];
}

float* Matrix44F::m(int i, int j)
{
	return &v[i * 4 + j];
}

float Matrix44F::M(int i, int j) const
{
	return v[i * 4 + j];
}

void Matrix44F::setIdentity()
{
	*m(0, 0) = *m(1, 1) = *m(2, 2) = *m(3, 3) = 1.0f;
	*m(0, 1) = *m(0, 2) = *m(0, 3) = *m(1, 0) = *m(1, 2) = *m(1, 3) = *m(2, 0) = *m(2, 1) = *m(2, 3) = *m(3, 0) = *m(3, 1) = *m(3, 2) = 0.0f;
}

float Matrix44F::determinant33( float a, float b, float c, float d, float e, float f, float g, float h, float i ) const
{
	return float( a*( e*i - h*f ) - b*( d*i - g*f ) + c*( d*h - g*e ) );
}

void Matrix44F::inverse()
{
	float det =       M(0, 0) * determinant33( M(1, 1), M(2, 1), M(3, 1), M(1, 2), M(2, 2), M(3, 2), M(1, 3), M(2, 3), M(3, 3) )
			- M(1, 0) * determinant33( M(0, 1), M(2, 1), M(3, 1), M(0, 2), M(2, 2), M(3, 2), M(0, 3), M(2, 3), M(3, 3) )
			+ M(2, 0) * determinant33( M(0, 1), M(1, 1), M(3, 1), M(0, 2), M(1, 2), M(3, 2), M(0, 3), M(1, 3), M(3, 3) )
			- M(3, 0) * determinant33( M(0, 1), M(1, 1), M(2, 1), M(0, 2), M(1, 2), M(2, 2), M(0, 3), M(1, 3), M(2, 3) );

	
	float m00 =   determinant33( M(1, 1), M(2, 1), M(3, 1), M(1, 2), M(2, 2), M(3, 2), M(1, 3), M(2, 3), M(3, 3) ) / det;
	float m10 = - determinant33( M(1, 0), M(2, 0), M(3, 0), M(1, 2), M(2, 2), M(3, 2), M(1, 3), M(2, 3), M(3, 3) ) / det;
	float m20 =   determinant33( M(1, 0), M(2, 0), M(3, 0), M(1, 1), M(2, 1), M(3, 1), M(1, 3), M(2, 3), M(3, 3) ) / det;
	float m30 = - determinant33( M(1, 0), M(2, 0), M(3, 0), M(1, 1), M(2, 1), M(3, 1), M(1, 2), M(2, 2), M(3, 2) ) / det;
	
	float m01 = - determinant33( M(0, 1), M(2, 1), M(3, 1), M(0, 2), M(2, 2), M(3, 2), M(0, 3), M(2, 3), M(3, 3) ) / det;
	float m11 =   determinant33( M(0, 0), M(2, 0), M(3, 0), M(0, 2), M(2, 2), M(3, 2), M(0, 3), M(2, 3), M(3, 3) ) / det;
	float m21 = - determinant33( M(0, 0), M(2, 0), M(3, 0), M(0, 1), M(2, 1), M(3, 1), M(0, 3), M(2, 3), M(3, 3) ) / det;
	float m31 =   determinant33( M(0, 0), M(2, 0), M(3, 0), M(0, 1), M(2, 1), M(3, 1), M(0, 2), M(2, 2), M(3, 2) ) / det;
	
	float m02 =   determinant33( M(0, 1), M(1, 1), M(3, 1), M(0, 2), M(1, 2), M(3, 2), M(0, 3), M(1, 3), M(3, 3) ) / det;
	float m12 = - determinant33( M(0, 0), M(1, 0), M(3, 0), M(0, 2), M(1, 2), M(3, 2), M(0, 3), M(1, 3), M(3, 3) ) / det;
	float m22 =   determinant33( M(0, 0), M(1, 0), M(3, 0), M(0, 1), M(1, 1), M(3, 1), M(0, 3), M(1, 3), M(3, 3) ) / det;
	float m32 = - determinant33( M(0, 0), M(1, 0), M(3, 0), M(0, 1), M(1, 1), M(3, 1), M(0, 2), M(1, 2), M(3, 2) ) / det;
	
	float m03 = - determinant33( M(0, 1), M(1, 1), M(2, 1), M(0, 2), M(1, 2), M(2, 2), M(0, 3), M(1, 3), M(2, 3) ) / det;
	float m13 =   determinant33( M(0, 0), M(1, 0), M(2, 0), M(0, 2), M(1, 2), M(2, 2), M(0, 3), M(1, 3), M(2, 3) ) / det;
	float m23 = - determinant33( M(0, 0), M(1, 0), M(2, 0), M(0, 1), M(1, 1), M(2, 1), M(0, 3), M(1, 3), M(2, 3) ) / det;
	float m33 =   determinant33( M(0, 0), M(1, 0), M(2, 0), M(0, 1), M(1, 1), M(2, 1), M(0, 2), M(1, 2), M(2, 2) ) / det;
	
	*m(0, 0) = m00;
	*m(0, 1) = m01;
	*m(0, 2) = m02;
	*m(0, 3) = m03;
	
	*m(1, 0) = m10;
	*m(1, 1) = m11;
	*m(1, 2) = m12;
	*m(1, 3) = m13;
	
	*m(2, 0) = m20;
	*m(2, 1) = m21;
	*m(2, 2) = m22;
	*m(2, 3) = m23;
	
	*m(3, 0) = m30;
	*m(3, 1) = m31;
	*m(3, 2) = m32;
	*m(3, 3) = m33;
}

Vector3F Matrix44F::transform(const Vector3F& p) const
{
	float tx = p.x * M(0, 0) + p.y * M(1, 0) + p.z * M(2, 0)  + M(3, 0);
	float ty = p.x* M(0, 1) + p.y* M(1, 1) + p.z* M(2, 1) + M(3, 1);
	float tz = p.x* M(0, 2) + p.y* M(1, 2) + p.z* M(2, 2) + M(3, 2);
		
	return Vector3F(tx, ty, tz);
}

Vector3F Matrix44F::transform(const Vector3F& p)
{
	float tx = p.x* *m(0, 0) + p.y* *m(1, 0) + p.z* *m(2, 0) + *m(3, 0);
	float ty = p.x* *m(0, 1) + p.y* *m(1, 1) + p.z* *m(2, 1) + *m(3, 1);
	float tz = p.x* *m(0, 2) + p.y* *m(1, 2) + p.z* *m(2, 2) + *m(3, 2);
		
	return Vector3F(tx, ty, tz);
}

Vector3F Matrix44F::transformAsNormal(const Vector3F& p) const
{
	float tx = p.x * M(0, 0) + p.y * M(1, 0) + p.z * M(2, 0);
	float ty = p.x * M(0, 1) + p.y * M(1, 1) + p.z * M(2, 1);
	float tz = p.x * M(0, 2) + p.y * M(1, 2) + p.z * M(2, 2);
		
	return Vector3F(tx, ty, tz);
}

Vector3F Matrix44F::transformAsNormal(const Vector3F& p)
{
	float tx = p.x* *m(0, 0) + p.y* *m(1, 0) + p.z* *m(2, 0);
	float ty = p.x* *m(0, 1) + p.y* *m(1, 1) + p.z* *m(2, 1);
	float tz = p.x* *m(0, 2) + p.y* *m(1, 2) + p.z* *m(2, 2);
		
	return Vector3F(tx, ty, tz);
}

void Matrix44F::translate(const Vector3F& p)
{
	translate( p.x, p.y, p.z);
}

void Matrix44F::setTranslation(const Vector3F& p)
{
	setTranslation( p.x, p.y, p.z);
}

void Matrix44F::translate(float x, float y, float z)
{
	*m(3, 0) += x;
	*m(3, 1) += y;
	*m(3, 2) += z;
}

void Matrix44F::setTranslation(float x, float y, float z)
{
	*m(3, 0) = x;
	*m(3, 1) = y;
	*m(3, 2) = z;
}

void Matrix44F::setOrientations(const Vector3F& side, const Vector3F& up, const Vector3F& front)
{
	*m(0, 0) = side.x;
	*m(0, 1) = side.y;
	*m(0, 2) = side.z;
	*m(1, 0) = up.x;
	*m(1, 1) = up.y;
	*m(1, 2) = up.z;
	*m(2, 0) = front.x;
	*m(2, 1) = front.y;
	*m(2, 2) = front.z;
}

Vector3F Matrix44F::getTranslation() const
{
	return Vector3F(M(3, 0), M(3, 1), M(3, 2));
}

Vector3F Matrix44F::getSide() const
{
	return Vector3F(M(0, 0), M(0, 1), M(0, 2));
}

Vector3F Matrix44F::getUp() const
{
	return Vector3F(M(1, 0), M(1, 1), M(1, 2));
}

Vector3F Matrix44F::getFront() const
{
	return Vector3F(M(2, 0), M(2, 1), M(2, 2));
}

void Matrix44F::transposed(float * mat) const
{
	mat[0] = M(0, 0);
	mat[1] = M(0, 1);
	mat[2] = M(0, 2);
	mat[3] = M(0, 3);
	mat[4] = M(1, 0);
	mat[5] = M(1, 1);
	mat[6] = M(1, 2);
	mat[7] = M(1, 3);
	mat[8] = M(2, 0);
	mat[9] = M(2, 1);
	mat[10] = M(2, 2);
	mat[11] = M(2, 3);
	mat[12] = M(3, 0);
	mat[13] = M(3, 1);
	mat[14] = M(3, 2);
	mat[15] = M(3, 3);
}
