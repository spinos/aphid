/*
 *  Matrix44F.cpp
 *  easymodel
 *
 *  Created by jian zhang on 10/30/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include <cmath>
#include "Matrix44F.h"
#include "Vector3F.h"

Matrix44F::Matrix44F() 
{
	setIdentity();
}

Matrix44F::Matrix44F(float x)
{
	for(int i = 0; i < 16; i++) v[i] = x; 
}

Matrix44F::Matrix44F(Matrix44F & a)
{
	for(int i = 0; i < 16; i++) v[i] = a.v[i];
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

Matrix44F Matrix44F::operator* (const Matrix44F & a) const
{
	Matrix44F r(0.f);
	int i, j, k;
	for(j = 0; j < 4; j++) {
		for(i = 0; i < 4; i++) {
			for(k = 0; k < 4; k++) {
				*r.m(i, j) += M(k, j) * a.M(i, k);
			}
		}
	}
	return r;
}

void Matrix44F::operator*= (const Matrix44F & a)
{
	multiply(a);
}

void Matrix44F::multiply(const Matrix44F & a)
{
	Matrix44F t(*this);
	setZero();
	int i, j, k;
	for(j = 0; j < 4; j++) {
		for(i = 0; i < 4; i++) {
			for(k = 0; k < 4; k++) {
				*m(i, j) += t.M(k, j) * a.M(i, k);
			}
		}
	}
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

void Matrix44F::setZero()
{
	for(int i = 0; i < 16; i++) v[i] = 0.f;
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

void Matrix44F::setFrontOrientation(const Vector3F& front)
{
    Vector3F side = front.perpendicular();
    Vector3F up = front.cross(side);
    setOrientations(side, up, front);
}

void Matrix44F::rotateX(float alpha)
{
	const float c = cos(alpha);
	const float s = sin(alpha);
	Matrix44F r;
	*r.m(1, 1) =  c; *r.m(2, 1) = s;
	*r.m(1, 2) = -s; *r.m(2, 2) = c;
	multiply(r);
}

void Matrix44F::rotateY(float beta)
{
	const float c = cos(beta);
	const float s = sin(beta);
	Matrix44F r;
	*r.m(0, 0) = c; *r.m(2, 0) = -s;
	*r.m(0, 2) = s; *r.m(2, 2) = c;
	multiply(r);
}

void Matrix44F::rotateZ(float gamma)
{
	const float c = cos(gamma);
	const float s = sin(gamma);
	Matrix44F r;
	*r.m(0, 0) =  c; *r.m(1, 0) = s;
	*r.m(0, 1) = -s; *r.m(1, 1) = c;
	multiply(r);
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

float Matrix44F::Determinant33( float a, float b, float c, float d, float e, float f, float g, float h, float i )
{
    return float( a*( e*i - h*f ) - b*( d*i - g*f ) + c*( d*h - g*e ) );
}

void Matrix44F::glMatrix(float m[16]) const
{
	m[0] = M(0,0); m[1] = M(0,1); m[2] = M(0,2); m[3] = 0.0;
    m[4] = M(1,0); m[5] = M(1,1); m[6] = M(1,2); m[7] = 0.0;
    m[8] = M(2,0); m[9] = M(2,1); m[10] =M(2,2); m[11] = 0.0;
    m[12] = M(3,0); m[13] = M(3,1); m[14] = M(3,2) ; m[15] = 1.0;
}
//:~
