/*
 *  Matrix33F.cpp
 *
 *  Created by jian zhang on 10/30/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include <cmath>
#include "Vector3F.h"
#include "Matrix33F.h"

Matrix33F::Matrix33F() 
{
	setIdentity();
}

Matrix33F::Matrix33F(Matrix33F & a)
{
	for(int i = 0; i < 9; i++) v[i] = a.v[i];
}

Matrix33F::Matrix33F(const Matrix33F & a)
{
	for(int i = 0; i < 9; i++) v[i] = a.v[i];
}

Matrix33F::~Matrix33F() {}

float Matrix33F::operator() (int i, int j)
{
	return v[i * 3 + j];
}

float Matrix33F::operator() (int i, int j) const
{
	return v[i * 3 + j];
}

Vector3F Matrix33F::operator*( Vector3F other ) const
{
	Vector3F v;
	v.x = M(0, 0) * other.x + M(0, 1) * other.y + M(0, 2) * other.z;
	v.y = M(1, 0) * other.x + M(1, 1) * other.y + M(1, 2) * other.z;
	v.z = M(2, 0) * other.x + M(2, 1) * other.y + M(2, 2) * other.z;
	
	return v;
}

Matrix33F Matrix33F::operator+( Matrix33F other ) const
{
	Matrix33F a;
	*a.m(0, 0) = M(0,0) + other.M(0,0);
	*a.m(0, 1) = M(0,1) + other.M(0,1);
	*a.m(0, 2) = M(0,2) + other.M(0,2);
	*a.m(1, 0) = M(1,0) + other.M(1,0);
	*a.m(1, 1) = M(1,1) + other.M(1,1);
	*a.m(1, 2) = M(1,2) + other.M(1,2);
	*a.m(2, 0) = M(2,0) + other.M(2,0);
	*a.m(2, 1) = M(2,1) + other.M(2,1);
	*a.m(2, 2) = M(2,2) + other.M(2,2);
	return a;
}

float* Matrix33F::m(int i, int j)
{
	return &v[i * 3 + j];
}

float Matrix33F::M(int i, int j) const
{
	return v[i * 3 + j];
}

void Matrix33F::setIdentity()
{
	*m(0, 0) = *m(1, 1) = *m(2, 2) = 1.0f;
	*m(0, 1) = *m(0, 2) = *m(1, 0) = *m(1, 2) = *m(2, 0) = *m(2, 1) = 0.0f;
}

void Matrix33F::setZero()
{
	for(int i = 0; i < 9; i++) v[i] = 0.f;
}

void Matrix33F::fill(const Vector3F& a, const Vector3F& b, const Vector3F& c)
{
	*m(0, 0) = a.x;
	*m(0, 1) = a.y;
	*m(0, 2) = a.z;
	*m(1, 0) = b.x;
	*m(1, 1) = b.y;
	*m(1, 2) = b.z;
	*m(2, 0) = c.x;
	*m(2, 1) = c.y;
	*m(2, 2) = c.z;
}

void Matrix33F::transpose()
{
    float tmp;
    tmp = M(0, 1);
    *m(0, 1) = M(1, 0);
    *m(1, 0) = tmp;

    tmp = M(0, 2);
    *m(0, 2) = M(2, 0);
    *m(2, 0) = tmp;
    
    tmp = M(1, 2);
    *m(1, 2) = M(2, 1);
    *m(2, 1) = tmp;
}

void Matrix33F::multiply(const Matrix33F& a)
{
	Matrix33F t(*this);
	setZero();
	int i, j, k;
	for(i = 0; i < 3; i++) {
		for(j = 0; j < 3; j++) {
			for(k = 0; k < 3; k++) {
				*m(i, j) += t.M(i, k) * a.M(k, j);
			}
		}
	}
}

Vector3F Matrix33F::transform(const Vector3F & a) const
{
	Vector3F b;
	b.x = a.x * M(0, 0) + a.y * M(1, 0) + a.z * M(2, 0);
	b.y = a.x * M(0, 1) + a.y * M(1, 1) + a.z * M(2, 1);
	b.z = a.x * M(0, 2) + a.y * M(1, 2) + a.z * M(2, 2);
	return b;
}

void Matrix33F::glMatrix(float m[16]) const
{
	m[0] = M(0,0); m[1] = M(0,1); m[2] = M(0,2); m[3] = 0.f;
    m[4] = M(1,0); m[5] = M(1,1); m[6] = M(1,2); m[7] = 0.f;
    m[8] = M(2,0); m[9] = M(2,1); m[10] =M(2,2); m[11] = 0.f;
    m[12] = m[13] = m[14] = 0.f; m[15] = 1.f;
}

/*
 *
 *  | a00 a01 a02 |-1             |   a22a11-a21a12  -(a22a01-a21a02)   a12a01-a11a02  |
 *  | a10 a11 a12 |    =  1/DET * | -(a22a10-a20a12)   a22a00-a20a02  -(a12a00-a10a02) |
 *  | a20 a21 a22 |               |   a21a10-a20a11  -(a21a00-a20a01)   a11a00-a10a01  |
 *
 *  with DET  =  a00(a22a11-a21a12)-a10(a22a01-a21a02)+a20(a12a01-a11a02)
 *
 */

void Matrix33F::inverse()
{
    const float det = determinant();
    
    *m(0, 0) =  determinant22(M(2, 2), M(1, 1), M(2, 1), M(1, 2)) / det;
    *m(0, 1) = -determinant22(M(2, 2), M(1, 0), M(2, 0), M(1, 2)) / det;
    *m(0, 2) =  determinant22(M(2, 1), M(1, 0), M(2, 0), M(1, 1)) / det;
	
	*m(1, 0) = -determinant22(M(2, 2), M(0, 1), M(2, 1), M(0, 2)) / det;
	*m(1, 1) =  determinant22(M(2, 2), M(0, 0), M(2, 0), M(0, 2)) / det;
	*m(1, 2) = -determinant22(M(2, 1), M(0, 0), M(2, 0), M(0, 1)) / det;
	
	*m(2, 0) =  determinant22(M(1, 2), M(0, 1), M(1, 1), M(0, 2)) / det;
	*m(2, 1) = -determinant22(M(1, 2), M(0, 0), M(1, 0), M(0, 2)) / det;
	*m(2, 2) =  determinant22(M(1, 1), M(0, 0), M(1, 0), M(0, 1)) / det;
	
	transpose();
}

float Matrix33F::determinant() const
{
    return M(0, 0) * determinant22(M(2, 2), M(1, 1), M(2, 1), M(1, 2)) - M(1, 0) * determinant22(M(2, 2), M(0, 1), M(2, 1), M(0, 2)) + M(2, 0) * determinant22(M(1, 2), M(0, 1), M(1, 1), M(0, 2));
}

float Matrix33F::determinant22(float a, float b, float c, float d) const
{
    return a * b - c * d;
}

void Matrix33F::rotateX(float alpha)
{
	const float c = cos(alpha);
	const float s = sin(alpha);
	Matrix33F r;
	*r.m(1, 1) =  c; *r.m(1, 2) = s;
	*r.m(2, 1) = -s; *r.m(2, 2) = c;
	multiply(r);
}

void Matrix33F::rotateY(float beta)
{
	const float c = cos(beta);
	const float s = sin(beta);
	Matrix33F r;
	*r.m(0, 0) = c; *r.m(0, 2) = -s;
	*r.m(2, 0) = s; *r.m(2, 2) = c;
	multiply(r);
}

void Matrix33F::rotateZ(float gamma)
{
	const float c = cos(gamma);
	const float s = sin(gamma);
	Matrix33F r;
	*r.m(0, 0) =  c; *r.m(0, 1) = s;
	*r.m(1, 0) = -s; *r.m(1, 1) = c;
	multiply(r);
}
