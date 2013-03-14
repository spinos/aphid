/*
 *  Matrix33F.cpp
 *
 *  Created by jian zhang on 10/30/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include "Vector3F.h"
#include "Matrix33F.h"

Matrix33F::Matrix33F() 
{
	setIdentity();
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

Matrix33F Matrix33F::multiply(const Matrix33F& a) const
{
    Matrix33F r;
	*(r.m(0, 0)) = M(0, 0) * a(0, 0) + M(0, 1) * a(1, 0) + M(0, 2) * a(2, 0);
	*(r.m(0, 1)) = M(0, 0) * a(0, 1) + M(0, 1) * a(1, 1) + M(0, 2) * a(2, 1);
	*(r.m(0, 2)) = M(0, 0) * a(0, 2) + M(0, 1) * a(1, 2) + M(0, 2) * a(2, 2);
	
	*(r.m(1, 0)) = M(1, 0) * a(0, 0) + M(1, 1) * a(1, 0) + M(1, 2) * a(2, 0);
	*(r.m(1, 1)) = M(1, 0) * a(0, 1) + M(1, 1) * a(1, 1) + M(1, 2) * a(2, 1);
	*(r.m(1, 2)) = M(1, 0) * a(0, 2) + M(1, 1) * a(1, 2) + M(1, 2) * a(2, 2);
	
	*(r.m(2, 0)) = M(2, 0) * a(0, 0) + M(2, 1) * a(2, 0) + M(2, 2) * a(2, 0);
	*(r.m(2, 1)) = M(2, 0) * a(0, 1) + M(2, 1) * a(2, 1) + M(2, 2) * a(2, 1);
	*(r.m(2, 2)) = M(2, 0) * a(0, 2) + M(2, 1) * a(2, 2) + M(2, 2) * a(2, 2);
	return r;
}

Vector3F Matrix33F::transform(const Vector3F & a) const
{
	Vector3F b;
	b.x = a.x * M(0, 0) + a.y * M(1, 0) + a.z * M(2, 0);
	b.y = a.x * M(0, 1) + a.y * M(1, 1) + a.z * M(2, 1);
	b.z = a.x * M(0, 2) + a.y * M(1, 2) + a.z * M(2, 2);
	return b;
}

