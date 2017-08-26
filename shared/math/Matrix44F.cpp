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
namespace aphid {

Matrix44F Matrix44F::IdentityMatrix;

Matrix44F::Matrix44F() 
{
	setIdentity();
}

Matrix44F::Matrix44F(float x)
{
	for(int i = 0; i < 16; i++) v[i] = x; 
}

Matrix44F::Matrix44F(const Matrix44F & a)
{ copy(a); }

Matrix44F::Matrix44F(const float * mat)
{ memcpy(v, mat, 64); }

Matrix44F::Matrix44F(const double * mat)
{ for(int i = 0; i < 16; i++) v[i] = mat[i]; }

Matrix44F::Matrix44F(const Matrix33F & r, const Vector3F & t)
{
	setRotation(r);
	setTranslation(t);
	*m(0,3) = 0.f;
	*m(1,3) = 0.f;
	*m(2,3) = 0.f;
	*m(3,3) = 1.f;
}

Matrix44F::~Matrix44F() {}

/*
 *   layout
 *   row-major
 *   0  1  2  3
 *   4  5  6  7
 *   8  9  10 11
 *   12 13 14 15
 *
 *   row    i
 *   column j
 *
 *   00 01 02 03
 *   10 11 12 13
 *   20 21 22 23
 *   30 31 32 33
 *   row3 is translation part
 */
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
	Matrix44F r(*this);
	r.multiply(a);
	return r;
}

void Matrix44F::operator*= (const Matrix44F & a)
{
	multiply(a);
}

void Matrix44F::operator*= (const Matrix33F & a)
{
	Matrix33F t = rotation();
	t *= a;
	setRotation(t);
}

void Matrix44F::multiply(const Matrix44F & a)
{
	Matrix44F t(*this);
	setZero();
	int i, j, k;
	for(i = 0; i < 4; i++) {
		for(j = 0; j < 4; j++) {
			for(k = 0; k < 4; k++) {
				*m(i, j) += t.M(i, k) * a.M(k, j);
			}
		}
	}
}

const Matrix44F Matrix44F::transformBy(const Matrix44F & a) const
{
	Matrix44F t; 
	int i, j;
	for(i = 0; i < 4; i++) {
		for(j = 0; j < 4; j++) {
			*t.m(i, j) = M(i, 0) * a.M(0, j) + M(i, 1) * a.M(1, j) + M(i, 2) * a.M(2, j) + M(i, 3) * a.M(3, j);
		}
	}

	return t;
}

float* Matrix44F::m(int i, int j)
{
	return &v[i * 4 + j];
}

const float & Matrix44F::M(int i, int j) const
{
	return v[i * 4 + j];
}

void Matrix44F::setIdentity()
{
	setZero();
	*m(0, 0) = *m(1, 1) = *m(2, 2) = *m(3, 3) = 1.0f;
}

void Matrix44F::setZero()
{
	memset(v, 0, 64);
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
	float tx = p.x * M(0, 0) + p.y * M(1, 0) + p.z * M(2, 0) + M(3, 0);
	float ty = p.x * M(0, 1) + p.y * M(1, 1) + p.z * M(2, 1) + M(3, 1);
	float tz = p.x * M(0, 2) + p.y * M(1, 2) + p.z * M(2, 2) + M(3, 2);
		
	return Vector3F(tx, ty, tz);
}

Vector2F Matrix44F::transform(const Vector2F& p) const
{
	float tx = p.x* M(0, 0) + p.y* M(1, 0) + M(3, 0);
	float ty = p.x* M(0, 1) + p.y* M(1, 1) + M(3, 1);
		
	return Vector2F(tx, ty);
}

Vector3F Matrix44F::transformAsNormal(const Vector3F& p) const
{
	float tx = p.x * M(0, 0) + p.y * M(1, 0) + p.z * M(2, 0);
	float ty = p.x * M(0, 1) + p.y * M(1, 1) + p.z * M(2, 1);
	float tz = p.x * M(0, 2) + p.y * M(1, 2) + p.z * M(2, 2);
		
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

void Matrix44F::setRotation(const Matrix33F & r)
{
	*m(0, 0) = r.M(0, 0);
	*m(0, 1) = r.M(0, 1);
	*m(0, 2) = r.M(0, 2);
	*m(1, 0) = r.M(1, 0);
	*m(1, 1) = r.M(1, 1);
	*m(1, 2) = r.M(1, 2);
	*m(2, 0) = r.M(2, 0);
	*m(2, 1) = r.M(2, 1);
	*m(2, 2) = r.M(2, 2);
}

Matrix33F Matrix44F::rotation() const
{
	Matrix33F r;
	*r.m(0, 0) = M(0, 0);
	*r.m(0, 1) = M(0, 1);
	*r.m(0, 2) = M(0, 2);
	*r.m(1, 0) = M(1, 0);
	*r.m(1, 1) = M(1, 1);
	*r.m(1, 2) = M(1, 2);
	*r.m(2, 0) = M(2, 0);
	*r.m(2, 1) = M(2, 1);
	*r.m(2, 2) = M(2, 2);
	return r;
}

void Matrix44F::rotateX(float alpha)
{
	const float c = cos(alpha);
	const float s = sin(alpha);
	Matrix44F r;
	*r.m(1, 1) =  c; *r.m(1, 2) = s;
	*r.m(2, 1) = -s; *r.m(2, 2) = c;
	multiply(r);
}

void Matrix44F::rotateY(float beta)
{
	const float c = cos(beta);
	const float s = sin(beta);
	Matrix44F r;
	*r.m(0, 0) = c; *r.m(0, 2) = -s;
	*r.m(2, 0) = s; *r.m(2, 2) = c;
	multiply(r);
}

void Matrix44F::rotateZ(float gamma)
{
	const float c = cos(gamma);
	const float s = sin(gamma);
	Matrix44F r;
	*r.m(0, 0) =  c; *r.m(0, 1) = s;
	*r.m(1, 0) = -s; *r.m(1, 1) = c;
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

/*
 *  __ 01 02 03
 *  __ __ 12 13
 *  __ __ __ 23
 *  __ __ __ __
 */

void Matrix44F::transpose()
{
	float tmp= M(0, 1);
    *m(0, 1) = M(1, 0);
    *m(1, 0) = tmp;

    tmp =      M(0, 2);
    *m(0, 2) = M(2, 0);
    *m(2, 0) = tmp;
	
	tmp =      M(0, 3);
    *m(0, 3) = M(3, 0);
    *m(3, 0) = tmp;
	
    tmp =      M(1, 2);
    *m(1, 2) = M(2, 1);
    *m(2, 1) = tmp;
	
	tmp =      M(1, 3);
    *m(1, 3) = M(3, 1);
    *m(3, 1) = tmp;
	
	tmp =      M(2, 3);
    *m(2, 3) = M(3, 2);
    *m(3, 2) = tmp;
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

const float Matrix44F::determinant() const
{
    return  ( M(0, 0) * determinant33( M(1, 1), M(2, 1), M(3, 1), M(1, 2), M(2, 2), M(3, 2), M(1, 3), M(2, 3), M(3, 3) )
			- M(1, 0) * determinant33( M(0, 1), M(2, 1), M(3, 1), M(0, 2), M(2, 2), M(3, 2), M(0, 3), M(2, 3), M(3, 3) )
			+ M(2, 0) * determinant33( M(0, 1), M(1, 1), M(3, 1), M(0, 2), M(1, 2), M(3, 2), M(0, 3), M(1, 3), M(3, 3) )
			- M(3, 0) * determinant33( M(0, 1), M(1, 1), M(2, 1), M(0, 2), M(1, 2), M(2, 2), M(0, 3), M(1, 3), M(2, 3) ) );

}
// #include <iostream>
void Matrix44F::setRotation(const Quaternion & q)
{
	// std::cout<<"q("<<q.w<<" "<<q.x<<" "<<q.y<<" "<<q.z<<")"<<q.magnitude()<<"\n";
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

	*m(0, 0) = 1.0f - qyqy2 - qzqz2; 
	*m(0, 1) = qxqy2 + qzqw2; 
	*m(0, 2) = qxqz2 - qyqw2;
	 
	*m(1, 0) = qxqy2 - qzqw2; 
	*m(1, 1) = 1.0f - qxqx2 - qzqz2; 
	*m(1, 2) = qyqz2 + qxqw2;
	 
	*m(2, 0) = qxqz2 + qyqw2; 
	*m(2, 1) = qyqz2 - qxqw2; 
	*m(2, 2) = 1.0f - qxqx2 - qyqy2;
}

void Matrix44F::scaleBy(float sc)
{
	*m(0, 0) *= sc;
	*m(0, 1) *= sc;
	*m(0, 2) *= sc;
	*m(1, 0) *= sc;
	*m(1, 1) *= sc;
	*m(1, 2) *= sc;
	*m(2, 0) *= sc;
	*m(2, 1) *= sc;
	*m(2, 2) *= sc;
}

void Matrix44F::scaleTranslationBy(float sc)
{
	*m(3, 0) *= sc;
	*m(3, 1) *= sc;
	*m(3, 2) *= sc;
}

std::ostream& operator<<(std::ostream &output, const Matrix44F & p) 
{
	output << "["<<p.v[0]<<", "<<p.v[1]<<", "<<p.v[2]<<", "<<p.v[3]
		<<"]\n["<<p.v[4]<<", "<<p.v[5]<<", "<<p.v[6]<<", "<<p.v[7]
		<<"]\n["<<p.v[8]<<", "<<p.v[9]<<", "<<p.v[10]<<", "<<p.v[11]
		<<"]\n["<<p.v[12]<<", "<<p.v[13]<<", "<<p.v[14]<<", "<<p.v[15]<<"]";
	return output;
}

void Matrix44F::copy(const Matrix44F & another)
{ memcpy(v, another.v, 64); }

Vector3F Matrix44F::scale() const
{ return Vector3F(getSide().length(),
					getUp().length(),
					getFront().length() ); }
					
void Matrix44F::scaleBy(const Vector3F & scv)
{
	*m(0, 0) *= scv.x;
	*m(0, 1) *= scv.x;
	*m(0, 2) *= scv.x;
	*m(1, 0) *= scv.y;
	*m(1, 1) *= scv.y;
	*m(1, 2) *= scv.y;
	*m(2, 0) *= scv.z;
	*m(2, 1) *= scv.z;
	*m(2, 2) *= scv.z;
}

bool Matrix44F::isEqual(const Matrix44F & another) const
{
	for(int i=0;i<16;++i) {
		if(v[i] != another.v[i]) return false;
	}
	return true;
}

}
//:~
