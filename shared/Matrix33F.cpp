/*
 *  Matrix33F.cpp
 *
 *  Created by jian zhang on 10/30/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include <cmath>
#include "Matrix33F.h"
#include <iostream>

Matrix33F Matrix33F::IdentityMatrix;

Matrix33F::Matrix33F() 
{
	setIdentity();
}

Matrix33F::Matrix33F(const Matrix33F & a)
{
	for(int i = 0; i < 9; i++) v[i] = a.v[i];
}

Matrix33F::Matrix33F(const Vector3F& r0, const Vector3F & r1, const Vector3F & r2)
{
    fill(r0, r1, r2);
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

Vector3F Matrix33F::row(int i) const
{ return Vector3F(v[i*3], v[i*3 + 1], v[i*3 + 2]); }

Vector3F Matrix33F::operator*( Vector3F other ) const
{
	Vector3F v;
	v.x = M(0, 0) * other.x + M(0, 1) * other.y + M(0, 2) * other.z;
	v.y = M(1, 0) * other.x + M(1, 1) * other.y + M(1, 2) * other.z;
	v.z = M(2, 0) * other.x + M(2, 1) * other.y + M(2, 2) * other.z;
	
	return v;
}

Matrix33F Matrix33F::operator*( Matrix33F other ) const
{
    return multiply(other);
}

Matrix33F Matrix33F::operator*( float scaling ) const
{
    Matrix33F t(*this);
    t *= scaling;
    return t;
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

void Matrix33F::operator*= (float scaling)
{
    for(int i = 0; i < 9; i++) v[i] *= scaling;
}

void Matrix33F::operator+=(Matrix33F other)
{
    for(int i = 0; i < 9; i++) v[i] += other.v[i];
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

/*
   i
   00 01 02
   10 11 12
   20 21 22
           j
*/

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

Matrix33F Matrix33F::multiply(const Matrix33F & a) const
{
	Matrix33F t;
	t.setZero();
	int i, j, k;
	for(i = 0; i < 3; i++) {
		for(j = 0; j < 3; j++) {
			for(k = 0; k < 3; k++) {
				*t.m(i, j) += M(i, k) * a.M(k, j);
			}
		}
	}
	return t;
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
 *  | a00 a01 a02 |-1  
 *  | a10 a11 a12 |    
 *  | a20 a21 a22 |    
 *  
 *  | +  -  + |
 *  | -  +  - |
 *  | +  -  + |
 *  
 *  http://www.mathsisfun.com/algebra/matrix-inverse-minors-cofactors-adjugate.html
 *
 */

void Matrix33F::inverse()
{
    const float det = determinant();
    
    const float m00 =  determinant22(M(1, 1), M(1, 2), M(2, 1), M(2, 2));
    const float m01 = -determinant22(M(1, 0), M(1, 2), M(2, 0), M(2, 2));
    const float m02 =  determinant22(M(1, 0), M(1, 1), M(2, 0), M(2, 1));
	
	const float m10 = -determinant22(M(0, 1), M(0, 2), M(2, 1), M(2, 2));
	const float m11 =  determinant22(M(0, 0), M(0, 2), M(2, 0), M(2, 2));
	const float m12 = -determinant22(M(0, 0), M(0, 1), M(2, 0), M(2, 1));
	
	const float m20 =  determinant22(M(0, 1), M(0, 2), M(1, 1), M(1, 2));
	const float m21 = -determinant22(M(0, 0), M(0, 2), M(1, 0), M(1, 2));
	const float m22 =  determinant22(M(0, 0), M(0, 1), M(1, 0), M(1, 1));
    
    *m(0, 0) = m00 / det;
    *m(0, 1) = m10 / det;
    *m(0, 2) = m20 / det;
	
	*m(1, 0) = m01 / det;
	*m(1, 1) = m11 / det;
	*m(1, 2) = m21 / det;
	
	*m(2, 0) = m02 / det;
	*m(2, 1) = m12 / det;
	*m(2, 2) = m22 / det;
}

/*   
 *  | a b c |
 *  | d e f |
 *  | g h i |
 *
 *  det33 = a*( e*i - h*f ) - b*( d*i - g*f ) + c*( d*h - g*e )
 */

float Matrix33F::determinant() const
{
    return M(0, 0) * determinant22(M(1, 1), M(1, 2), M(2, 1), M(2, 2)) 
            - M(0, 1) * determinant22(M(1, 0), M(1, 2), M(2, 0), M(2, 2)) 
            + M(0, 2) * determinant22(M(1, 0), M(1, 1), M(2, 0), M(2, 1));
}

/*  |a b|
 *  |c d|
 *  
 *  det22 = ad - bc
 */
 
float Matrix33F::determinant22(float a, float b, float c, float d) const
{
    return a * d - b * c;
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

void Matrix33F::rotateEuler(float phi, float theta, float psi, RotateOrder order)
{
	Matrix33F B;
	const float cpsi = cos(phi);
	const float spsi = sin(phi);
	*B.m(1, 1) = cpsi;
	*B.m(1, 2) = spsi;
	*B.m(2, 1) = -spsi;
	*B.m(2, 2) = cpsi;
	
	const float ctheta = cos(theta);
	const float stheta = sin(theta);
	Matrix33F C;
	*C.m(0, 0) = ctheta;
	*C.m(0, 2) = -stheta;
	*C.m(2, 0) = stheta;
	*C.m(2, 2) = ctheta;
	
	const float cphi = cos(psi);
	const float sphi = sin(psi);
	Matrix33F D;
	*D.m(0, 0) = cphi;
	*D.m(0, 1) = sphi;
	*D.m(1, 0) = -sphi;
	*D.m(1, 1) = cphi;
	
	switch(order) {
	    case XYZ:
	        multiply(B);
	        multiply(C);
	        multiply(D);
	        break;
		case YZX:
		    multiply(C);
		    multiply(D);
		    multiply(B);
		    break;
		case ZXY:
		    multiply(D);
		    multiply(B);
		    multiply(C);
		    break;
		case XZY:
		    multiply(B);
		    multiply(D);
		    multiply(C);
		    break;
		case YXZ:
		    multiply(C);
		    multiply(B);
		    multiply(D);
		    break;
		case ZYX:
		    multiply(D);
		    multiply(C);
		    multiply(B);
		    break;
		default:
		    break;
	}
}

Vector3F Matrix33F::scale() const
{
	Vector3F vx(M(0, 0), M(0, 1), M(0, 2));
	Vector3F vy(M(1, 0), M(1, 1), M(1, 2));
	Vector3F vz(M(2, 0), M(2, 1), M(2, 2));
	return Vector3F(vx.length(), vy.length(), vz.length());
}

void Matrix33F::orthoNormalize()
{
// The Gram–Schmidt process
// reference https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
    Vector3F r0(M(0, 0), M(0, 1), M(0, 2));
    Vector3F r1(M(1, 0), M(1, 1), M(1, 2));
    Vector3F r2(M(2, 0), M(2, 1), M(2, 2));
    
    float l0 = r0.length();
    if(l0 > 0.f) r0 /= l0;
    
    r1 -= r0 * r0.dot(r1);
    float l1 = r1.length();
    if(l1 > 0.f) r1 /= l1;
    
    r2 = r0.cross(r1);
    fill(r0, r1, r2);
}

void Matrix33F::set(const Quaternion & q)
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

Vector3F Matrix33F::eigenVector(float & lambda) const
{
// power iterative method finds dominant eigen pair
// reference http://www.math.pitt.edu/~sussmanm/2071Spring09/lab08/index.html
// reference http://mathfaculty.fullerton.edu/mathews/n2003/PowerMethodMod.html
	
	Vector3F bk(0.f, 0.f, 1.f);
	Vector3F bk1;
	float ck = 1.f;
	Vector3F yk;
	
	int i = 0;
	for(;i<100;i++) {
		yk = (*this) * bk;
		ck = yk.length();
		bk1 = yk / ck;

		if(bk1.distanceTo(bk)<1e-6f) break;
		bk = bk1;
	}
	
	lambda = ck;
	return bk1;
}

float Matrix33F::trace() const
{ return v[0] + v[4] + v[8]; }

bool Matrix33F::isSymmetric() const
{
	if(v[1] != v[3]) return false;
	if(v[2] != v[6]) return false;
	if(v[5] != v[7]) return false;
	return true;
}

Vector3F Matrix33F::eigenValues() const
{
// when A is symmetric 
// reference https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3.C3.973_matrices

	float p1 = v[1] * v[1] + v[2] * v[2] + v[5] * v[5];
	if(p1 < 1e-7f) return Vector3F(v[0], v[4], v[8]);
	
	float q = trace() / 3.f;
	float p2 = (M(0,0) - q) * (M(0,0) - q)
				+ (M(1,1) - q) * (M(1,1) - q)
				+ (M(2,2) - q) * (M(2,2) - q)
				+ 2.f * p1;
	float p = sqrt(p2 / 6.f);
	Matrix33F B = *this + IdentityMatrix * (-q);
	B *= 1.f / p;
	float r = B.determinant() * .5f;
	
	float phi;
	if(r <= -1.f) phi = 3.141592653589f / 3.f;
	else if(r>= 1.f) phi = 0.f;
	else phi = acos(r) / 3.f;
	
	float eig1 = q + 2.f * p * cos(phi);
	float eig3 = q + 2.f * p * cos(phi + (2* 3.141592653589f/3.f));
	float eig2 = 3.f * q - eig1 - eig3;

	return Vector3F(eig1, eig2, eig3);
}

Matrix33F Matrix33F::eigenSystem(Vector3F & values) const
{
	values = eigenValues();
	
	Matrix33F Vk;
	Matrix33F Vk1;
	Matrix33F A(*this);
	int i = 0;
	for(;i<50;i++) {
		Vk1 = Vk * A;
		Vk1.orthoNormalize();
		
		if(Vk.distanceTo(Vk1)<1e-9f) break;
		Vk = Vk1;
	}
	
	Vector3F vx(Vk1.v[0], Vk1.v[1], Vk1.v[2]);
	values.x = (A * vx).length();
	Vector3F vy(Vk1.v[3], Vk1.v[4], Vk1.v[5]);
	values.y = (A * vy).length();
	Vector3F vz(Vk1.v[6], Vk1.v[7], Vk1.v[8]);
	values.z = (A * vz).length();
	return Vk1;
}

float Matrix33F::distanceTo(const Matrix33F & another) const
{
	float r = 0.f;
	float d;
	int i=0;
	for(;i<9;i++) {
		d = v[i] - another.v[i];
		r += d*d;
	}
	return r;
}

const std::string Matrix33F::str() const
{
	std::stringstream sst;
	sst.str("\n");
    sst<<"["<<v[0]<<", "<<v[1]<<", "<<v[2]<<"]\n";
    sst<<"["<<v[3]<<", "<<v[4]<<", "<<v[5]<<"]\n";
	sst<<"["<<v[6]<<", "<<v[7]<<", "<<v[8]<<"]\n";
	
	return sst.str();
}

Vector3F Matrix33F::SolveAxb(const Matrix33F & A, const Vector3F & b)
{
// gaussian elimination
// reference http://mathworld.wolfram.com/GaussianElimination.html

	float m[3][4];
	m[0][0] = A.v[0]; m[0][1] = A.v[1]; m[0][2] = A.v[2]; m[0][3] = b.x;
	m[1][0] = A.v[3]; m[1][1] = A.v[4]; m[1][2] = A.v[5]; m[1][3] = b.y;
	m[2][0] = A.v[6]; m[2][1] = A.v[7]; m[2][2] = A.v[8]; m[2][3] = b.z;
	
	float t;
	if(m[0][0] < 1e-3f && m[0][0] > 1e-3f) {
		t = m[2][0];
		m[2][0] = m[0][0];
		m[0][0] = t;
		
		t = m[2][1];
		m[2][1] = m[0][1];
		m[0][1] = t;
		
		t = m[2][2];
		m[2][2] = m[0][2];
		m[0][2] = t;
		
		t = m[2][3];
		m[2][3] = m[0][3];
		m[0][3] = t;
	}
	
	int i;
	t = m[1][0] / m[0][0];
	for(i=0;i<4;i++) m[1][i] -= m[0][i] * t;
	
	t = m[2][0] / m[0][0];
	for(i=0;i<4;i++) m[2][i] -= m[0][i] * t;
	
	t = m[2][1] / m[1][1];
	for(i=1;i<4;i++) m[2][i] -= m[1][i] * t;
	
	std::cout<<" arg m\n["<<m[0][0]<<","<<m[0][1]<<","<<m[0][2]<<","<<m[0][3]<<"]"
					<<"\n["<<m[1][0]<<","<<m[1][1]<<","<<m[1][2]<<","<<m[1][3]<<"]"
					<<"\n["<<m[2][0]<<","<<m[2][1]<<","<<m[2][2]<<","<<m[2][3]<<"]";
					
	Vector3F sol;
	sol.z =	  m[2][3]	/ m[2][2];
	sol.y = ( m[1][3]                   - sol.z * m[1][2] ) / m[1][1];
	sol.x = ( m[0][3] - sol.y * m[0][1] - sol.z * m[0][2] ) / m[0][0];
	return sol;
}
//:~