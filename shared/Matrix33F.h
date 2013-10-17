#pragma once
/*
 *  Matrix33F.h
 *
 *  Created by jian zhang on 10/30/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include "Vector3F.h"
 class Matrix33F
 {
 public:
	enum RotateOrder {
		XYZ = 0,
		ZYX = 1
	};
	
	Matrix33F();
	Matrix33F(const Matrix33F & a);
	~Matrix33F();
	float Matrix33F::operator() (int i, int j);
	float Matrix33F::operator() (int i, int j) const;
	Vector3F operator*( Vector3F other ) const;
	Matrix33F operator+( Matrix33F other ) const;	
	void setIdentity();
	void setZero();
	void inverse();
	float* m(int i, int j);
	float M(int i, int j) const;
	
	void fill(const Vector3F & a, const Vector3F & b, const Vector3F & c);
	void transpose();
	void multiply(const Matrix33F & a);
	Matrix33F multiply(const Matrix33F & a) const;
	Vector3F transform(const Vector3F & a) const;
	void glMatrix(float m[16]) const;
	float determinant() const;
	float determinant22(float a, float b, float c, float d) const;
	
	void rotateX(float alpha);
	void rotateY(float beta);
	void rotateZ(float gamma);
	void rotateEuler(float phi, float theta, float psi, RotateOrder order = XYZ);
	
	float v[9];
 };

