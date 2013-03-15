#pragma once
/*
 *  Matrix33F.h
 *
 *  Created by jian zhang on 10/30/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
 class Vector3F;
 class Matrix33F
 {
 public:
	Matrix33F();
	~Matrix33F();
	float Matrix33F::operator() (int i, int j);
	float Matrix33F::operator() (int i, int j) const;
	Vector3F operator*( Vector3F other ) const;
	Matrix33F operator+( Matrix33F other ) const;	
	void setIdentity();
	
	float* m(int i, int j);
	float M(int i, int j) const;
	
	void fill(const Vector3F & a, const Vector3F & b, const Vector3F & c);
	void transpose();
	Matrix33F multiply(const Matrix33F & a) const;
	Vector3F transform(const Vector3F & a) const;
	float v[9];
 };

