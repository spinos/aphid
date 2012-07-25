#ifndef MATRIX44F_H
#define MATRIX44F_H

/*
 *  Matrix44F.h
 *  easymodel
 *
 *  Created by jian zhang on 10/30/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
 class Vector3F;
 class Matrix44F
 {
 public:
	Matrix44F();
	~Matrix44F();
	float Matrix44F::operator() (int i, int j);
	float Matrix44F::operator() (int i, int j) const;
	void setIdentity();
	float* m(int i, int j);
	float M(int i, int j) const;
	
	float determinant33( float a, float b, float c, float d, float e, float f, float g, float h, float i ) const;
	void inverse();
	Vector3F transform(const Vector3F& p) const;
	Vector3F transform(const Vector3F& p);
	Vector3F transformAsNormal(const Vector3F& p) const;
	Vector3F transformAsNormal(const Vector3F& p);
	void translate(const Vector3F& p);
	void translate(float x, float y, float z);
	void setTranslation(const Vector3F& p);
	void setTranslation(float x, float y, float z);
	void setOrientations(const Vector3F& side, const Vector3F& up, const Vector3F& front);
	Vector3F getTranslation() const;
	Vector3F getSide() const;
	Vector3F getUp() const;
	Vector3F getFront() const;
	void transposed(float * mat) const;
	float v[16];
 };
 

#endif        //  #ifndef MATRIX44F_H

