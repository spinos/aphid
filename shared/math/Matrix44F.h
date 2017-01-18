#ifndef APH_MATRIX_44_F_H
#define APH_MATRIX_44_F_H

/*
 *  Matrix44F.h
 *
 *  Created by jian zhang on 10/30/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include <math/Matrix33F.h>

namespace aphid {

 class Matrix44F
 {
 public:
	Matrix44F();
	Matrix44F(float x);
	Matrix44F(const Matrix44F & a);
	Matrix44F(const float * mat);
    Matrix44F(const double * mat);
	Matrix44F(const Matrix33F & r, const Vector3F & t);
	~Matrix44F();
	float operator() (int i, int j);
	float operator() (int i, int j) const;
	Matrix44F operator* (const Matrix44F & a) const;
	void operator*= (const Matrix44F & a);
	void multiply(const Matrix44F & a);
	void setIdentity();
	void setZero();
	float* m(int i, int j);
	const float & M(int i, int j) const;
	
	float determinant33( float a, float b, float c, float d, float e, float f, float g, float h, float i ) const;
	void inverse();
	Vector3F transform(const Vector3F& p) const;
	Vector3F transform(const Vector3F& p);
	Vector3F transformAsNormal(const Vector3F& p) const;
	Vector3F transformAsNormal(const Vector3F& p);
	const Matrix44F transformBy(const Matrix44F & a) const;
	void translate(const Vector3F& p);
	void translate(float x, float y, float z);
	void setTranslation(const Vector3F& p);
	void setTranslation(float x, float y, float z);
	void setOrientations(const Vector3F& side, const Vector3F& up, const Vector3F& front);
	void setFrontOrientation(const Vector3F& front);
	void setRotation(const Matrix33F & r);
	void setRotation(const Quaternion & q);
	Matrix33F rotation() const;
	void rotateX(float alpha);
	void rotateY(float beta);
	void rotateZ(float gamma);
	Vector3F getTranslation() const;
	Vector3F getSide() const;
	Vector3F getUp() const;
	Vector3F getFront() const;
	void transposed(float * mat) const;
	void transpose();
	void glMatrix(float *m) const;
	const float determinant() const;
/// 3x3 only
	void scaleBy(float sc);
	void scaleTranslationBy(float sc);
	void copy(const Matrix44F & another);
	
	static float Determinant33( float a, float b, float c, float d, float e, float f, float g, float h, float i );
	static Matrix44F IdentityMatrix;
	
	friend std::ostream& operator<<(std::ostream &output, const Matrix44F & p);

	float v[16];
 };
 
}
#endif        //  #ifndef MATRIX44F_H

