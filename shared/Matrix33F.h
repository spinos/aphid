#pragma once
/*
 *  Matrix33F.h
 *
 *  Created by jian zhang on 10/30/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include "Vector3F.h"
#include "Quaternion.h"
namespace aphid {

 class Matrix33F
 {
 public:
	enum RotateOrder {
		Unknown = 0,
	    XYZ = 1,
		YZX = 2,
		ZXY = 3,
		XZY = 4,
		YXZ = 5,
		ZYX = 6
	};
	
	Matrix33F();
	Matrix33F(const Matrix33F & a);
	Matrix33F(const Vector3F& r0, const Vector3F & r1, const Vector3F & r2);
	~Matrix33F();
	float operator() (int i, int j);
	float operator() (int i, int j) const;
	Vector3F row(int i) const;
	void setRow(int i, const Vector3F & r);
	Vector3F operator*( Vector3F other ) const;
	void operator*= (float scaling);
	void operator+=(Matrix33F other);
	Matrix33F operator+( Matrix33F other ) const;
	Matrix33F operator*( Matrix33F other ) const;
	Matrix33F operator*( float scaling ) const;	
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
	
	Vector3F scale() const;
	void orthoNormalize();
	void set(const Quaternion & q);
	
	Vector3F eigenVector(float & lambda) const;
	Vector3F eigenValues() const;
	Matrix33F eigenSystem(Vector3F & values) const;
	float trace() const;
	bool isSymmetric() const;
	float distanceTo(const Matrix33F & another) const;
	static Vector3F SolveAxb(const Matrix33F & A, const Vector3F & b);
	
	static Matrix33F IdentityMatrix;
	
    friend std::ostream& operator<<(std::ostream &output, const Matrix33F & p) {
        output << p.str();
        return output;
    }
    
	const std::string Matrix33F::str() const;
	
	float v[9];
 };

}