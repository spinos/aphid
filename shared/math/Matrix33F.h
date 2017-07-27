#ifndef APH_MATRIX_33_F_H
#define APH_MATRIX_33_F_H
/*
 *  Matrix33F.h
 *
 *  row    i
 *  column j
 *  | 0 1 2 |
 *  | 3 4 5 |
 *  | 6 7 8 |
 *  index
 *  | 00 01 02 | 
 *  | 10 11 12 |    
 *  | 20 21 22 |  
 *
 *  Created by jian zhang on 10/30/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include <math/Quaternion.h>

namespace aphid {

 class Matrix33F
 {
	float v[9];
	
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
	Matrix33F(const Quaternion & q);
	~Matrix33F();
/// ith-row jth-column
	float operator() (int i, int j);
	float operator() (int i, int j) const;
	Vector3F row(int i) const;
	void setRow(int i, const Vector3F & r);
	Vector3F operator*( Vector3F other ) const;
	void operator*= (float scaling);
	void operator*= (const Matrix33F & another);
	void operator+=(Matrix33F other);
	Matrix33F operator+( const Matrix33F& other ) const;
	Matrix33F operator-( const Matrix33F& other ) const;
	Matrix33F operator*( Matrix33F other ) const;
	Matrix33F operator*( float scaling ) const;	
	void setIdentity();
	void setZero();
	void inverse();
	float* m(int i, int j);
	float M(int i, int j) const;
	
	void fill(const Vector3F & a, const Vector3F & b, const Vector3F & c);
	void transpose();
	Matrix33F transposed() const;
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
	void getSide(Vector3F & dst) const;
	void getUp(Vector3F & dst) const;
	void getFront(Vector3F & dst) const;
	void copy(const Matrix33F & another);
	void asCrossProductMatrix(const Vector3F& a);

	static Vector3F SolveAxb(const Matrix33F & A, const Vector3F & b);
	
	static Matrix33F IdentityMatrix;
	
    friend std::ostream& operator<<(std::ostream &output, const Matrix33F & p);
    
	
 };

}
#endif