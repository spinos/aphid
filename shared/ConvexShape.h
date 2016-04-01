/*
 *  ConvexShape.h
 *  
 *
 *  Created by jian zhang on 11/4/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include "Matrix44F.h"
#include "BoundingBox.h"
#include "Ray.h"
#include "GjkIntersection.h"
#include "BarycentricCoordinate.h"

namespace aphid {
    
namespace cvx {
    
    enum ShapeType {
        TUnknown = 0,
        TSphere = 1,
        TCube = 2,
        TCapsule = 3,
		TTriangle = 4
    };

class Frustum {

	/// 1-------0
	/// |       |
	/// |  far  |
	/// 2-------3
	///
	/// 5-------4
	/// |       |
	/// |  near |
	/// 6-------7
	///
	Vector3F m_corners[8];
	
public:
	Frustum();
	
	void set(float nearClip, float farClip,
			float horizontalAperture, float verticalAperture,
			float angleOfView,
			const Matrix44F & space);
			
	Vector3F * x();
	Vector3F X(int idx) const;
	int numPoints() const;
	Vector3F supportPoint(const Vector3F & v, Vector3F * localP = NULL) const;

	/// 2----p1 p1---3 
	/// |     | |    |
	/// |     | |    |
	/// 0----p0 p0---1
	///
	/// 6----p3 p3---7
	/// |     | |    |
	/// |     | |    |
	/// 4----p2 p2---5
	///
	/// 2------------3 
	/// |            |
	/// p0-----------p1
	/// p0-----------p1 
	/// |            |
	/// 0------------1
	///
	/// 6------------7
	/// |            |
	/// p2-----------p3
	/// p2-----------p3
	/// |            |
	/// 4------------5
	///
	void split(Frustum & child0, Frustum & child1, float alpha, bool alongX = true) const;
	
protected:

private:

};

class Sphere {
  
    Vector3F m_p;
    float m_r;
    
public:
    Sphere();
    void set(const Vector3F & x, const float & r);
    
    BoundingBox calculateBBox() const;
    
    static ShapeType ShapeTypeId;
	static std::string GetTypeStr();
	
};

class Cube {
    
    Vector3F m_p;
    float m_r;
    
public:
    Cube();
    void set(const Vector3F & x, const float & r);
    
    BoundingBox calculateBBox() const;
	bool intersect(const Ray &ray, float *hitt0, float *hitt1) const;
	
	template<typename T>
	bool exactIntersect(const T & b) const {
		return true;
	}
    
    static ShapeType ShapeTypeId;
	static std::string GetTypeStr();
};

class Capsule {
    
    Vector3F m_p0;
    float m_r0;
    Vector3F m_p1;
    float m_r1;
    
public:
    Capsule();
    void set(const Vector3F & x0, const float & r0,
            const Vector3F & x1, const float & r1);
    
    BoundingBox calculateBBox() const;
    
    static ShapeType ShapeTypeId;
};

class Triangle {

	Vector3F m_p0; int m_nc0;
	Vector3F m_p1; int m_nc1;
	Vector3F m_p2; int m_nc2;

public:
	Triangle();
	void setP(const Vector3F & p, const int & idx);
	void resetNC();
	void setN(const Vector3F & n, const int & idx);
	void setC(const Vector3F & c, const int & idx);
	void setInd(const int & x, const int & idx);
	
	const Vector3F * p(int idx) const;
	const Vector3F & P(int idx) const;
	Vector3F N(int idx) const;
	const int & ind0() const;
	const int & ind1() const;
	Vector3F calculateNormal() const;
	void translate(const Vector3F & v);

	BoundingBox calculateBBox() const;
	bool intersect(const Ray &ray, float *hitt0, float *hitt1) const;
	
	template<typename T>
	bool exactIntersect(const T & b) const {
	
		return gjk::Intersect1<Triangle, T>::Evaluate(*this, b);
	}
	
	template<typename T>
	void closestToPoint(T * result) const;
	
	const Vector3F & X(int idx) const;
	const Vector3F & supportPoint(const Vector3F & v, Vector3F * localP = NULL) const;
	
	static ShapeType ShapeTypeId;
	static std::string GetTypeStr();
	
private:
	
};

template<typename T>
void Triangle::closestToPoint(T * result) const
{
	BarycentricCoordinate & bar = result->_bar;
	bar.create(m_p0, m_p1, m_p2);
	float d = bar.project(result->_toPoint);
	if(d>=result->_distance) return;
	bar.compute();
	if(!bar.insideTriangle()) bar.computeClosest();
	
	Vector3F clampledP = bar.getClosest();
	d = (clampledP - result->_toPoint).length();
	if(d>=result->_distance) return;
	
	result->_distance = d;
	result->_hasResult = true;
	result->_hitPoint = clampledP;
	result->_contributes[0] = bar.getV(0);
	result->_contributes[1] = bar.getV(1);
	result->_contributes[2] = bar.getV(2);
	result->_hitNormal = bar.getNormal();
	result->_igeometry = ind0();
	result->_icomponent = ind1();
}

}

}