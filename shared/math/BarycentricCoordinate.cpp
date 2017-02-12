/*
 *  BarycentricCoordinate.cpp
 *  hc
 *
 *  Created by jian zhang on 4/8/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include <math/BarycentricCoordinate.h>
#include <line_math.h>

namespace aphid {

inline float barycentric_coord(float ax, float ay, float bx, float by, float x, float y)
{
	return (ay - by)*x + (bx - ax)*y +ax*by - bx*ay;
}

BarycentricCoordinate::BarycentricCoordinate() {}

void BarycentricCoordinate::create(const Vector3F& a, const Vector3F& b, const Vector3F& c)
{
	m_p[0] = a;
	m_p[1] = b;
	m_p[2] = c;
	
	Vector3F ba = b - a;
	Vector3F ca = c - a;
	m_n = ba.cross(ca);
	m_n.normalize();

	m_area = m_n.dot(ba.cross(ca));
}

float BarycentricCoordinate::project(const Vector3F & pos)
{
	Vector3F pa = pos - getP(0);
	
	float t = pa.dot(m_n);
	m_onplane = pos - m_n * t;
	
	if(t < 0.f) {
		t = -t;
	}
	return t;
}

void BarycentricCoordinate::compute()
{
	computeContribute(m_onplane);	
	computeInsideTriangle();
	
	if(m_isInsideTriangle) {
		m_closest = m_onplane;
	}
}

void BarycentricCoordinate::computeClosest()
{	
	float lpva = m_onplane.distanceTo(getP(0) );
	float minD = lpva;
	m_closest = getP(0);
	m_v[0] = 1.f;
	m_v[1] = m_v[2] = 0.f;
	
	float lpvb = m_onplane.distanceTo(getP(1) );
	if(minD > lpvb) {
		minD = lpvb;
		m_closest = getP(1);
		m_v[1] = 1.f;
		m_v[0] = m_v[2] = 0.f;
	}
	
	float lpvc = m_onplane.distanceTo(getP(2) );
	if(minD > lpvc) {
		minD = lpvc;
		m_closest = getP(2);
		m_v[2] = 1.f;
		m_v[0] = m_v[1] = 0.f;
	}
	
	float lpea;
	if(distancePointLineSegment(lpea, m_onplane, getP(0), getP(1) ) ) {
		if(minD > lpea) {
			minD = lpea;
			projectPointLineSegment(m_closest, lpea, m_onplane, getP(0), getP(1) );
			m_v[0] = getP(1).distanceTo(m_closest) / getP(1).distanceTo(getP(0) );
			m_v[1] = 1.f - m_v[0];
			m_v[2] = 0.f;
		}
	}
	
	float lpeb;
	if(distancePointLineSegment(lpeb, m_onplane, getP(1), getP(2) ) ) {
		if(minD > lpeb) {
			minD = lpeb;
			projectPointLineSegment(m_closest, lpeb, m_onplane, getP(1), getP(2) );
			m_v[1] = getP(2).distanceTo(m_closest) / getP(2).distanceTo(getP(1) );
			m_v[2] = 1.f - m_v[1];
			m_v[0] = 0.f;
		}
	}
	
	float lpec;
	if(distancePointLineSegment(lpec, m_onplane, getP(2), getP(0) ) ) {
		if(minD > lpec) {
			minD = lpec;
			projectPointLineSegment(m_closest, lpec, m_onplane, getP(2), getP(0) );
			m_v[2] = getP(0).distanceTo(m_closest) / getP(0).distanceTo(getP(2) );
			m_v[0] = 1.f - m_v[2];
			m_v[1] = 0.f;
		}
	}
}

void BarycentricCoordinate::computeContribute(const Vector3F & q)
{
	Vector3F bp = m_p[1] - q;
	Vector3F cp = m_p[2] - q;
	Vector3F ap = m_p[0] - q;
	
	float areaPBC = m_n.dot(bp.cross(cp));
	
	m_v[0] = areaPBC / m_area;

	float areaPCA = m_n.dot(cp.cross(ap));
	
	m_v[1] = areaPCA / m_area;
	
	float areaPAB = m_n.dot(ap.cross(bp));
	m_v[2] = areaPAB / m_area;
}

const float * BarycentricCoordinate::getValue() const
{
	return m_v;
}

Vector3F BarycentricCoordinate::getP(unsigned idx) const
{
	return m_p[idx];
}

float BarycentricCoordinate::getV(unsigned idx) const
{
	return m_v[idx];
}

void BarycentricCoordinate::computeInsideTriangle()
{
	m_isInsideTriangle = false;
		
	if(m_v[0] < 0.f || m_v[0] > 1.f ) {
		return;
	}
	
	if(m_v[1] < 0.f || m_v[1] > 1.f ) {
		return;
	}

	if(m_v[2] < 0.f || m_v[2] > 1.f ) {
		return;
	}

	m_isInsideTriangle = true;
}

Vector3F BarycentricCoordinate::getClosest() const
{
	return m_closest;
}

Vector3F BarycentricCoordinate::getOnPlane() const
{
	return m_onplane;
}

Vector3F BarycentricCoordinate::getNormal() const
{
	return m_n;
}

const bool & BarycentricCoordinate::insideTriangle() const
{ return m_isInsideTriangle; }

}
//:~
