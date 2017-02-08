/*
 *  BarycentricCoordinate.cpp
 *  hc
 *
 *  Created by jian zhang on 4/8/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include <math/BarycentricCoordinate.h>

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
	Vector3F bp = m_p[1] - m_onplane;
	Vector3F cp = m_p[2] - m_onplane;
	Vector3F ap = m_p[0] - m_onplane;
	
	float areaPBC = m_n.dot(bp.cross(cp));
	
	m_v[0] = areaPBC / m_area;

	float areaPCA = m_n.dot(cp.cross(ap));
	
	m_v[1] = areaPCA / m_area;
	
	float areaPAB = m_n.dot(ap.cross(bp));
	m_v[2] = areaPAB / m_area;;	
	
	computeInsideTriangle();
	
	if(m_isInsideTriangle) {
		m_closest = m_onplane;
	}
}

void BarycentricCoordinate::computeClosest()
{	
	Vector3F ba = m_p[1] - m_p[0];
	float la = ba.length();
	ba /= la;
	Vector3F na = ba.cross(m_n);
	
	Vector3F cb = m_p[2] - m_p[1];
	float lb = cb.length();
	cb /= lb;
	Vector3F nb = cb.cross(m_n);
	
	Vector3F ac = m_p[0] - m_p[2];
	float lc = ac.length();
	ac /= lc;
	Vector3F nc = ac.cross(m_n);
	
	Vector3F pa = m_onplane - getP(0);
	float da = pa.dot(na);
	m_onEdge[0] = m_onplane - na * da;
	
	Vector3F pb = m_onplane - getP(1);
	float db = pb.dot(nb);
	m_onEdge[1] = m_onplane - nb * db;
	
	Vector3F pc = m_onplane - getP(2);
	float dc = pc.dot(nc);
	m_onEdge[2] = m_onplane - nc * dc;
	
	float lpa = pa.length();
	float lpb = pb.length();
	float lpc = pc.length();
	
	float mindist = lpa;
/// to vertex
	m_closest = getP(0);
	m_v[0] = 1.f;
	m_v[1] = 0.f;
	m_v[2] = 0.f;
	
	if(lpb < mindist) {
		mindist = lpb;
		m_closest = getP(1);
		m_v[0] = 0.f;
		m_v[1] = 1.f;
		m_v[2] = 0.f;
	}
	if(lpc < mindist) {
		mindist = lpc;
		m_closest = getP(2);
		m_v[0] = 0.f;
		m_v[1] = 0.f;
		m_v[2] = 1.f;
	}
/// to edge
	const Vector3F v1e0 = m_onEdge[0] - getP(1);
	if(da > 0.f && da < mindist && v1e0.dot(m_onEdge[0] - getP(0)) < 0.f) {
		mindist = da;
		m_closest = m_onEdge[0];
		m_v[0] = v1e0.length() / la;
		m_v[1] = 1.f - m_v[0];
		m_v[2] = 0.f;
	}
	const Vector3F v2e1 = m_onEdge[1] - getP(2);
	if(db > 0.f && db < mindist && v2e1.dot(m_onEdge[1] - getP(1)) < 0.f) {
		mindist = db;
		m_closest = m_onEdge[1];
		m_v[0] = 0.f;
		m_v[1] = v2e1.length() / lb;
		m_v[2] = 1.f - m_v[1];
	}
	const Vector3F v0e2 = m_onEdge[2] - getP(0);
	if(dc > 0.f && dc < mindist && v0e2.dot(m_onEdge[2] - getP(2)) < 0.f) {
		mindist = dc;
		m_closest = m_onEdge[2];
		m_v[1] = 0.f;
		m_v[2] = v0e2.length() / lc;
		m_v[0] = 1.f - m_v[2];
	}
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
