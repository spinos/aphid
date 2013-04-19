/*
 *  BarycentricCoordinate.cpp
 *  hc
 *
 *  Created by jian zhang on 4/8/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "BarycentricCoordinate.h"
#include <iostream>
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

void BarycentricCoordinate::compute(const Vector3F & p)
{
	Vector3F pa = p - getP(0);
	
	Vector3F projp = p - m_n * pa.dot(m_n);
	m_onplane = projp;
	
	Vector3F bp = m_p[1] - projp;
	Vector3F cp = m_p[2] - projp;
	Vector3F ap = m_p[0] - projp;
	
	float areaPBC = m_n.dot(bp.cross(cp));
	
	m_v[0] = areaPBC / m_area;

	float areaPCA = m_n.dot(cp.cross(ap));
	
	m_v[1] = areaPCA / m_area;
	
	float areaPAB = m_n.dot(ap.cross(bp));
	m_v[2] = areaPAB / m_area;;	
	
	if(insideTriangle()) m_closest = m_onplane;
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
	
	Vector3F onEdge[3];
	
	Vector3F pa = m_onplane - getP(0);
	float da = pa.dot(na);
	onEdge[0] = m_onplane - na * da;
	
	Vector3F pb = m_onplane - getP(1);
	float db = pb.dot(nb);
	onEdge[1] = m_onplane - nb * db;
	
	Vector3F pc = m_onplane - getP(2);
	float dc = pc.dot(nc);
	onEdge[2] = m_onplane - nc * dc;
	
	float mindist = 10e6;
	if(pa.length() < mindist) {
		mindist = pa.length();
		m_closest = getP(0);
		m_v[0] = 1.f;
		m_v[1] = 0.f;
		m_v[2] = 0.f;
	}
	if(pb.length() < mindist) {
		mindist = pb.length();
		m_closest = getP(1);
		m_v[0] = 0.f;
		m_v[1] = 1.f;
		m_v[2] = 0.f;
	}
	if(pc.length() < mindist) {
		mindist = pc.length();
		m_closest = getP(2);
		m_v[0] = 0.f;
		m_v[1] = 0.f;
		m_v[2] = 1.f;
	}

	if(da > 0.f && da < mindist && (onEdge[0] - getP(1)).dot(onEdge[0] - getP(0)) < 0.f) {
		mindist = da;
		m_closest = onEdge[0];
		m_v[0] = (onEdge[0] - getP(1)).length() / la;
		m_v[1] = 1.f - m_v[0];
		m_v[2] = 0.f;
	}
	
	if(db > 0.f && db < mindist && (onEdge[1] - getP(2)).dot(onEdge[1] - getP(1)) < 0.f) {
		mindist = db;
		m_closest = onEdge[1];
		m_v[0] = 0.f;
		m_v[1] = (onEdge[1] - getP(2)).length() / lb;
		m_v[2] = 1.f - m_v[1];
	}
	
	if(dc > 0.f && dc < mindist && (onEdge[2] - getP(0)).dot(onEdge[2] - getP(2)) < 0.f) {
		mindist = dc;
		m_closest = onEdge[2];
		m_v[1] = 0.f;
		m_v[2] = (onEdge[2] - getP(0)).length() / lc;
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

char BarycentricCoordinate::insideTriangle() const
{
	if(m_v[0] < 0.f || m_v[0] > 1.f ) return 0;
	if(m_v[1] < 0.f || m_v[1] > 1.f ) return 0;
	if(m_v[2] < 0.f || m_v[2] > 1.f ) return 0;
	return 1;
}

Vector3F BarycentricCoordinate::getClosest() const
{
	return m_closest;
}

Vector3F BarycentricCoordinate::getOnPlane() const
{
	return m_onplane;
}
