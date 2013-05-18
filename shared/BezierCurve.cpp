/*
 *  BezierCurve.cpp
 *  knitfabric
 *
 *  Created by jian zhang on 5/18/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "BezierCurve.h"

BezierCurve::BezierCurve() {}
BezierCurve::~BezierCurve() {}

Vector3F BezierCurve::interpolate(float param, Vector3F * data)
{
	m_t = 0.f;
	m_k0 = 0;
	m_k1 = numVertices() - 1;
	
	if(param <= 0.f) {
		m_k1 = m_k0;
		m_k11 = m_k0;
		m_k00 = m_k0;
		return data[m_k0];
	}
	
	if(param >= 1.f) {
		m_k0 = m_k1;
		m_k11 = m_k1;
		m_k00 = m_k1;
		return data[m_k1];
	}
	
	findNeighborKnots(param, m_k0, m_k1);
	
	fourControlKnots();
	calculateT(param);
	
	return calculateBezierPoint(m_t, m_k00, m_k0, m_k1, m_k11, data);
}

Vector3F BezierCurve::interpolateByKnot(float param, Vector3F * data)
{
	m_t = 0.f;
	m_k0 = 0;
	m_k1 = numVertices() - 1;
	
	if(param <= m_k0) {
		m_k1 = m_k0;
		m_k11 = m_k0;
		m_k00 = m_k0;
		return data[m_k0];
	}
	
	if(param >= m_k1) {
		m_k0 = m_k1;
		m_k11 = m_k1;
		m_k00 = m_k1;
		return data[m_k1];
	}
	
	m_k0 = (unsigned)param;
	m_k1 = m_k0 + 1;
	float realparam = m_knots[m_k0] * (1.f - (param - m_k0)) + m_knots[m_k1] * (param - m_k0);
	
	fourControlKnots();
	calculateT(realparam);
	
	return calculateBezierPoint(m_t, m_k00, m_k0, m_k1, m_k11, data);
}

void BezierCurve::fourControlKnots()
{
	int k00 = m_k0 - 1;
	if(k00 < 0) k00 = 0;
	
	m_k00 = k00;
	
	m_k11 = m_k1 + 1;
	if(m_k11 > numVertices() - 1) m_k11 = numVertices() - 1;
}

void BezierCurve::calculateT(float param)
{
	m_t = (param - m_knots[m_k00]) / (m_knots[m_k11] - m_knots[m_k00]);

}

Vector3F BezierCurve::calculateBezierPoint(float t, unsigned k00, unsigned k0, unsigned k1, unsigned k11, Vector3F * data) const
{
	float u = 1.f - t;
	float tt = t * t;
	float uu = u*u;
	float uuu = uu * u;
	float ttt = tt * t;

	Vector3F p0 = data[k00];
	Vector3F p1 = data[k0];
	Vector3F p2 = data[k1];
	Vector3F p3 = data[k11];

	Vector3F p = p0 * uuu; //first term
	p += p1 * 3.f * uu * t; //second term
	p += p2 * 3.f * u * tt; //third term
	p += p3 * ttt; //fourth term
	return p;
}

Vector3F BezierCurve::interpolate(Vector3F * data) const
{
	return calculateBezierPoint(m_t, m_k00, m_k0, m_k1, m_k11, data);
}
