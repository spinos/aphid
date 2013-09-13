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
	//findNeighborKnots(param, m_k0, m_k1);
	//m_k0 = (unsigned)(param * (numVertices()-1));
	//m_k1 = m_k0 + 1;
	unsigned seg = param * numSegments();
	Vector3F p[4];
	
	if(seg == 0) p[0] = m_cvs[0];
	else p[0] = (m_cvs[seg - 1] * 0.33f + m_cvs[seg] * 0.67f) * .5f + (m_cvs[seg + 1] * 0.33f + m_cvs[seg] * 0.67f) * .5f;
	
	if(seg >= numSegments() - 1) p[3] = m_cvs[numSegments()];
	else p[3] = (m_cvs[seg] * 0.33f + m_cvs[seg + 1] * 0.67f) * .5f + (m_cvs[seg + 2] * 0.33f + m_cvs[seg + 1] * 0.67f) * .5f;

	p[1] = m_cvs[seg + 1] * 0.33f + m_cvs[seg] * 0.67f;
	p[2] = m_cvs[seg] * 0.33f + m_cvs[seg + 1] * 0.67f;
	//fourControlKnots();
	//calculateT(param);
	float t = param * numSegments() - seg;
	
	return calculateBezierPoint(t, p);
}

void BezierCurve::fourControlKnots()
{
}

Vector3F BezierCurve::calculateBezierPoint(float t, Vector3F * data) const
{
	float u = 1.f - t;
	float tt = t * t;
	float uu = u*u;
	float uuu = uu * u;
	float ttt = tt * t;

	Vector3F p0 = data[0];
	Vector3F p1 = data[1];
	Vector3F p2 = data[2];
	Vector3F p3 = data[3];

	Vector3F p = p0 * uuu; //first term
	p += p1 * 3.f * uu * t; //second term
	p += p2 * 3.f * u * tt; //third term
	p += p3 * ttt; //fourth term
	return p;
}
