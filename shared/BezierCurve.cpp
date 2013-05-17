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

Vector3F BezierCurve::interpolate(float param, Vector3F * data) const
{
	unsigned k0 = 0;
	unsigned k1 = numVertices() - 1;
	
	if(param <= 0.f) return data[k0];
	if(param >= 1.f) return data[k1];
	
	findNeighborKnots(param, k0, k1);
	//printf("%f %i %i  ", param, k0, k1);
	return calculateBezierPoint(param, k0, k1, data);
}

Vector3F BezierCurve::interpolateByKnot(float param, Vector3F * data) const
{
	unsigned k0 = 0;
	unsigned k1 = numVertices() - 1;
	
	if(param <= k0) return data[k0];
	if(param >= k1) return data[k1];
	
	k0 = (unsigned)param;
	k1 = k0 + 1;
	float realparam = m_knots[k0] * (1.f - (param - k0)) + m_knots[k1] * (param - k0);

	return calculateBezierPoint(realparam, k0, k1, data);
}


Vector3F BezierCurve::calculateBezierPoint(float param, unsigned k0, unsigned k1, Vector3F * data) const
{
	int k00 = k0 - 1;
	if(k00 < 0) k00 = 0;
	
	unsigned k11 = k1 + 1;
	if(k11 > numVertices() - 1) k11 = numVertices() - 1;
	
	float t = (param - m_knots[k00]) / (m_knots[k11] - m_knots[k00]);
	
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