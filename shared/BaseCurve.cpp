/*
 *  BaseCurve.cpp
 *  fit
 *
 *  Created by jian zhang on 4/22/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "BaseCurve.h"
#include "BoundingBox.h"
#include <Ray.h>
#include <line_math.h>

namespace aphid {

float BaseCurve::RayIntersectionTolerance = 1.f;

BaseCurve::BaseCurve() 
{
	m_numVertices = 0;
	m_cvs = 0;
	m_knots = 0;
}

BaseCurve::~BaseCurve() 
{
	cleanup();
}

const TypedEntity::Type BaseCurve::type() const
{ return TCurve; }

void BaseCurve::cleanup()
{
	if(m_knots) delete[] m_knots;
	if(m_cvs) delete[] m_cvs;
}

void BaseCurve::createVertices(unsigned num)
{
	cleanup();
	m_numVertices = num;
	m_cvs = new Vector3F[m_numVertices];
	m_knots = new float[m_numVertices];
}

const unsigned & BaseCurve::numVertices() const
{
	return m_numVertices;
}

const unsigned BaseCurve::numSegments() const
{
    return m_numVertices - 1;
}

unsigned BaseCurve::segmentByParameter(float param) const
{
	if(param <= 0.f) return 0;
	if(param >= 1.f) return numSegments() - 1;
	return param * numSegments();
}

unsigned BaseCurve::segmentByLength(float param) const
{
	unsigned nei0, nei1;
	findNeighborKnots(param, nei0, nei1);
	return nei0;
}

void BaseCurve::computeKnots()
{
	m_hullLength = 0;
	for(unsigned i = 1; i < numVertices(); i++) {
		m_hullLength += m_cvs[i].distanceTo(m_cvs[i-1]);
	}
	
	m_knots[0] = 0.f;
	
	float knotL = 0.f;
	for(unsigned i = 1; i < numVertices(); i++) {
		knotL += m_cvs[i].distanceTo( m_cvs[i-1]);
		m_knots[i] = knotL / m_hullLength;
	}
}

const Vector3F & BaseCurve::getCv(unsigned idx) const
{
	return m_cvs[idx];
}

const float & BaseCurve::getKnot(unsigned idx) const
{
	return m_knots[idx];
}

void BaseCurve::fitInto(BaseCurve & another)
{
	for(unsigned i = 0; i < numVertices(); i++) {
		float param = m_knots[i];
		m_cvs[i] = another.interpolate(param, another.m_cvs);
	}
}

Vector3F BaseCurve::interpolate(float param) const
{
	unsigned seg = segmentByParameter(param);
	float t = param * numSegments() - seg;
	return m_cvs[seg] * (1.f - t) + m_cvs[seg+1] * t;
}

Vector3F BaseCurve::interpolate(float param, Vector3F * data) const
{
	unsigned seg = segmentByParameter(param);
	float t = param * numSegments() - seg;
	return data[seg] * (1.f - t) + data[seg+1] * t;
}

void BaseCurve::findNeighborKnots(float param, unsigned & nei0, unsigned & nei1) const
{
	if(nei1 == nei0 + 1) return;
	unsigned mid = (nei0 + nei1) / 2;
	if(m_knots[mid] > param)
		nei1 = mid;
	else 
		nei0 = mid;
		
	findNeighborKnots(param, nei0, nei1);
}

Vector3F BaseCurve::calculateStraightPoint(float t, unsigned k0, unsigned k1, Vector3F * data) const
{
	return data[k0] * ( 1.f - t) + data[k1] * t;
}

float BaseCurve::length() const
{
	return m_hullLength;
}

const unsigned BaseCurve::numComponents() const 
{ return numSegments(); }

const BoundingBox BaseCurve::calculateBBox() const
{ 
	BoundingBox b;
	for(unsigned i = 0; i < numVertices(); i++)
        b.expandBy(m_cvs[i]);
	return b;
}

const BoundingBox BaseCurve::calculateBBox(unsigned icomponent) const
{
	BoundingBox b;
	b.expandBy(m_cvs[icomponent]);
	b.expandBy(m_cvs[icomponent+1]);
	return b;
}

bool BaseCurve::intersectRay(const Ray * r)
{ 
	const unsigned ns = numSegments();
	unsigned i = 0;
	for(;i<ns;i++)
		if(intersectRay(i, r)) return true;
	return false; 
}

bool BaseCurve::intersectRay(unsigned icomponent, const Ray * r)
{ 
	Vector3F P1 = r->m_origin + r->m_dir * r->m_tmin;
	Vector3F P2 = r->m_origin + r->m_dir * r->m_tmax;
	return (distanceBetweenLines(P1, P2, m_cvs[icomponent], m_cvs[icomponent + 1]) < RayIntersectionTolerance);
}

const float * BaseCurve::cvV() const
{ return (const float *)m_cvs; }

}
