/*
 *  BaseCurve.cpp
 *  fit
 *
 *  Created by jian zhang on 4/22/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "BaseCurve.h"
std::vector<Vector3F> BaseCurve::BuilderVertices;
BaseCurve::BaseCurve() 
{
	m_cvs = 0;
	m_knots = 0;
}

BaseCurve::~BaseCurve() 
{
	cleanup();
}

void BaseCurve::cleanup()
{
	if(m_knots) delete[] m_knots;
	if(m_cvs) delete[] m_cvs;
}

void BaseCurve::createVertices(unsigned num)
{
	m_numVertices = num;
	m_cvs = new Vector3F[m_numVertices];
}

void BaseCurve::addVertex(const Vector3F & vert)
{
	BuilderVertices.push_back(vert);
}

void BaseCurve::finishAddVertex()
{
    m_numVertices = (unsigned)BuilderVertices.size();
	
	m_cvs = new Vector3F[numVertices()];
	for(unsigned i = 0; i < numVertices(); i++) m_cvs[i] = BuilderVertices[i];
	
	BuilderVertices.clear();
}

unsigned BaseCurve::numVertices() const
{
	return m_numVertices;
}

unsigned BaseCurve::numSegments() const
{
    return m_numVertices - 1;
}

void BaseCurve::computeKnots()
{
	m_length = 0;
	for(unsigned i = 1; i < numVertices(); i++) {
		m_length += (m_cvs[i] - m_cvs[i-1]).length();
	}
	
	m_knots = new float[numVertices()];
	m_knots[0] = 0.f;
	
	float knotL = 0.f;
	for(unsigned i = 1; i < numVertices(); i++) {
		knotL += (m_cvs[i] - m_cvs[i-1]).length();
		m_knots[i] = knotL / m_length;
	}
}

Vector3F BaseCurve::getCv(unsigned idx) const
{
	return m_cvs[idx];
}

float BaseCurve::getKnot(unsigned idx) const
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

Vector3F BaseCurve::interpolate(float param, Vector3F * data)
{
	m_t = 0.f;
	m_k0 = 0;
	m_k1 = numVertices() - 1;
	
	if(param <= 0.f) {
		m_k1 = m_k0;
		return data[m_k0];
	}
	
	if(param >= 1.f) {
		m_k0 = m_k1;
		return data[m_k1];
	}
	
	findNeighborKnots(param, m_k0, m_k1);
	
	calculateT(param);
	
	return calculateStraightPoint(m_t, m_k0, m_k1, data);
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

Vector3F BaseCurve::interpolateByKnot(float param, Vector3F * data)
{
	m_t = 0.f;
	m_k0 = 0;
	m_k1 = numVertices() - 1;
	
	if(param <= m_k0) {
		m_k1 = m_k0;
		return data[m_k0];
	}
	
	if(param >= m_k1) {
		m_k0 = m_k1;
		return data[m_k1];
	}
	
	m_k0 = (unsigned)param;
	m_k1 = m_k0 + 1;
	float realparam = m_knots[m_k0] * (1.f - (param - m_k0)) + m_knots[m_k1] * (param - m_k0);
	
	calculateT(realparam);

	return calculateStraightPoint(m_t, m_k0, m_k1, data);
}

void BaseCurve::calculateT(float param)
{
	m_t = (param - m_knots[m_k0]) / (m_knots[m_k1] - m_knots[m_k0]);
}

Vector3F BaseCurve::calculateStraightPoint(float t, unsigned k0, unsigned k1, Vector3F * data) const
{
	return data[k0] * ( 1.f - t) + data[k1] * t;
}

Vector3F BaseCurve::interpolate(Vector3F * data) const
{
	return calculateStraightPoint(m_t, m_k0, m_k1, data);
}
