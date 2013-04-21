/*
 *  BaseCurve.cpp
 *  fit
 *
 *  Created by jian zhang on 4/22/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "BaseCurve.h"

BaseCurve::BaseCurve() {}
BaseCurve::~BaseCurve() 
{
	m_vertices.clear();
	m_knots.clear();
}

void BaseCurve::addVertex(const Vector3F & vert)
{
	m_vertices.push_back(vert);
}

unsigned BaseCurve::numVertices() const
{
	return (unsigned)m_vertices.size();
}

void BaseCurve::computeKnots()
{
	m_length = 0;
	for(unsigned i = 1; i < numVertices(); i++) {
		m_length += (m_vertices[i] - m_vertices[i-1]).length();
	}
	
	m_knots.push_back(0.f);
	
	float knotL = 0.f;
	for(unsigned i = 1; i < numVertices(); i++) {
		knotL += (m_vertices[i] - m_vertices[i-1]).length();
		m_knots.push_back(knotL/ m_length);
	}
}

Vector3F BaseCurve::getVertex(unsigned idx) const
{
	return m_vertices[idx];
}

float BaseCurve::getKnot(unsigned idx) const
{
	return m_knots[idx];
}
