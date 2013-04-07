/*
 *  TargetGraph.cpp
 *  hc
 *
 *  Created by jian zhang on 4/7/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "TargetGraph.h"

TargetGraph::TargetGraph() {}
TargetGraph::~TargetGraph() {}

void TargetGraph::createVertexWeights(unsigned num)
{
	m_vertexWeights = new float[num];
}

void TargetGraph::createTargetIndices(unsigned num)
{
	m_targetIndices = new unsigned[num];
}

void TargetGraph::setTargetTriangle(unsigned idx, unsigned a, unsigned b, unsigned c)
{
	unsigned *i = m_targetIndices;
	i += idx * 3;
	*i = a;
	i++;
	*i = b;
	i++;
	*i = c;
}

void TargetGraph::setControlId(unsigned idx)
{
	m_controlId = idx;
}

void TargetGraph::initCoords()
{
	unsigned * in = indices();
	Vector3F * v = vertices();
	const unsigned nf = getNumFaces();
	m_baryc = new BarycentricCoordinate[nf];
	unsigned a, b, c;
	for(unsigned i = 0; i < nf; i++) {
		a = in[i*3];
		b = in[i*3 + 1];
		c = in[i*3 + 2];
		m_baryc[i].create(v[a], v[b], v[c]);
	}
}

void TargetGraph::reset()
{
	m_vertexWeights[0] = 1.f;
	const unsigned nv = getNumVertices();
	for(unsigned i = 1; i < nv; i++) {
		m_vertexWeights[i] = 0.f;
	}
	m_handlePos = vertices()[0];
}

Vector3F TargetGraph::getHandlePos() const
{
	return m_handlePos;
}

void TargetGraph::computeWeight(unsigned faceIdx, const Vector3F & pos)
{
	m_handlePos = pos;
	BarycentricCoordinate &co = m_baryc[faceIdx];
	co.compute(pos);
	unsigned * in = indices();
	unsigned a, b, c;
	a = in[faceIdx*3];
	b = in[faceIdx*3 + 1];
	c = in[faceIdx*3 + 2];
	m_vertexWeights[a] = co.getValue()[0];
	m_vertexWeights[b] = co.getValue()[1];
	m_vertexWeights[c] = co.getValue()[2];
}
//:~