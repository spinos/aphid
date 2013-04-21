/*
 *  TargetGraph.cpp
 *  hc
 *
 *  Created by jian zhang on 4/7/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "TargetGraph.h"

TargetGraph::TargetGraph() 
{
	m_previousFace = 0;
}

TargetGraph::~TargetGraph() {}

void TargetGraph::createVertexWeights(unsigned num)
{
	m_vertexWeights = new float[num];
}

void TargetGraph::createTargetIndices(unsigned num)
{
	m_targetIndices = new unsigned[num];
}

void TargetGraph::setTarget(unsigned idx, unsigned a)
{
	m_targetIndices[idx] = a;
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
	co.project(pos);
	co.compute();
	unsigned * in = indices();
	unsigned a, b, c;
	
	m_dirtyTargets.clear();
	if(m_previousFace != faceIdx) {
		a = in[m_previousFace*3];
		b = in[m_previousFace*3 + 1];
		c = in[m_previousFace*3 + 2];
		m_vertexWeights[a] = 0.f;
		m_vertexWeights[b] = 0.f;
		m_vertexWeights[c] = 0.f;
		addDirtyTargets(m_previousFace);
	}
	
	a = in[faceIdx*3];
	b = in[faceIdx*3 + 1];
	c = in[faceIdx*3 + 2];
	m_vertexWeights[a] = co.getValue()[0];
	m_vertexWeights[b] = co.getValue()[1];
	m_vertexWeights[c] = co.getValue()[2];
	
	//printf("set %i %f %i %f %i %f\n", a, co.getValue()[0], b, co.getValue()[1], c, co.getValue()[2]);
			

	addDirtyTargets(faceIdx);
	m_previousFace = faceIdx;
}

void TargetGraph::addDirtyTargets(unsigned faceIdx)
{
	unsigned a, b, c;
	a = m_targetIndices[faceIdx*3];
	b = m_targetIndices[faceIdx*3 + 1];
	c = m_targetIndices[faceIdx*3 + 2];
	if(a > 0) 
		m_dirtyTargets[a] = indices()[faceIdx*3];
	if(b > 0)
		m_dirtyTargets[b] = indices()[faceIdx*3+1];
	if(c > 0)
		m_dirtyTargets[c] = indices()[faceIdx*3+1];
}

unsigned TargetGraph::firstDirtyTarget()
{
	m_dirtyTargetIt = m_dirtyTargets.begin();
	return m_dirtyTargetIt->first;
}

unsigned TargetGraph::nextDirtyTarget()
{
	m_dirtyTargetIt++;
	if(!hasDirtyTarget()) return 0;
	return m_dirtyTargetIt->first;
}

bool TargetGraph::hasDirtyTarget()
{
	return m_dirtyTargetIt != m_dirtyTargets.end();
}

unsigned TargetGraph::getControlId() const
{
	return m_controlId;
}

float TargetGraph::targetWeight(unsigned idx) const
{
	const unsigned nv = getNumVertices();
	for(unsigned i = 0; i < nv; i++) {
		if(m_targetIndices[i] == idx) {
			return m_vertexWeights[i];
		}
	}
	return 0.f;
}
//:~