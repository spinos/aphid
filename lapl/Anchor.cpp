/*
 *  Anchor.cpp
 *  lapl
 *
 *  Created by jian zhang on 3/19/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "Anchor.h"
#include <Vertex.h>
#include <Ray.h>
namespace aphid {

Anchor::Anchor() {}

Anchor::Anchor(SelectionArray & sel)
{
	const unsigned nv = sel.numVertices();
	for(unsigned i=0; i < nv; i++) {
		Vector3F v = sel.getVertexP(i);
		AnchorPoint *a = new AnchorPoint();
		a->worldP = v;
		a->w = 1.f;
		addPoint(sel.getVertexId(i), a);
	}
	
	computeLocalSpace();
	keepOriginalSpace();
}

Anchor::~Anchor() 
{
	for(m_anchorPointIt = m_anchorPoints.begin(); m_anchorPointIt != m_anchorPoints.end(); ++m_anchorPointIt) {
		delete (*m_anchorPointIt).second;
	}
	m_anchorPoints.clear();
}

void Anchor::placeAt(const Vector3F & cen)
{
	m_space.setIdentity();
	m_space.setTranslation(cen);
	for(m_anchorPointIt = m_anchorPoints.begin(); m_anchorPointIt != m_anchorPoints.end(); ++m_anchorPointIt) {
		Vector3F & pos = ((*m_anchorPointIt).second)->p;
		((*m_anchorPointIt).second)->worldP = m_space.transform(pos);
	}
}

void Anchor::addPoint(unsigned vertexId, AnchorPoint * ap)
{
	m_anchorPoints[vertexId] = ap;
	m_pointIndex.push_back(vertexId);
}

void Anchor::setWeight(float wei)
{
	for(m_anchorPointIt = m_anchorPoints.begin(); m_anchorPointIt != m_anchorPoints.end(); ++m_anchorPointIt) {
		(*m_anchorPointIt).second->w = wei;
	}
}

void Anchor::addWeight(float delta)
{
    for(m_anchorPointIt = m_anchorPoints.begin(); m_anchorPointIt != m_anchorPoints.end(); ++m_anchorPointIt) {
		float wei = (*m_anchorPointIt).second->w + delta;
		if(wei > 1.f) wei = 1.f;
		if(wei < 0.f) wei = 0.f;
        (*m_anchorPointIt).second->w = wei;
	}
}

unsigned Anchor::numPoints() const
{
	return (unsigned) m_anchorPoints.size();
}

Anchor::AnchorPoint * Anchor::firstPoint(unsigned &idx)
{
	m_anchorPointIt = m_anchorPoints.begin();
	idx = (*m_anchorPointIt).first;
	return (*m_anchorPointIt).second;
}

Anchor::AnchorPoint * Anchor::nextPoint(unsigned &idx)
{
	m_anchorPointIt++;
	if(!hasPoint()) return 0;
	idx = (*m_anchorPointIt).first;
	return (*m_anchorPointIt).second;
}

bool Anchor::hasPoint()
{
	return m_anchorPointIt != m_anchorPoints.end();
}

bool Anchor::intersect(const Ray &ray, float &t, float threshold)
{
	const float axis = ray.m_dir.longestAxis();
	for(m_anchorPointIt = m_anchorPoints.begin(); m_anchorPointIt != m_anchorPoints.end(); ++m_anchorPointIt) {
		Vector3F & pos = ((*m_anchorPointIt).second)->worldP;
		t = (pos.comp(axis) - ray.m_origin.comp(axis)) / ray.m_dir.comp(axis);
		if(t > 0.f) {
			Vector3F pop = ray.travel(t);
			if((pop - pos).length() < threshold)
				return true;
		}
	}
	return false;
}

void Anchor::translate(Vector3F & dis)
{
	m_space.translate(dis);
	
	for(m_anchorPointIt = m_anchorPoints.begin(); m_anchorPointIt != m_anchorPoints.end(); ++m_anchorPointIt) {
		Vector3F & pos = ((*m_anchorPointIt).second)->p;
		((*m_anchorPointIt).second)->worldP = m_space.transform(pos);
	}
}

Anchor::AnchorPoint * Anchor::getPoint(unsigned idx)
{
	const unsigned linearIdx = m_pointIndex[idx];
	return m_anchorPoints[linearIdx];
}

unsigned Anchor::getVertexIndex(unsigned idx)
{
	return m_pointIndex[idx];
}

void Anchor::computeLocalSpace()
{
	Vector3F cen;
	cen.setZero();
	for(m_anchorPointIt = m_anchorPoints.begin(); m_anchorPointIt != m_anchorPoints.end(); ++m_anchorPointIt) {
		cen += ((*m_anchorPointIt).second)->worldP;
	}
	cen /= numPoints();
	m_space.setIdentity();
	m_space.setTranslation(cen);
	
	Matrix44F invs = m_space;
	invs.inverse();
	
	for(m_anchorPointIt = m_anchorPoints.begin(); m_anchorPointIt != m_anchorPoints.end(); ++m_anchorPointIt) {
		Vector3F & pos = ((*m_anchorPointIt).second)->worldP;
		((*m_anchorPointIt).second)->p = invs.transform(pos);
	}
}

void Anchor::clear()
{
	for(m_anchorPointIt = m_anchorPoints.begin(); m_anchorPointIt != m_anchorPoints.end(); ++m_anchorPointIt) {
		delete (*m_anchorPointIt).second;
	}
	m_anchorPoints.clear();
	m_pointIndex.clear();
}

}
//:~
