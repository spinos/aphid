/*
 *  BuildKdTreeContext.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/20/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "BuildKdTreeContext.h"
BuildKdTreeContext::BuildKdTreeContext() {}

void BuildKdTreeContext::appendMesh(BaseMesh* mesh)
{
	unsigned numFace = mesh->getNumFaces();
	unsigned offset = m_primitives.size();
	m_primitives.allocate(numFace);
	for(unsigned i = 0; i < numFace; i++) {
		m_primitives[offset + i].setGeom((char *)mesh->getFace(i));
		m_primitives[offset + i].setType(0);
	}
}

void BuildKdTreeContext::initIndices()
{
	unsigned numPrim = m_primitives.size();
	m_indices.allocate(numPrim);
	for(unsigned i = 0; i < numPrim; i++) {
		m_indices[i] = i;
	}
}

void BuildKdTreeContext::partition(const SplitCandidate & split)
{
	split.verbose();
	unsigned numPrim = getNumPrimitives();
	m_leftIndices.allocate(numPrim);
	m_rightIndices.allocate(numPrim);
	ClassificationStorage classification;
	classification.setPrimitiveCount(numPrim);
	for(unsigned i = 0; i < numPrim; i++) {
		unsigned idx = m_indices[i];
		const Triangle *tri = m_primitives.asTriangle(idx);
		int side = tri->classify(split);
		classification.set(i, side);
	}
	
	m_leftIndices.start();
	m_rightIndices.start();
	for(unsigned i = 0; i < numPrim; i++) {
		unsigned idx = m_indices[i];
		int side = classification.get(i);
		if(side < 2)
			m_leftIndices.take(idx);
		if(side > 0)
			m_rightIndices.take(idx);
	}
}

const unsigned BuildKdTreeContext::getNumPrimitives() const
{
	return m_indices.size();
}

const BoundingBox BuildKdTreeContext::calculateTightBBox() const
{
	BoundingBox bbox;
	unsigned numPrim = getNumPrimitives();	
	for(unsigned i = 0; i < numPrim; i++) {
		const Triangle *tri = m_primitives.asTriangle(i);
		tri->expandBBox(bbox);
	}
	return bbox;
}