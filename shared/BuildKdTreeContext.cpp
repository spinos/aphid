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

SplitCandidate BuildKdTreeContext::bestSplit()
{
	int axis = m_bbox.getLongestAxis();
	float pos = (m_bbox.getMin(axis) + m_bbox.getMax(axis)) * 0.5f;
	SplitCandidate candidate;
	candidate.setPos(pos);
	candidate.setAxis(axis);
	return candidate;
}

void BuildKdTreeContext::partition(const SplitCandidate & split)
{
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
	m_leftIndices.resizeToTaken();
	m_rightIndices.resizeToTaken();
}

void BuildKdTreeContext::setBBox(const BoundingBox &bbox)
{
	m_bbox = bbox;
}

void BuildKdTreeContext::setPrimitives(const PrimitiveArray &prims)
{
	m_primitives = prims;
}

void BuildKdTreeContext::setIndices(const IndexArray &indices)
{
	m_indices = indices;
}

const unsigned BuildKdTreeContext::getNumPrimitives() const
{
	return m_indices.size();
}

const BoundingBox & BuildKdTreeContext::getBBox() const
{
	return m_bbox;
}
	
const PrimitiveArray &BuildKdTreeContext::getPrimitives() const
{
	return m_primitives;
}	

const IndexArray &BuildKdTreeContext::getLeftIndices() const
{
	return m_leftIndices;
}
	
const IndexArray &BuildKdTreeContext::getRightIndices() const
{
	return m_rightIndices;
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

void BuildKdTreeContext::verbose() const
{
	printf("ctx partition %i primitives:\n", getNumPrimitives());
	unsigned leftCount = m_leftIndices.taken();
	unsigned rightCount = m_rightIndices.taken();
	printf("%i to left side:\n", leftCount);
	for(unsigned i = 0; i < leftCount; i++) {
		printf("%i ", m_leftIndices[i]);
	}
	printf("\n%i to right side:\n", rightCount);
	for(unsigned i = 0; i < rightCount; i++) {
		printf("%i ", m_rightIndices[i]);
	}
	printf("\n");
}