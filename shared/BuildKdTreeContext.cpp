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

BuildKdTreeContext::BuildKdTreeContext(BuildKdTreeStream &data)
{
	create(data.getNumPrimitives());
	
	PrimitiveArray &primitives = data.primitives();
	primitives.begin();
	for(unsigned i = 0; i < m_numPrimitive; i++) {
		m_indices[i] = i;
		
		Primitive *p = primitives.asPrimitive();
		BaseMesh *mesh = (BaseMesh *)(p->getGeometry());
		unsigned triIdx = p->getComponentIndex();
		
		m_primitiveBoxes[i] = mesh->calculateBBox(triIdx);
		primitives.next();
	}
}

BuildKdTreeContext::~BuildKdTreeContext() 
{
	delete[] m_indices;
	delete[] m_primitiveBoxes;
}

void BuildKdTreeContext::create(const unsigned &count)
{
	m_numPrimitive = count;
	m_indices = new unsigned[m_numPrimitive];
	m_primitiveBoxes = new BoundingBox[m_numPrimitive];
}

void BuildKdTreeContext::setBBox(const BoundingBox &bbox)
{
	m_bbox = bbox;
}

BoundingBox BuildKdTreeContext::getBBox() const
{
	return m_bbox;
}

const unsigned BuildKdTreeContext::getNumPrimitives() const
{
	return m_numPrimitive;
}

const unsigned *BuildKdTreeContext::getIndices() const
{
	return m_indices;
}
	
unsigned *BuildKdTreeContext::indices()
{
	return m_indices;
}

void BuildKdTreeContext::setPrimitiveIndex(const unsigned &idx, const unsigned &val)
{
	m_indices[idx] = val;
}

void BuildKdTreeContext::setPrimitiveBBox(const unsigned &idx, const BoundingBox &val)
{
	m_primitiveBoxes[idx] = val;
}

void BuildKdTreeContext::verbose() const
{
	//printf("indices state:\n");
	//m_indices.verbose();
	//printf("nodes state:\n");
	//m_nodes.verbose();
}