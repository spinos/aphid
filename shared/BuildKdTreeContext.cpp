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
	
	BoundingBox *primBoxes = m_primitiveBoxes.ptr();
	unsigned *primIndex = m_indices.ptr();
	
	PrimitiveArray &primitives = data.primitives();
	primitives.begin();
	for(unsigned i = 0; i < m_numPrimitive; i++) {
		primIndex[i] = i;
		
		Primitive *p = primitives.asPrimitive();
		BaseMesh *mesh = (BaseMesh *)(p->getGeometry());
		unsigned triIdx = p->getComponentIndex();
		
		primBoxes[i] = mesh->calculateBBox(triIdx);
		primitives.next();
	}
}

BuildKdTreeContext::~BuildKdTreeContext() 
{
}

void BuildKdTreeContext::create(const unsigned &count)
{
	m_numPrimitive = count;
	m_indices.create(m_numPrimitive+1);
	m_primitiveBoxes.create(m_numPrimitive+1);
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

unsigned *BuildKdTreeContext::indices()
{
	return m_indices.ptr();
}

void BuildKdTreeContext::verbose() const
{
	//printf("indices state:\n");
	//m_indices.verbose();
}