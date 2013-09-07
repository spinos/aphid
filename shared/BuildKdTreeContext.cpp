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
		
		unsigned compIdx = p->getComponentIndex();
		
		primBoxes[i] = mesh->calculateBBox(compIdx);
		primBoxes[i].expand(10e-6);
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

BoundingBox *BuildKdTreeContext::boxes()
{
	return m_primitiveBoxes.ptr();
}

float BuildKdTreeContext::visitCost() const
{
	return 2.f * m_numPrimitive;
}

void BuildKdTreeContext::verbose() const
{
	//printf("indices state:\n");
	//m_indices.verbose();
}