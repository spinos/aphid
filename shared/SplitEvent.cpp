/*
 *  SplitEvent.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/22/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include "SplitEvent.h"
#include <BaseMesh.h>
#include <BuildKdTreeContext.h>
int SplitEvent::Dimension = 3;
BuildKdTreeContext *SplitEvent::Context = 0;

SplitEvent::SplitEvent() 
{
}

SplitEvent::~SplitEvent() 
{
	//printf("event quit\n");
}

void SplitEvent::clear()
{
}

void SplitEvent::setPos(float val)
{
	m_pos = val;
}

void SplitEvent::setAxis(int val)
{
	m_axis = val;
}
	
float SplitEvent::getPos() const
{
	return m_pos;
}

int SplitEvent::getAxis() const
{
	return m_axis;
}

void SplitEvent::calculateSides(const PartitionBound &bound)
{
	IndexArray &indices = Context->indices();
	PrimitiveArray &primitives = Context->primitives();
	
	m_sides.setPrimitiveCount(bound.numPrimitive());
	for(unsigned i = bound.parentMin; i < bound.parentMax; i++) {
		unsigned idx = *indices.asIndex(i);
		BaseMesh *mesh = (BaseMesh *)(primitives.asPrimitive(idx)->getGeometry());
		const unsigned triIdx = primitives.asPrimitive(idx)->getComponentIndex();
		const int side = mesh->faceOnSideOf(triIdx, getAxis(), getPos());
		
		m_sides.set(i - bound.parentMin, side);
	}
}

const ClassificationStorage *SplitEvent::getSides() const
{
	return &m_sides;
}
