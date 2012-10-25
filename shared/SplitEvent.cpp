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
//BuildKdTreeContext *SplitEvent::Context = 0;

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

void SplitEvent::calculateSides(const PrimitivePtr *primitives, const unsigned &count)
{
	//m_sides.setPrimitiveCount(count);
	for(unsigned i = 0; i < count; i++) {
		Primitive *prim = primitives[i];
		BaseMesh *mesh = (BaseMesh *)(prim->getGeometry());
		unsigned triIdx = prim->getComponentIndex();
		int side = mesh->faceOnSideOf(triIdx, getAxis(), getPos());
		//m_sides.set(i, side);
		//m_sides.set(i, 2);
	}
}

