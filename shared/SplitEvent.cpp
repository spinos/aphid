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
unsigned SplitEvent::NumPrimitive = 0;
unsigned *SplitEvent::PrimitiveIndices = 0;
BoundingBox *SplitEvent::PrimitiveBoxes = 0;
BoundingBox SplitEvent::ParentBox;
//BuildKdTreeContext *SplitEvent::Context = 0;

SplitEvent::SplitEvent() 
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

const float SplitEvent::getCost() const
{
	return m_cost;
}

int SplitEvent::leftCount() const
{
	return m_leftTouch;
}

int SplitEvent::rightCount() const
{
	return m_rightTouch;
}

int SplitEvent::side(const BoundingBox &box) const
{
	int side = 1;
	if(box.getMax(m_axis) < m_pos)
		side = 0;
	else if(box.getMin(m_axis) >= m_pos)
		side = 2;
	return side;
}

void SplitEvent::calculateTightBBoxes(const BoundingBox &box, BoundingBox &leftBBox, BoundingBox &rightBBox)
{
	const int s = side(box);
	if(s == 0) {
		m_leftTouch++;
		leftBBox.expandBy(box);
	}
	else if(s == 2 ) {
		m_rightTouch++;
		rightBBox.expandBy(box);
	}
	else {
		m_leftTouch++;
		m_rightTouch++;
		leftBBox.expandBy(box);
		rightBBox.expandBy(box);
	}
}

void SplitEvent::calculateCost()
{
	m_cost = 10e8;
	m_leftTouch = 0;
	m_rightTouch = 0;
	BoundingBox leftBBox, rightBBox;
	for(unsigned i = 0; i < NumPrimitive; i++) {
		//unsigned &primIdx = PrimitiveIndices[i];
		BoundingBox &primBox = PrimitiveBoxes[i];
		calculateTightBBoxes(primBox, leftBBox, rightBBox);
	}
	
	m_cost = 15.f + 20.f * (leftBBox.area() * m_leftTouch + rightBBox.area() * m_rightTouch ) / ParentBox.area();
}

void SplitEvent::verbose() const
{
	printf("%i: %i + %i c %f \n", m_axis, m_leftTouch, m_rightTouch, m_cost);
	
}
