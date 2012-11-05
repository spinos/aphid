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
float SplitEvent::ParentBoxArea = 1.f;

SplitEvent::SplitEvent() 
{
	m_cost = 10e8;
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

void SplitEvent::setLeftRightNumPrim(const unsigned &leftNumPrim, const unsigned &rightNumPrim)
{
	m_leftNumPrim = leftNumPrim;
	m_rightNumPrim = rightNumPrim;
}

const float SplitEvent::getCost() const
{
	return m_cost;
}

int SplitEvent::leftCount() const
{
	return m_leftNumPrim;
}

int SplitEvent::rightCount() const
{
	return m_rightNumPrim;
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
		leftBBox.expandBy(box);
	}
	else if(s == 2 ) {
		rightBBox.expandBy(box);
	}
	else {
		leftBBox.expandBy(box);
		rightBBox.expandBy(box);
	}
}

void SplitEvent::updateLeftBox(const BoundingBox &box)
{
	m_leftBox.expandBy(box);
}

void SplitEvent::updateRightBox(const BoundingBox &box)
{
	m_rightBox.expandBy(box);	
}

void SplitEvent::calculateCost()
{/*
	BoundingBox leftBBox, rightBBox;
	for(unsigned i = 0; i < NumPrimitive; i++) {
		//unsigned &primIdx = PrimitiveIndices[i];
		BoundingBox &primBox = PrimitiveBoxes[i];
		calculateTightBBoxes(primBox, leftBBox, rightBBox);
	}
	
	m_cost = 15.f + 20.f * (leftBBox.area() * m_leftNumPrim + rightBBox.area() * m_rightNumPrim) / ParentBoxArea;
	*/
	
	m_cost = 150.f + 200.f * (m_leftBox.area() * m_leftNumPrim + m_rightBox.area() * m_rightNumPrim) / ParentBoxArea;
	
}

void SplitEvent::verbose() const
{
	printf("%i: %i + %i c %f \n", m_axis, m_leftNumPrim, m_rightNumPrim, m_cost);
}
