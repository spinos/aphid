/*
 *  SplitEvent.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/22/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include "SplitEvent.h"
#include <iostream>

namespace aphid {

SplitEvent::SplitEvent() : m_isEmpty(1)
{
	m_cost = 1e38f;
}

void SplitEvent::setEmpty()
{ m_isEmpty = 1; }

void SplitEvent::setBAP(const BoundingBox & b, int a, float p)
{
	b.split(a, p, m_lftBound, m_rgtBound);
	m_axis = a;
	m_pos = p;
	m_isEmpty = 0;
	//m_leftBox.reset();
	//m_rightBox.reset();
}
	
const float & SplitEvent::getPos() const
{
	return m_pos;
}

const int & SplitEvent::getAxis() const
{
	return m_axis;
}

void SplitEvent::setLeftRightNumPrim(const unsigned &leftNumPrim, const unsigned &rightNumPrim)
{
	m_leftNumPrim = leftNumPrim;
	m_rightNumPrim = rightNumPrim;
}

const float & SplitEvent::getCost() const
{
	return m_cost;
}

const int & SplitEvent::leftCount() const
{ return m_leftNumPrim; }

const int & SplitEvent::rightCount() const
{ return m_rightNumPrim; }

int SplitEvent::side(const BoundingBox &box) const
{
/// both
	int side = 1;
/// left only
	if(box.getMax(m_axis) <= m_pos)
		side = 0;
/// right only
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

void SplitEvent::limitBox(const BoundingBox & b)
{
    if(m_isEmpty) return;
    if(m_leftBox.isValid()) m_leftBox.shrinkBy(b);
    if(m_rightBox.isValid()) m_rightBox.shrinkBy(b);
}

void SplitEvent::calculateCost(const float & a)
{/*
	BoundingBox leftBBox, rightBBox;
	for(unsigned i = 0; i < NumPrimitive; i++) {
		//unsigned &primIdx = PrimitiveIndices[i];
		BoundingBox &primBox = PrimitiveBoxes[i];
		calculateTightBBoxes(primBox, leftBBox, rightBBox);
	}
	
	m_cost = 15.f + 20.f * (leftBBox.area() * m_leftNumPrim + rightBBox.area() * m_rightNumPrim) / ParentBoxArea;
	*/
	if(m_isEmpty) return;
	m_cost = 1.5f;
	if(m_leftBox.isValid()) {
		//m_leftBox.shrinkBy(b);
		m_cost += 2.f * (m_leftBox.area() * m_leftNumPrim) / a;
	}
	if(m_rightBox.isValid()) {
		//m_rightBox.shrinkBy(b);
		m_cost += 2.f * (m_rightBox.area() * m_rightNumPrim) / a;
	}
}

float SplitEvent::area() const
{
	return m_leftBox.area() + m_rightBox.area();
}

bool SplitEvent::hasBothSides() const
{
	return (m_leftNumPrim > 0 && m_rightNumPrim > 0);
}

const BoundingBox & SplitEvent::leftBound() const
{ return m_lftBound; }

const BoundingBox & SplitEvent::rightBound() const
{ return m_rgtBound; }

void SplitEvent::verbose() const
{
	std::cout<<"\n split event along "<<m_axis
	<<" at "<<m_pos<<" cost "<<m_cost;
	if(m_isEmpty) {
		std::cout<<" is empty";
		return;
	}
	std::cout<<"\n prim count "<<m_leftNumPrim<<"/"<<m_rightNumPrim
	<<"\n bound "<<m_lftBound
	<<"/"<<m_rgtBound
	<<"\n box "<<m_leftBox
	<<"/"<<m_rightBox;
	//printf("cost %f left %i %f right %i %f\n", m_cost, m_leftNumPrim, m_leftBox.area(), m_rightNumPrim, m_rightBox.area());
}

const char & SplitEvent::isEmpty() const
{ return m_isEmpty; }
}
//:~