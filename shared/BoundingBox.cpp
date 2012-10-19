/*
 *  BoundingBox.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/17/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "BoundingBox.h"

BoundingBox::BoundingBox()
{
	m_min.x = m_min.y = m_min.z = 10e8;
	m_max.x = m_max.y = m_max.z = -10e8;
}

void BoundingBox::updateMin(const Vector3F & p)
{
	if(m_min.x > p.x) m_min.x = p.x;
	if(m_min.y > p.y) m_min.y = p.y;
	if(m_min.z > p.z) m_min.z = p.z;
}

void BoundingBox::updateMax(const Vector3F & p)
{
	if(m_max.x < p.x) m_max.x = p.x;
	if(m_max.y < p.y) m_max.y = p.y;
	if(m_max.z < p.z) m_max.z = p.z;
}

int BoundingBox::getLongestAxis() const
{
	Vector3F d = m_max - m_min;
	if(d.y > d.x && d.y > d.z) return 1;
	if(d.z > d.x && d.z > d.y) return 2;
	return 0;
}

void BoundingBox::split(int axis, float pos, BoundingBox & left, BoundingBox & right) const
{
	left = *this;
	right = *this;
	if(axis == 0) {
		left.m_max.x = pos;
		right.m_min.x = pos;
	}
	else if(axis == 1) {
		left.m_max.y = pos;
		right.m_min.y = pos;
	}
	else {
		left.m_max.z = pos;
		right.m_min.z = pos;
	}
}