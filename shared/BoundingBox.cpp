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
	reset();
}

void BoundingBox::reset()
{
	m_min.x = m_min.y = m_min.z = 10e8;
	m_max.x = m_max.y = m_max.z = -10e8;
}

void BoundingBox::setMin(float x, float y, float z)
{
	m_min.x = x; m_min.y = y; m_min.z = z;
}

void BoundingBox::setMax(float x, float y, float z)
{
	m_max.x = x; m_max.y = y; m_max.z = z;
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
	if(d.y >= d.x && d.y >= d.z) return 1;
	if(d.z >= d.x && d.z >= d.y) return 2;
	return 0;
}

const float BoundingBox::getMin(int axis) const
{
	if(axis == 0) {
		return m_min.x;
	}
	if(axis == 1) {
		return m_min.y;
	}
	return m_min.z;
}

const float BoundingBox::getMax(int axis) const
{
	if(axis == 0) {
		return m_max.x;
	}
	if(axis == 1) {
		return m_max.y;
	}
	return m_max.z;
}

const float BoundingBox::area() const
{
	return ((m_max.x - m_min.x) * (m_max.y - m_min.y) + (m_max.x - m_min.x) * (m_max.z - m_min.z) + (m_max.y - m_min.y) * (m_max.z - m_min.z)) * 2.f;
}

const float BoundingBox::crossSectionArea(const int &axis) const
{
	if(axis == 0) {
		return (m_max.y - m_min.y) * (m_max.z - m_min.z);
	}
	if(axis == 1) {
		return (m_max.x - m_min.x) * (m_max.z - m_min.z);
	}
	return (m_max.x - m_min.x) * (m_max.y - m_min.y);
}

const float BoundingBox::distance(const int &axis) const
{
	if(axis == 0) {
		return m_max.x - m_min.x;
	}
	if(axis == 1) {
		return m_max.y - m_min.y;
	}
	return m_max.z - m_min.z;
}

void BoundingBox::split(int axis, float pos, BoundingBox & left, BoundingBox & right) const
{
	left.m_min = m_min;
	left.m_max = m_max;
	right.m_min = m_min;
	right.m_max = m_max;
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

void BoundingBox::expandBy(const BoundingBox &another)
{
	updateMin(another.m_min);
	updateMax(another.m_max);
}
