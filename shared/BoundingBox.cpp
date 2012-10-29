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
	m_min_x = m_min_y = m_min_z = 10e8;
	m_max_x = m_max_y = m_max_z = -10e8;
}

void BoundingBox::setMin(float x, float y, float z)
{
	m_min_x = x; m_min_y = y; m_min_z = z;
}

void BoundingBox::setMax(float x, float y, float z)
{
	m_max_x = x; m_max_y = y; m_max_z = z;
}

void BoundingBox::updateMin(const Vector3F & p)
{
	if(m_min_x > p.x) m_min_x = p.x;
	if(m_min_y > p.y) m_min_y = p.y;
	if(m_min_z > p.z) m_min_z = p.z;
}

void BoundingBox::updateMax(const Vector3F & p)
{
	if(m_max_x < p.x) m_max_x = p.x;
	if(m_max_y < p.y) m_max_y = p.y;
	if(m_max_z < p.z) m_max_z = p.z;
}

int BoundingBox::getLongestAxis() const
{
	Vector3F d(m_max_x - m_min_x, m_max_y - m_min_y, m_max_z - m_min_z);
	if(d.y >= d.x && d.y >= d.z) return 1;
	if(d.z >= d.x && d.z >= d.y) return 2;
	return 0;
}

const float BoundingBox::getMin(int axis) const
{
	if(axis == 0) {
		return m_min_x;
	}
	if(axis == 1) {
		return m_min_y;
	}
	return m_min_z;
}

const float BoundingBox::getMax(int axis) const
{
	if(axis == 0) {
		return m_max_x;
	}
	if(axis == 1) {
		return m_max_y;
	}
	return m_max_z;
}

const float BoundingBox::area() const
{
	return ((m_max_x - m_min_x) * (m_max_y - m_min_y) + (m_max_x - m_min_x) * (m_max_z - m_min_z) + (m_max_y - m_min_y) * (m_max_z - m_min_z)) * 2.f;
}

const float BoundingBox::crossSectionArea(const int &axis) const
{
	if(axis == 0) {
		return (m_max_y - m_min_y) * (m_max_z - m_min_z);
	}
	if(axis == 1) {
		return (m_max_x - m_min_x) * (m_max_z - m_min_z);
	}
	return (m_max_x - m_min_x) * (m_max_y - m_min_y);
}

const float BoundingBox::distance(const int &axis) const
{
	if(axis == 0) {
		return m_max_x - m_min_x;
	}
	if(axis == 1) {
		return m_max_y - m_min_y;
	}
	return m_max_z - m_min_z;
}

void BoundingBox::split(int axis, float pos, BoundingBox & left, BoundingBox & right) const
{
	left = right = *this;
	
	if(axis == 0) {
		left.m_max_x = pos;
		right.m_min_x = pos;
	}
	else if(axis == 1) {
		left.m_max_y = pos;
		right.m_min_y = pos;
	}
	else {
		left.m_max_z = pos;
		right.m_min_z = pos;
	}
}

void BoundingBox::expandBy(const BoundingBox &another)
{
	if(m_min_x > another.m_min_x) m_min_x = another.m_min_x;
	if(m_min_y > another.m_min_y) m_min_y = another.m_min_y;
	if(m_min_z > another.m_min_z) m_min_z = another.m_min_z;
	
	if(m_max_x < another.m_max_x) m_max_x = another.m_max_x;
	if(m_max_y < another.m_max_y) m_max_y = another.m_max_y;
	if(m_max_z < another.m_max_z) m_max_z = another.m_max_z;
}
