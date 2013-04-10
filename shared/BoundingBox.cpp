/*
 *  BoundingBox.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/17/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include "Aphid.h"
#include "BoundingBox.h"
#include <Ray.h>

BoundingBox::BoundingBox()
{
	reset();
}

void BoundingBox::reset()
{
	m_data[0] = m_data[1] = m_data[2] = 10e8;
	m_data[3] = m_data[4] = m_data[5] = -10e8;
}

void BoundingBox::setMin(float x, float y, float z)
{
	m_data[0] = x; m_data[1] = y; m_data[2] = z;
}

void BoundingBox::setMax(float x, float y, float z)
{
	m_data[3] = x; m_data[4] = y; m_data[5] = z;
}

void BoundingBox::updateMin(const Vector3F & p)
{
	if(m_data[0] > p.x) m_data[0] = p.x;
	if(m_data[1] > p.y) m_data[1] = p.y;
	if(m_data[2] > p.z) m_data[2] = p.z;
}

void BoundingBox::updateMax(const Vector3F & p)
{
	if(m_data[3] < p.x) m_data[3] = p.x;
	if(m_data[4] < p.y) m_data[4] = p.y;
	if(m_data[5] < p.z) m_data[5] = p.z;
}

int BoundingBox::getLongestAxis() const
{
	Vector3F d(m_data[3] - m_data[0], m_data[4] - m_data[1], m_data[5] - m_data[2]);
	if(d.y >= d.x && d.y >= d.z) return 1;
	if(d.z >= d.x && d.z >= d.y) return 2;
	return 0;
}

const float BoundingBox::getMin(int axis) const
{
	return m_data[axis];
}

const float BoundingBox::getMax(int axis) const
{
	return m_data[axis + 3];
}

const float BoundingBox::area() const
{
	return ((m_data[3] - m_data[0]) * (m_data[4] - m_data[1]) + (m_data[3] - m_data[0]) * (m_data[5] - m_data[2]) + (m_data[4] - m_data[1]) * (m_data[5] - m_data[2])) * 2.f;
}

const float BoundingBox::crossSectionArea(const int &axis) const
{
	if(axis == 0) {
		return (m_data[4] - m_data[1]) * (m_data[5] - m_data[2]);
	}
	if(axis == 1) {
		return (m_data[3] - m_data[0]) * (m_data[5] - m_data[2]);
	}
	return (m_data[3] - m_data[0]) * (m_data[4] - m_data[1]);
}

const float BoundingBox::distance(const int &axis) const
{
	if(axis == 0) {
		return m_data[3] - m_data[0];
	}
	if(axis == 1) {
		return m_data[4] - m_data[1];
	}
	return m_data[5] - m_data[2];
}

void BoundingBox::split(int axis, float pos, BoundingBox & left, BoundingBox & right) const
{
	left = right = *this;
	
	if(axis == 0) {
		left.m_data[3] = pos;
		right.m_data[0] = pos;
	}
	else if(axis == 1) {
		left.m_data[4] = pos;
		right.m_data[1] = pos;
	}
	else {
		left.m_data[5] = pos;
		right.m_data[2] = pos;
	}
}

void BoundingBox::expandBy(const BoundingBox &another)
{
	if(m_data[0] > another.m_data[0]) m_data[0] = another.m_data[0];
	if(m_data[1] > another.m_data[1]) m_data[1] = another.m_data[1];
	if(m_data[2] > another.m_data[2]) m_data[2] = another.m_data[2];
	
	if(m_data[3] < another.m_data[3]) m_data[3] = another.m_data[3];
	if(m_data[4] < another.m_data[4]) m_data[4] = another.m_data[4];
	if(m_data[5] < another.m_data[5]) m_data[5] = another.m_data[5];
}

void BoundingBox::expand(float val)
{
    m_data[0] -= val;
    m_data[1] -= val;
    m_data[2] -= val;
    m_data[3] += val;
    m_data[4] += val;
    m_data[5] += val;
}

char BoundingBox::intersect(const Ray &ray, float *hitt0, float *hitt1) const 
{
    float t0 = ray.m_tmin, t1 = ray.m_tmax;
    for (int i = 0; i < 3; ++i) {
		const float diri = ray.m_dir.comp(i);
		const Vector3F o = ray.m_origin;
		if(IsValueNearZero(diri)) {
			if(i == 0) {
				if(o.x < m_data[0] || o.x > m_data[3]) return 0;
			}
			else if(i == 1) {
				if(o.y < m_data[1] || o.y > m_data[4]) return 0;
			}
			else {
				if(o.z < m_data[2] || o.z > m_data[5]) return 0;
			}
			continue;
		}
        // Update interval for _i_th bounding box slab
        float invRayDir = 1.f / ray.m_dir.comp(i);
        float tNear = (getMin(i) - ray.m_origin.comp(i)) * invRayDir;
        float tFar  = (getMax(i) - ray.m_origin.comp(i)) * invRayDir;

        // Update parametric interval from slab intersection $t$s
        if (tNear > tFar) SwapValues(tNear, tFar);
        t0 = tNear > t0 ? tNear : t0;
        t1 = tFar  < t1 ? tFar  : t1;
        if (t0 > t1) return 0;
    }
    if (hitt0) *hitt0 = t0;
    if (hitt1) *hitt1 = t1;
    return 1;
}

char BoundingBox::isPointInside(const Vector3F & p) const
{
	if(p.x < getMin(0) || p.x > getMax(0)) return 0;
	if(p.y < getMin(1) || p.y > getMax(1)) return 0;
	if(p.z < getMin(2) || p.z > getMax(2)) return 0;
	return 1;
}
