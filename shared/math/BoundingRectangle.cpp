/*
 *  BoundingRectangle.cpp
 *  mallard
 *
 *  Created by jian zhang on 10/3/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 *  quadruple of
 *  (xmin, ymin, xmax, ymax)
 */

#include "BoundingRectangle.h"
#include <sstream>

namespace aphid {
    
void RectangleI::set(int x1, int y1)
{
	m_v[0] = 0; m_v[1] = 0;
	m_v[2] = x1; m_v[3] = y1;
}

void RectangleI::set(int x0, int y0, int x1, int y1)
{
	m_v[0] = x0; m_v[1] = y0;
	m_v[2] = x1; m_v[3] = y1;
}

int RectangleI::width() const
{ return m_v[2] - m_v[0]; }

int RectangleI::height() const
{ return m_v[3] - m_v[1]; }

int RectangleI::area() const
{ return width() * height(); }

bool RectangleI::isLandscape() const
{ return width() >= height(); }

const std::string RectangleI::str() const
{
	std::stringstream sst;
	sst<<"(("<<m_v[0]<<","<<m_v[1]<<"),("<<m_v[2]<<","<<m_v[3]<<"))";
	return sst.str();
}

void RectangleI::split(RectangleI & r0, RectangleI & r1, float & alpha, bool alongX) const
{
	int s;
	if(alongX) {
		s = width() / 2;
		if(s & 31) s = (s / 32 + 1)<<5;
		alpha = (float)s/(float)width();
		s--;
		r0.set(m_v[0],       m_v[1], m_v[0] + s, m_v[3]);
		r1.set(m_v[0] + s+1, m_v[1], m_v[2],     m_v[3]);
	}
	else {
		s = height() / 2;
		if(s & 31) s = (s / 32 + 1)<<5;
		alpha = (float)s/(float)height();
		s--;
		r0.set(m_v[0], m_v[1],       m_v[2], m_v[1] + s);
		r1.set(m_v[0], m_v[1] + s+1, m_v[2],     m_v[3]);
	}
}


BoundingRectangle::BoundingRectangle() 
{
	reset();
}

void BoundingRectangle::reset()
{
	m_data[0] = m_data[1] = 1e8f;
	m_data[2] = m_data[3] = -1e8f;
}

void BoundingRectangle::set(float minx, float miny, float maxx, float maxy)
{
	m_data[0] = minx;
	m_data[1] = miny;
	m_data[2] = maxx;
	m_data[3] = maxy;
}

void BoundingRectangle::update(const Vector2F & p)
{
	updateMin(p);
	updateMax(p);
}

void BoundingRectangle::updateMin(const Vector2F & p)
{
	if(m_data[0] > p.x) m_data[0] = p.x;
	if(m_data[1] > p.y) m_data[1] = p.y;
}
	
void BoundingRectangle::updateMax(const Vector2F & p)
{
	if(m_data[2] < p.x) m_data[2] = p.x;
	if(m_data[3] < p.y) m_data[3] = p.y;
}

void BoundingRectangle::translate(const Vector2F & d)
{
	m_data[0] += d.x;
	m_data[2] += d.x;
	m_data[1] += d.y;
	m_data[3] += d.y;
}

const float BoundingRectangle::getMin(int axis) const
{
	return m_data[axis];
}

const float BoundingRectangle::getMax(int axis) const
{
	return m_data[axis + 2];
}

const float BoundingRectangle::distance(const int &axis) const
{
	return m_data[axis + 2] - m_data[axis];
}

bool BoundingRectangle::isPointInside(const Vector2F & p) const
{
	if(p.x < getMin(0) || p.x > getMax(0)) return false;
	if(p.y < getMin(1) || p.y > getMax(1)) return false;
	return true;
}

}