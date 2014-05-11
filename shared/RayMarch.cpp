/*
 *  RayMarch.cpp
 *  btree
 *
 *  Created by jian zhang on 5/7/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "RayMarch.h"
#include <iostream>
RayMarch::RayMarch() {}

void RayMarch::initialize(const BoundingBox & bb, const float & gridSize) 
{
	m_limit = bb;
	m_gridSize = gridSize;
}

bool RayMarch::begin(const Ray & r) 
{ 
	float hitMin, hitMax;
	if(!m_limit.intersect(r, &hitMin, &hitMax)) return false;
	
	m_path = r;
	m_path.m_tmax = hitMax;
	if(hitMin > 0.f) {
		m_path.m_origin += m_path.m_dir * hitMin;
		m_path.m_tmax -= hitMin;
	}
	
	m_path.m_origin += m_path.m_dir * 10e-5;

	return true;
}

bool RayMarch::end() 
{ 
	if(m_path.m_tmax < 10e-5) return true;
	
	return false; 
}

void RayMarch::step() 
{
	m_current = computeBBox(m_path.m_origin);
	float hitMin, hitMax;
	m_current.intersect(m_path, &hitMin, &hitMax);
	
	if(hitMax < 10e-5) {
		std::cout<<" step "<<hitMin<<","<<hitMax<<","<<m_path.m_tmax;
		hitMax = 10e-5;
	}
	hitMax += 10e-5;
	m_path.m_origin += m_path.m_dir * hitMax;
	m_path.m_tmax -= hitMax;
}

const BoundingBox RayMarch::gridBBox() const
{
	return m_current;
}

const std::deque<Vector3F> RayMarch::touched(const float & threshold) const
{
	std::deque<Vector3F> r;
	const Vector3F cen = gridBBox().center();
	const int l = 1 + threshold / m_gridSize;
	int i, j, k;
	Vector3F p;
	for(k= -l; k <= l; k++) {
		p.z = cen.z + m_gridSize * k;
		for(j= -l; j <= l; j++) {
			p.y = cen.y + m_gridSize * j;
			for(i= -l; i <= l; i++) {
				p.x = cen.x + m_gridSize * i;
				if(m_limit.isPointInside(p)) r.push_back(p);
			}
		}
	}
	return r;
}

const BoundingBox RayMarch::computeBBox(const Vector3F & p) const
{
	int cx, cy, cz;
	cx = p.x / m_gridSize; if(p.x < 0.f) cx--;
	cy = p.y / m_gridSize; if(p.y < 0.f) cy--;
	cz = p.z / m_gridSize; if(p.z < 0.f) cz--;
	BoundingBox b;
	b.setMin(m_gridSize * cx, m_gridSize * cy, m_gridSize * cz);
	b.setMax(m_gridSize * (cx + 1), m_gridSize * (cy + 1), m_gridSize * (cz + 1));
	return b;
}
