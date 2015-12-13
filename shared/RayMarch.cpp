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
	
	m_path.m_origin += m_path.m_dir * 10e-4;

	return true;
}

bool RayMarch::end() 
{ 
	if(m_path.m_tmax < 10e-4) return true;
	
	return false; 
}

void RayMarch::step() 
{
	m_current = computeBBox(m_path.m_origin);
	float hitMin, hitMax;
	m_current.intersect(m_path, &hitMin, &hitMax);
	
	if(hitMax < 10e-4) {
		//std::cout<<" step "<<hitMin<<","<<hitMax<<","<<m_path.m_tmax;
		hitMax = 10e-4;
	}
	hitMax += 10e-4;
	m_path.m_origin += m_path.m_dir * hitMax;
	m_path.m_tmax -= hitMax;
}

const BoundingBox RayMarch::gridBBox() const
{
	return m_current;
}

const std::deque<Vector3F> RayMarch::touched(const float & threshold, BoundingBox & limit) const
{
	const Vector3F u(threshold, threshold, threshold);
	const Vector3F p0 = m_path.m_origin - u;
	const Vector3F p1 = m_path.m_origin + u;
	
	const BoundingBox blo = computeBBox(p0);
	const BoundingBox bhi = computeBBox(p1);
	
	limit.reset();
	limit.expandBy(blo);
	limit.expandBy(bhi);
	limit.shrinkBy(m_limit);
	
	std::deque<Vector3F> r;
/// center of cell entered
	const Vector3F c = computeBBox(m_path.m_origin).center();
	const int ng = threshold / m_gridSize + 1;
	int i, j, k;
	Vector3F p;
/// check neighbors
	for(k= -ng; k <= ng; k++) {
		p.z = c.z + m_gridSize * k;
		for(j= -ng; j <= ng; j++) {
			p.y = c.y + m_gridSize * j;
			for(i= -ng; i <= ng; i++) {
				p.x = c.x + m_gridSize * i;
				if(m_limit.isPointInside(p))
					if(limit.isPointInside(p)) 
						r.push_back(p);
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
