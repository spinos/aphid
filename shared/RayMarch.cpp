/*
 *  RayMarch.cpp
 *  btree
 *
 *  Created by jian zhang on 5/7/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "RayMarch.h"

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
	
	m_path.m_origin += m_path.m_dir * 10e-6;

	return true;
}

bool RayMarch::end() 
{ 
	if(m_path.m_tmax < 10e-6) return true;
	
	return false; 
}

void RayMarch::step() 
{
	Vector3F p = m_path.m_origin;
	int cx, cy, cz;
	cx = p.x / m_gridSize; if(p.x < 0.f) cx--;
	cy = p.y / m_gridSize; if(p.y < 0.f) cy--;
	cz = p.z / m_gridSize; if(p.z < 0.f) cz--;
	m_current.setMin(m_gridSize * cx, m_gridSize * cy, m_gridSize * cz);
	m_current.setMax(m_gridSize * (cx + 1), m_gridSize * (cy + 1), m_gridSize * (cz + 1));
	float hitMin, hitMax;
	m_current.intersect(m_path, &hitMin, &hitMax);
	hitMax += 10e-6;
	m_path.m_origin += m_path.m_dir * hitMax;
	m_path.m_tmax -= hitMax;
}

const BoundingBox RayMarch::gridBBox() const
{
	return m_current;
}
