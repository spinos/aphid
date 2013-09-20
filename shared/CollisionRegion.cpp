/*
 *  CollisionRegion.cpp
 *  mallard
 *
 *  Created by jian zhang on 9/21/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "CollisionRegion.h"

CollisionRegion::CollisionRegion() : m_regionElementStart(UINT_MAX) {}
CollisionRegion::~CollisionRegion() {}

void CollisionRegion::resetCollisionRegion(unsigned idx)
{
	m_regionElementStart = idx;
}

unsigned CollisionRegion::numRegionElements() const
{
	return m_regionElementIndices.size();
}

unsigned CollisionRegion::regionElementIndex(unsigned idx) const
{
	return m_regionElementIndices[idx];
}

unsigned CollisionRegion::regionElementStart() const
{
	return m_regionElementStart;
}

void CollisionRegion::setRegionElementStart(unsigned x)
{
	m_regionElementStart = x;
}

std::vector<unsigned> * CollisionRegion::regionElementIndices()
{
	return &m_regionElementIndices;
}
