/*
 *  CollisionRegion.cpp
 *  mallard
 *
 *  Created by jian zhang on 9/21/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "CollisionRegion.h"
#include <IntersectionContext.h>

CollisionRegion::CollisionRegion() : m_regionElementStart(UINT_MAX) 
{
	m_ctx = new IntersectionContext;
}

CollisionRegion::~CollisionRegion() 
{
	m_regionElementIndices.clear();
	delete m_ctx;
}

Vector3F CollisionRegion::getClosestPoint(const Vector3F & origin)
{
	m_ctx->reset();
	closestPoint(origin, m_ctx);
	return m_ctx->m_closestP;
}

void CollisionRegion::pushPlane(Patch::PushPlaneContext * ctx) const
{
}

void CollisionRegion::resetCollisionRegion(unsigned idx)
{
	m_regionElementStart = idx;
	m_regionElementIndices.clear();
}

void CollisionRegion::resetCollisionRegionAround(unsigned idx, const Vector3F & p, const float & d)
{
	resetCollisionRegion(idx);
}

void CollisionRegion::closestPoint(const Vector3F & origin, IntersectionContext * ctx) const
{}

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
