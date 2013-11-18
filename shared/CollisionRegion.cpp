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
#include <MeshTopology.h>
#include <AccPatchMesh.h>
#include <BaseImage.h>
CollisionRegion::CollisionRegion() : m_regionElementStart(UINT_MAX) 
{
	m_ctx = new IntersectionContext;
	m_topo = 0;
	m_body = 0;
	m_distribution = 0;
}

CollisionRegion::~CollisionRegion() 
{
	m_regionElementIndices.clear();
	delete m_ctx;
}

AccPatchMesh * CollisionRegion::bodyMesh() const
{
	return m_body;
}

void CollisionRegion::setBodyMesh(AccPatchMesh * mesh, MeshTopology * topo)
{
	m_body = mesh;
	m_topo = topo;
}

Vector3F CollisionRegion::getClosestPoint(const Vector3F & origin)
{
	m_ctx->reset();
	closestPoint(origin, m_ctx);
	return m_ctx->m_closestP;
}

void CollisionRegion::resetCollisionRegion(unsigned idx)
{
	//if(idx == regionElementStart()) return;
	m_regionElementStart = idx;
	m_regionElementIndices.clear();
	m_topo->growAroundQuad(idx, *regionElementIndices());
}

void CollisionRegion::resetCollisionRegionAround(unsigned idx, const Vector3F & p, const float & d)
{
	resetCollisionRegion(idx);
	m_topo->growAroundQuad(idx, *regionElementIndices());
	for(unsigned i = 1; i < numRegionElements(); i++) {
		BoundingBox bb = m_body->calculateBBox(regionElementIndex(i));
		if(bb.isPointAround(p, d))
			m_topo->growAroundQuad(regionElementIndex(i), *regionElementIndices());
	}
	for(unsigned i = 1; i < numRegionElements(); i++) {
		BoundingBox bb = m_body->calculateBBox(regionElementIndex(i));
		if(!bb.isPointAround(p, d)) {
			regionElementIndices()->erase(regionElementIndices()->begin() + i);
			i--;
		}
	}
}

void CollisionRegion::closestPoint(const Vector3F & origin, IntersectionContext * ctx) const
{
	for(unsigned i=0; i < numRegionElements(); i++) {
		m_body->closestPoint(regionElementIndex(i), origin, ctx);
	}
}

void CollisionRegion::pushPlane(Patch::PushPlaneContext * ctx) const
{
	for(unsigned i=0; i < numRegionElements(); i++) {
		m_body->pushPlane(regionElementIndex(i), ctx);
	}
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

void CollisionRegion::setDistributionMap(BaseImage * image)
{
    m_distribution = image;
}

void CollisionRegion::selectRegion(unsigned idx, const Vector2F & patchUV)
{
    if(!m_distribution) return;
    Vector3F meshUV;
    bodyMesh()->texcoordOnPatch(idx, patchUV.x, patchUV.y, meshUV);
    float samples[3];
    m_distribution->sample(meshUV.x, meshUV.y, 3, samples);
}
