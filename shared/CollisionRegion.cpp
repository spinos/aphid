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
    Vector3F baseColor;
    m_distribution->sample(meshUV.x, meshUV.y, 3, (float *)&baseColor);
	resetCollisionRegion(idx);
	for(unsigned i = 1; i < numRegionElements(); i++) {
	    if(faceColorMatches(regionElementIndex(i), baseColor))
			m_topo->growAroundQuad(regionElementIndex(i), *regionElementIndices());
	}
	if(numRegionElements() > 0) {
	    std::cout<<"n sel"<<numRegionElements();
	    createBuffer(numRegionElements() * 32);
	    rebuildBuffer();
	}
}

char CollisionRegion::faceColorMatches(unsigned idx, const Vector3F & refCol) const
{
    float u, v;
    Vector3F texcoord, curCol, difCol;
    unsigned i;
    for(i = 0; i < 32; i++) {
        u = ((float)(rand()%591))/591.f;
		v = ((float)(rand()%593))/593.f;
        m_body->texcoordOnPatch(idx, u, v, texcoord);
        m_distribution->sample(texcoord.x, texcoord.y, 3, (float *)&curCol);
        difCol = curCol - refCol;
        if(difCol.length() < 0.067f) return 1;
    }
    return 0;
}

void CollisionRegion::rebuildBuffer() 
{
    unsigned i, k;
    for(i = 0; i < numRegionElements(); i++) {
        for(k = 0; k < 4; k++)
            fillPatchEdge(regionElementIndex(i), k, i * 32 + k * 8);
    }
}

void CollisionRegion::fillPatchEdge(unsigned iface, unsigned iedge, unsigned vstart)
{
    Vector3F p;
    float u, v;
    unsigned i, acc = 0;
    for(i = 0; i < 4; i++) {
        if(iedge == 0) {
            v = 0.f;
            u = .25f * i;
        }
        else if(iedge == 1) {
            u = 1.f;
            v = .25f * i;
        }
        else if(iedge == 2) {
            v = 1.f;
            u = 1.f - .25f * i;
        }
        else {
            u = 0.f;
            v = 1.f - .25f * i;
        }
        
        m_body->pointOnPatch(iface, u, v, p);
        vertices()[acc + vstart] = p;
        acc++;
        
        if(iedge == 0) {
            u = .25f * i + .25;
        }
        else if(iedge == 1) {
            v = .25f * i + .25;
        }
        else if(iedge == 2) {
            u = 1.f - .25f * i - .25;
        }
        else {
            v = 1.f - .25f * i - .25;
        }
        
        m_body->pointOnPatch(iface, u, v, p);
        vertices()[acc + vstart] = p;
        acc++;
    }
}
