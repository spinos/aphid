/*
 *  PatchMesh.cpp
 *  catmullclark
 *
 *  Created by jian zhang on 5/13/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "PatchMesh.h"
#include <PointInsidePolygonTest.h>
#include <Plane.h>

PatchMesh::PatchMesh() 
{
    setEntityType(TypedEntity::TPatchMesh);
}

PatchMesh::~PatchMesh() 
{
	delete[] m_u;
	delete[] m_v;
	delete[] m_uvIds;
}

void PatchMesh::prePatchUV(unsigned numUVs, unsigned numUVIds)
{
	m_u = new float[numUVs];
	m_v = new float[numUVs];
	m_uvIds = new unsigned[numUVIds];
	m_numUVs = numUVs;
	m_numUVIds = numUVIds;
}

unsigned PatchMesh::numPatches() const
{
	return m_numQuads;
}

float * PatchMesh::us()
{
	return m_u;
}

float * PatchMesh::vs()
{
	return m_v;
}

unsigned * PatchMesh::uvIds()
{
	return m_uvIds;
}

const BoundingBox PatchMesh::calculateBBox(const unsigned &idx) const
{
    BoundingBox box;
	unsigned *qudi = &m_quadIndices[idx * 4];
	Vector3F *p0 = &_vertices[*qudi];
	qudi++;
	Vector3F *p1 = &_vertices[*qudi];
	qudi++;
	Vector3F *p2 = &_vertices[*qudi];
	qudi++;
	Vector3F *p3 = &_vertices[*qudi];
		
	box.updateMin(*p0);
	box.updateMax(*p0);
	box.updateMin(*p1);
	box.updateMax(*p1);
	box.updateMin(*p2);
	box.updateMax(*p2);
	box.updateMin(*p3);
	box.updateMax(*p3);

	return box;
}

char PatchMesh::intersect(unsigned idx, const Ray & ray, IntersectionContext * ctx) const
{
	Vector3F pop[4];
	unsigned *qudi = &m_quadIndices[idx * 4];
	Vector3F *p0 = &_vertices[*qudi];
	qudi++;
	Vector3F *p1 = &_vertices[*qudi];
	qudi++;
	Vector3F *p2 = &_vertices[*qudi];
	qudi++;
	Vector3F *p3 = &_vertices[*qudi];
	Plane pl(*p0, *p1, *p2, *p3);
	
	Vector3F px;
	float t;
	if(!pl.rayIntersect(ray, px, t)) return 0;
	
	if(t < 0.f || t > ray.m_tmax) return 0;
	if(t > ctx->m_minHitDistance) return 0;
	
	pl.projectPoint(*p0, pop[0]);
	pl.projectPoint(*p1, pop[1]);
	pl.projectPoint(*p2, pop[2]);
	pl.projectPoint(*p3, pop[3]);
	
	Vector3F pn;
	pl.getNormal(pn);
	
	PointInsidePolygonTest pipt;
	if(!pipt.isPointInside(px, pn, pop, 4)) return 0;
	
	ctx->m_hitP = px;
	ctx->m_hitN = pn;
	ctx->m_minHitDistance = t;
	ctx->m_geometry = (Geometry*)this;
	
	if(ctx->getComponentFilterType() == PrimitiveFilter::TFace) {
	    ctx->m_componentIdx = idx;
	}
	else {
		int vertInFace = pipt.closestVertex(px, pop, 4);
	    ctx->m_componentIdx = m_quadIndices[idx * 4 + vertInFace];
	}
	
	return 1;
}
//:~
