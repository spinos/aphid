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
	Vector3F po[4];
	unsigned *qudi = &m_quadIndices[idx * 4];
	po[0] = _vertices[*qudi];
	qudi++;
	po[1] = _vertices[*qudi];
	qudi++;
	po[2] = _vertices[*qudi];
	qudi++;
	po[3] = _vertices[*qudi];
	
	if(!planarIntersect(po, ray, ctx)) return 0;
	
	if(ctx->getComponentFilterType() == PrimitiveFilter::TFace)
	    ctx->m_componentIdx = idx;
	else
	    ctx->m_componentIdx = closestVertex(idx, ctx->m_hitP);
		
	return 1;
}

char PatchMesh::planarIntersect(const Vector3F * fourCorners, const Ray & ray, IntersectionContext * ctx) const
{
	Plane pl(fourCorners[0], fourCorners[1], fourCorners[2], fourCorners[3]);
	
	Vector3F px;
	float t;
	if(!pl.rayIntersect(ray, px, t)) return 0;
	
	if(t < 0.f || t > ray.m_tmax) return 0;
	if(t > ctx->m_minHitDistance) return 0;
	
	Vector3F pop[4];
	pl.projectPoint(fourCorners[0], pop[0]);
	pl.projectPoint(fourCorners[1], pop[1]);
	pl.projectPoint(fourCorners[2], pop[2]);
	pl.projectPoint(fourCorners[3], pop[3]);
	
	Vector3F pn;
	pl.getNormal(pn);
	
	PointInsidePolygonTest pipt;
	if(!pipt.isPointInside(px, pn, pop, 4)) return 0;
	
	ctx->m_hitP = px;
	ctx->m_hitN = pn;
	ctx->m_minHitDistance = t;
	ctx->m_geometry = (Geometry*)this;
	
	return 1;
}

unsigned PatchMesh::closestVertex(unsigned idx, const Vector3F & px) const
{
	unsigned *qudi = &m_quadIndices[idx * 4];
	float mag, minDist = 10e8;
	unsigned vert = 0;
	for(int i = 0; i < 4; i++) {
		Vector3F v = _vertices[*qudi] - px;
		
		mag = v.length();
		
		if(mag < minDist) {
			minDist = mag;
			vert = *qudi;
		}
		qudi++;
	}
	return vert;
}
//:~
