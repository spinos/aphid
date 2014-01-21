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
#include <InverseBilinearInterpolate.h>
#include <BiLinearInterpolate.h>

PatchMesh::PatchMesh() 
{
    setEntityType(TypedEntity::TPatchMesh);
	m_numQuads = 0;
}

PatchMesh::~PatchMesh() 
{
	m_quadIndices.reset();
	m_quadUVIds.reset();
}

unsigned PatchMesh::getNumFaces() const
{
	return m_numQuads;
}

const BoundingBox PatchMesh::calculateBBox() const
{
	return BaseMesh::calculateBBox();
}

const BoundingBox PatchMesh::calculateBBox(const unsigned &idx) const
{
    BoundingBox box;
	unsigned *qudi = &m_quadIndices[idx * 4];
	for(int i=0; i < 4; i++) {
		Vector3F *p0 = &_vertices[*qudi];
		qudi++;
		box.updateMin(*p0);
		box.updateMax(*p0);
	}

	return box;
}

char PatchMesh::intersect(unsigned idx, IntersectionContext * ctx) const
{
	PointInsidePolygonTest pa = patchAt(idx);

	if(!patchIntersect(pa, ctx)) return 0;
	
	postIntersection(idx, ctx);
	
	return 1;
}

char PatchMesh::patchIntersect(PointInsidePolygonTest & pa, IntersectionContext * ctx) const
{
	Vector3F px;
	if(!pa.intersect(ctx->m_ray, px)) return 0;
	
	float d = Vector3F(ctx->m_ray.m_origin, px).length();
	if(d > ctx->m_minHitDistance) return 0;
	
	InverseBilinearInterpolate invbil;
	invbil.setVertices(pa.vertex(0), pa.vertex(1), pa.vertex(3), pa.vertex(2));
	
	ctx->m_patchUV = invbil(px);
	ctx->m_hitP = px;
	pa.getNormal(ctx->m_hitN);
	ctx->m_minHitDistance = d;
	ctx->m_geometry = (Geometry*)this;
	ctx->m_success = 1;

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

char PatchMesh::closestPoint(unsigned idx, const Vector3F & origin, IntersectionContext * ctx) const
{
	PointInsidePolygonTest pa = patchAt(idx);
	
	Vector3F px;
	char inside = 1;
	float d = pa.distanceTo(origin, px, inside);
	
	if(d > ctx->m_minHitDistance) 
		return 0;
		
	ctx->m_minHitDistance = d;
	ctx->m_componentIdx = idx;
	ctx->m_closestP = px;
	ctx->m_hitP = px;

	return 1;
}

PointInsidePolygonTest PatchMesh::patchAt(unsigned idx) const
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
	
	return PointInsidePolygonTest(po[0], po[1], po[2], po[3]);
}

void PatchMesh::perVertexVectorOfPatch(unsigned idx, Vector3F * dst) const
{
	unsigned *qudi = &m_quadIndices[idx * 4];
	dst[0] = perVertexVector()[*qudi];
	qudi++;
	dst[1] = perVertexVector()[*qudi];
	qudi++;
	dst[2] = perVertexVector()[*qudi];
	qudi++;
	dst[3] = perVertexVector()[*qudi];
}

void PatchMesh::perVertexFloatOnPatch(unsigned idx, float u, float v, float * dst) const
{
	float pvf[4];
	unsigned *qudi = &m_quadIndices[idx * 4];
	pvf[0] = perVertexFloat()[*qudi];
	qudi++;
	pvf[1] = perVertexFloat()[*qudi];
	qudi++;
	pvf[2] = perVertexFloat()[*qudi];
	qudi++;
	pvf[3] = perVertexFloat()[*qudi];
	BiLinearInterpolate bili;
	*dst = bili.interpolate(u, v, pvf);
}

void PatchMesh::interpolateVectorOnPatch(unsigned idx, float u, float v, Vector3F * src, Vector3F * dst)
{
	Vector3F pvv[4];
	unsigned *qudi = &m_quadIndices[idx * 4];
	pvv[0] = src[*qudi];
	qudi++;
	pvv[1] = src[*qudi];
	qudi++;
	pvv[2] = src[*qudi];
	qudi++;
	pvv[3] = src[*qudi];
	BiLinearInterpolate bili;
	bili.interpolate3(u, v, pvv, dst);
}

unsigned * PatchMesh::quadIndices()
{
    return m_quadIndices.get();
}

unsigned * PatchMesh::getQuadIndices() const
{
	return m_quadIndices.get();
}

unsigned PatchMesh::processQuadFromPolygon()
{
	unsigned i, j;
	m_numQuads = 0;
    for(i = 0; i < m_numPolygons; i++) {
		if(m_polygonCounts[i] < 5)
			m_numQuads++;
	}
		
	if(m_numQuads < 1) return 0;
	
	m_quadIndices.reset(new unsigned[m_numQuads * 4]);
	m_quadUVIds.reset(new unsigned[m_numQuads * 4]);
	
	unsigned * polygonIndir = polygonIndices();
	unsigned * uvIndir = uvIds();
	unsigned fc, ie = 0;
	for(i = 0; i < m_numPolygons; i++) {
		fc = m_polygonCounts[i];
		if(fc == 4) {
			for(j = 0; j < 4; j++) {
				m_quadIndices[ie] = polygonIndir[j];
				m_quadUVIds[ie] = uvIndir[j];
				ie++;
			}
		}
		else if(fc == 3) {
			for(j = 0; j < 3; j++) {
				m_quadIndices[ie] = polygonIndir[j];
				m_quadUVIds[ie] = uvIndir[j];
				ie++;
			}
			m_quadIndices[ie] = m_quadIndices[ie - 3];
			m_quadUVIds[ie] = m_quadUVIds[ie - 3];
			ie++;
		}
		
		polygonIndir += fc;
		uvIndir += fc;
	}
	
	return m_numQuads;
}

unsigned PatchMesh::numQuads() const
{
	return m_numQuads;
}

unsigned * PatchMesh::quadUVIds()
{
	return m_quadUVIds.get();
}

unsigned * PatchMesh::getQuadUVIds() const
{
	return m_quadUVIds.get();
}

Vector3F PatchMesh::getFaceNormal(const unsigned & idx) const
{
	PointInsidePolygonTest pa = patchAt(idx);
	return pa.normal();
}
//:~
