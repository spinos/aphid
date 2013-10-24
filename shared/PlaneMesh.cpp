/*
 *  PlaneMesh.cpp
 *  aphid
 *
 *  Created by jian zhang on 10/24/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "PlaneMesh.h"
#include <Patch.h>
#include <BaseCurve.h>

PlaneMesh::PlaneMesh(const Vector3F & bottomLeft, const Vector3F & bottomRight, const Vector3F & topRight, const Vector3F & topLeft, unsigned gu, unsigned gv) : PatchMesh() 
{
	setFaceConnection(gu, gv);
	unsigned i, j;
	Patch pl(bottomLeft, bottomRight, topRight, topLeft);
	const float du = 1.f/ (float)gu;
	const float dv = 1.f/ (float)gv;
	
	for(j = 0; j <= gv; j++)
	for(i = 0; i <= gu; i++) {
		_vertices[j * (gu + 1) + i] = pl.point(du * i, dv * j);
	}
	
	processTriangleFromPolygon();
	processQuadFromPolygon();
}

PlaneMesh::PlaneMesh(const BaseCurve & udir, const BaseCurve & vdir, unsigned gu, unsigned gv) : PatchMesh()
{
	setFaceConnection(gu, gv);
	unsigned i, j;
	const float du = 1.f/ (float)gu;
	const float dv = 1.f/ (float)gv;
 
	for(j = 0; j <= gv; j++)
	for(i = 0; i <= gu; i++) {
		_vertices[j * (gu + 1) + i] = udir.interpolate(du * i) + vdir.interpolate(dv * j);
	}
	
	processTriangleFromPolygon();
	processQuadFromPolygon();
}

PlaneMesh::~PlaneMesh() {}

void PlaneMesh::setFaceConnection(unsigned gu, unsigned gv)
{
	createPolygonCounts(gu * gv);
	createPolygonIndices(gu * gv * 4);
	createVertices((gu + 1) * (gv + 1));
	
	unsigned i, j, face;
	for(i = 0; i < m_numPolygons; i++) m_polygonCounts[i] = 4;
	
	face = 0;
	for(j = 0; j < gv; j++)
	for(i = 0; i < gu; i++) {
		m_polygonIndices[face * 4] = j * (gu + 1) + i;
		m_polygonIndices[face * 4 + 1] = j * (gu + 1) + i + 1;
		m_polygonIndices[face * 4 + 2] = (j + 1) * (gu + 1) + i + 1;
		m_polygonIndices[face * 4 + 3] = (j + 1) * (gu + 1) + i;
		face++;
	}
}