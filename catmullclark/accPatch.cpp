/*
 *  accPatch.cpp
 *  catmullclark
 *
 *  Created by jian zhang on 10/28/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "accPatch.h"
#include "accStencil.h"
#include <iostream>

AccPatch::AccPatch() {}
AccPatch::~AccPatch() {}

AccStencil* AccPatch::stencil = 0;

void AccPatch::evaluateContolPoints()
{
    for(int i = 0; i < 4; i++) {
        stencil->findCorner(i);
        processCornerControlPoints(i);
    }
	
	for(int i = 0; i < 4; i++) {
        stencil->findEdge(i);
        processEdgeControlPoints(i);
    }
	
	for(int i = 0; i < 4; i++) {
        stencil->findInterior(i);
        processInteriorControlPoints(i);
    }
}

void AccPatch::processCornerControlPoints(int i)
{
	const int cornerIndex[4] = {0, 3, 15, 12};
	const int j = cornerIndex[i];
	AccCorner &topo = stencil->m_corners[i];
	_contorlPoints[j] = topo.computePosition();
	_normals[j] = topo.computeNormal();
}

void AccPatch::processEdgeControlPoints(int i)
{
	const int edgeIndex[8] = {1, 2, 7, 11, 14, 13, 8, 4};
	AccEdge &topo = stencil->m_edges[i];
	int v0 = edgeIndex[i * 2];
	int v1 = edgeIndex[i * 2 + 1];
	_contorlPoints[v0] = topo.computePosition(0);
	_normals[v0] = topo.computeNormal(0);
	
	_contorlPoints[v1] = topo.computePosition(1);
	_normals[v1] = topo.computeNormal(1);
}

void AccPatch::processInteriorControlPoints(int i)
{
	const int interiorIndex[4] = {5, 6, 10, 9};
	const int j = interiorIndex[i];
	AccInterior &topo = stencil->m_interiors[i];
	_contorlPoints[j] = topo.computePosition();
	_normals[j] = topo.computeNormal();
}
