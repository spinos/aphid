/*
 *  accPatch.cpp
 *  catmullclark
 *
 *  Created by jian zhang on 10/28/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "accPatch.h"
#include "patchTopology.h"
#include "accStencil.h"
#include <iostream>
AccPatch::AccPatch() {}
AccPatch::~AccPatch() {}

AccStencil* AccPatch::stencil;

void AccPatch::evaluateContolPoints(PatchTopology& topo)
{
	processCornerControlPoints(topo);
	processEdgeControlPoints(topo);
	processInteriorControlPoints(topo);
}

void AccPatch::processCornerControlPoints(PatchTopology& topo)
{
	const int cornerIndex[4] = {0, 3, 15, 12};
	for(int j = 0; j < 4; j++)
	{
		stencil->centerIndex = topo.getCornerIndex(j);
	
		if(topo.isCornerOnBoundary(j))
		{
			topo.getBoundaryEdgesOnCorner(j, stencil->edgeIndices);
			_contorlPoints[cornerIndex[j]] = stencil->computePositionOnCornerOnBoundary();
			_normals[cornerIndex[j]] = stencil->computeNormalOnCornerOnBoundary();
		}
		else
		{
			topo.getFringeAndEdgesOnCorner(j, stencil->cornerIndices, stencil->edgeIndices, stencil->m_isCornerBehindEdge);
			stencil->valence = topo.getValenceOnCorner(j);
			_contorlPoints[cornerIndex[j]] = stencil->computePositionOnCorner();
			_normals[cornerIndex[j]] = stencil->computeNormalOnCorner();
		}
	}	
}

void AccPatch::processEdgeControlPoints(PatchTopology& topo)
{
	const int edgeIndex[8] = {1, 2, 7, 11, 14, 13, 8, 4};
	for(int j = 0; j < 8; j++)
	{
		int edge = j / 2;
		int side = j % 2;
		if(topo.isEdgeOnBoundary(edge))
		{
			topo.getEdgeBySide(edge, side, stencil->edgeIndices);
			_contorlPoints[edgeIndex[j]] = stencil->computePositionOnEdgeOnBoundary();
			_normals[edgeIndex[j]] = stencil->computeNormalOnEdgeOnBoundary();
		}
		else
		{
			topo.getFringeAndEdgesOnEdgeBySide(edge, side, stencil->cornerIndices, stencil->edgeIndices);
			stencil->valence = topo.getCornerValenceByEdgeBySide(edge, side);
			_contorlPoints[edgeIndex[j]] = stencil->computePositionOnEdge();
			_normals[edgeIndex[j]] = stencil->computeNormalOnEdge();
		}
	}
}

void AccPatch::processInteriorControlPoints(PatchTopology& topo)
{
	const int interiorIndex[4] = {5, 6, 10, 9};
	for(int j = 0; j < 4; j++)
	{
		topo.getFringeOnFaceByCorner(j, stencil->cornerIndices);
		stencil->valence = topo.getCornerValenceByEdgeBySide(j, 0);
		_contorlPoints[interiorIndex[j]] = stencil->computePositionInterior();
		_normals[interiorIndex[j]] = stencil->computeNormalInterior();
	}
}
