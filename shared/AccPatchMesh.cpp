/*
 *  AccPatchMesh.cpp
 *  mallard
 *
 *  Created by jian zhang on 8/30/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "AccPatchMesh.h"
#include <accPatch.h>
#include <accStencil.h>
#include <MeshTopology.h>

AccPatchMesh::AccPatchMesh() 
{
	if(!AccPatch::stencil) {
		AccStencil* sten = new AccStencil();
		AccPatch::stencil = sten;
	}
}

AccPatchMesh::~AccPatchMesh() 
{
	delete[] m_bezier;
}

void AccPatchMesh::setup(MeshTopology * topo)
{
	Vector3F* cvs = getVertices();
	Vector3F* normal = getNormals();
	float* ucoord = us();
	float* vcoord = vs();
	unsigned * uvs = uvIds();
	const int numFace = numPatches();

	AccStencil* sten = AccPatch::stencil;
	sten->setVertexPosition(cvs);
	sten->setVertexNormal(normal);
	
	sten->m_vertexAdjacency = topo->getTopology();

	m_bezier = new AccPatch[numFace];
	unsigned * quadV = quadIndices();
	for(int j = 0; j < numFace; j++) {
		sten->m_patchVertices[0] = quadV[0];
		sten->m_patchVertices[1] = quadV[1];
		sten->m_patchVertices[2] = quadV[2];
		sten->m_patchVertices[3] = quadV[3];
		
		m_bezier[j].setTexcoord(ucoord, vcoord, &uvs[j * 4]);
		m_bezier[j].evaluateContolPoints();
		m_bezier[j].evaluateTangents();
		m_bezier[j].evaluateBinormals();
		
		quadV += 4;
	}
}

AccPatch* AccPatchMesh::beziers() const
{
	return m_bezier;
}

const BoundingBox AccPatchMesh::calculateBBox(const unsigned &idx) const
{
	return beziers()[idx].controlBBox();
}

char AccPatchMesh::intersect(unsigned idx, IntersectionContext * ctx) const
{
    if(!recursiveBezierIntersect(&beziers()[idx], ctx, 0)) return 0;

	postIntersection(idx, ctx);
	
	return 1;
}

char AccPatchMesh::recursiveBezierIntersect(BezierPatch* patch, IntersectionContext * ctx, int level) const
{
    Ray ray = ctx->m_ray;
    BoundingBox controlbox = patch->controlBBox();
	float hitt0, hitt1;
	if(!controlbox.intersect(ray, &hitt0, &hitt1))
		return 0;

	if(hitt1 > ctx->m_minHitDistance) return 0;
	
	if(level > 3 || controlbox.area() < .1f) {
	    Vector3F fourCorners[4];
	    fourCorners[0] = patch->_contorlPoints[0];
	    fourCorners[1] = patch->_contorlPoints[3];
	    fourCorners[2] = patch->_contorlPoints[15];
	    fourCorners[3] = patch->_contorlPoints[12];
	    return planarIntersect(fourCorners, ctx);
	}
	
	level++;
	
	BezierPatch children[4];
	patch->decasteljauSplit(children);
	
	if(recursiveBezierIntersect(&children[0], ctx, level)) return 1;
	if(recursiveBezierIntersect(&children[1], ctx, level)) return 1;
	if(recursiveBezierIntersect(&children[2], ctx, level)) return 1;
	if(recursiveBezierIntersect(&children[3], ctx, level)) return 1;
	return 0;
}
//:~