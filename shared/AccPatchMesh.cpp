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
//:~