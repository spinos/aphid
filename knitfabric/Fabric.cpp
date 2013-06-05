/*
 *  Fabric.cpp
 *  knitfabric
 *
 *  Created by jian zhang on 6/4/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "Fabric.h"
#include <PatchMesh.h>
#include <MeshTopology.h>
#include <accStencil.h>

Fabric::Fabric() {}

void Fabric::setMesh(PatchMesh * mesh, MeshTopology * topo)
{
	m_mesh = mesh;
	const unsigned numPatch = m_mesh->numPatches();
	
	Vector3F* cvs = m_mesh->getVertices();
	Vector3F* normal = m_mesh->getNormals();
	float* ucoord = m_mesh->us();
	float* vcoord = m_mesh->vs();
	unsigned * uvIds = m_mesh->uvIds();
	unsigned * quadV = m_mesh->quadIndices();
	
	AccStencil* sten = new AccStencil();
	AccPatch::stencil = sten;
	sten->setVertexPosition(cvs);
	sten->setVertexNormal(normal);
	sten->m_vertexAdjacency = topo->getTopology();
	
	m_bezier = new YarnPatch[numPatch];
	for(unsigned i = 0; i < numPatch; i++) {
		sten->m_patchVertices[0] = quadV[0];
		sten->m_patchVertices[1] = quadV[1];
		sten->m_patchVertices[2] = quadV[2];
		sten->m_patchVertices[3] = quadV[3];
		
		m_bezier[i].setTexcoord(ucoord, vcoord, &uvIds[i * 4]);
		m_bezier[i].evaluateContolPoints();
		m_bezier[i].evaluateTangents();
		m_bezier[i].evaluateBinormals();
		m_bezier[i].setQuadVertices(quadV);

		quadV += 4;
	}
}

unsigned Fabric::numPatches() const
{
	return m_mesh->numPatches();
}

YarnPatch Fabric::getPatch(unsigned idx) const
{
	return m_bezier[idx];
}

YarnPatch * Fabric::patch(unsigned idx)
{
	return &m_bezier[idx];
}
