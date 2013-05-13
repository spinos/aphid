/*
 *  PatchMesh.cpp
 *  catmullclark
 *
 *  Created by jian zhang on 5/13/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "PatchMesh.h"

PatchMesh::PatchMesh() {}
PatchMesh::~PatchMesh() 
{
	delete[] m_vertexValence;
	delete[] m_patchVertices;
	delete[] m_patchBoundary;
}

void PatchMesh::createVertexValence(unsigned num)
{
	m_vertexValence = new unsigned[num];
}

void PatchMesh::prePatchValence()
{
	m_patchVertices = new unsigned[numPatches() * 24];
	m_patchBoundary = new char[numPatches() * 15];
}

unsigned PatchMesh::numPatches() const
{
	return m_numQuads;
}

unsigned * PatchMesh::vertexValence()
{
	return m_vertexValence;
}

unsigned * PatchMesh::patchVertices()
{
	return m_patchVertices;
}

char * PatchMesh::patchBoundaries()
{
	return m_patchBoundary;
}