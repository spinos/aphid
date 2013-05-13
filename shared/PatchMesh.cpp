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
	delete[] m_u;
	delete[] m_v;
	delete[] m_uvIds;
}

void PatchMesh::prePatchValence()
{
	m_vertexValence = new unsigned[getNumVertices()];
	m_patchVertices = new unsigned[numPatches() * 24];
	m_patchBoundary = new char[numPatches() * 15];
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
