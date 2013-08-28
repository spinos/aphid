/*
 *  PatchMesh.cpp
 *  catmullclark
 *
 *  Created by jian zhang on 5/13/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "PatchMesh.h"

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
	unsigned *qudi = &m_quadIndices[idx * 3];
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
#include <iostream>
char PatchMesh::intersect(unsigned idx, const Ray & ray, IntersectionContext * ctx) const
{
    printf("pat inst");
    return BaseMesh::intersect(idx, ray, ctx);
}
//:~
