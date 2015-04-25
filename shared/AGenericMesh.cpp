/*
 *  AGenericMesh.cpp
 *  aphid
 *
 *  Created by jian zhang on 4/25/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "AGenericMesh.h"
#include "BaseBuffer.h"

AGenericMesh::AGenericMesh() 
{
	m_points = new BaseBuffer;
	m_indices = new BaseBuffer;
	m_numPoints = m_numIndices = 0;
}

AGenericMesh::~AGenericMesh() 
{
	delete m_points;
	delete m_indices;
}

const TypedEntity::Type AGenericMesh::type() const
{ return TGenericMesh; }

const BoundingBox AGenericMesh::calculateBBox() const
{
	BoundingBox box;
	const unsigned nv = numPoints();
	Vector3F * p = points();
	for(unsigned i = 0; i < nv; i++) {
		box.updateMin(p[i]);
		box.updateMax(p[i]);
	}
	return box;
}

void AGenericMesh::createBuffer(unsigned np, unsigned ni)
{
	m_points->create(np * 12);
	m_indices->create(ni * 4);
}

Vector3F * AGenericMesh::points() const
{ return (Vector3F *)m_points->data(); }
	
unsigned * AGenericMesh::indices() const
{ return (unsigned *)m_indices->data(); }

void AGenericMesh::setNumPoints(unsigned n)
{ m_numPoints = n; }

void AGenericMesh::setNumIndices(unsigned n)
{ m_numIndices = n; }

const unsigned AGenericMesh::numPoints() const
{ return m_numPoints; }

const unsigned AGenericMesh::numIndices() const
{ return m_numIndices; }
