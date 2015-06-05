/*
 *  ATriangleMesh.cpp
 *  aphid
 *
 *  Created by jian zhang on 4/25/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "ATriangleMesh.h"
#include "BaseBuffer.h"

ATriangleMesh::ATriangleMesh() 
{
}

ATriangleMesh::~ATriangleMesh() 
{
}

const TypedEntity::Type ATriangleMesh::type() const
{ return TTriangleMesh; }

const unsigned ATriangleMesh::numComponents() const
{ return numTriangles(); }

const unsigned ATriangleMesh::numTriangles() const
{ return numIndices() / 3; }

const BoundingBox ATriangleMesh::calculateBBox(unsigned icomponent) const
{
	Vector3F * p = points();
	unsigned * v = &indices()[icomponent*3];
	BoundingBox box;
	box.updateMin(p[v[0]]);
	box.updateMax(p[v[0]]);
	box.updateMin(p[v[1]]);
	box.updateMax(p[v[1]]);
	box.updateMin(p[v[2]]);
	box.updateMax(p[v[2]]);
	return box;
}

void ATriangleMesh::create(unsigned np, unsigned nt)
{
	createBuffer(np, nt * 3);
	setNumPoints(np);
	setNumIndices(nt * 3);
}

unsigned * ATriangleMesh::triangleIndices(unsigned idx) const
{
	return &indices()[idx*3];
}
