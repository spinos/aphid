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
#include "BarycentricCoordinate.h"

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
	unsigned * v = triangleIndices(icomponent);
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
{ return &indices()[idx*3]; }

void ATriangleMesh::closestToPoint(unsigned icomponent, ClosestToPointTestResult * result)
{
	Vector3F * p = points();
	unsigned * v = triangleIndices(icomponent);
	BarycentricCoordinate bar;
	bar.create(p[v[0]], p[v[1]], p[v[2]]);
	float d = bar.project(result->_toPoint);
	if(d>=result->_distance) return;
	bar.compute();
	if(!bar.insideTriangle()) bar.computeClosest();
	
	Vector3F clampledP = bar.getClosest();
	d = (clampledP - result->_toPoint).length();
	if(d>=result->_distance) return;
	
	result->_distance = d;
	result->_hasResult = true;
	result->_hitPoint = clampledP;
	result->_tricoord[0] = bar.getV(0);
	result->_tricoord[1] = bar.getV(1);
	result->_tricoord[2] = bar.getV(2);
	result->_hitNormal = bar.getNormal();
	result->_icomponent = icomponent;
}
//:~