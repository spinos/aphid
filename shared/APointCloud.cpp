/*
 *  APointCloud.cpp
 *  aphid
 *
 *  Created by jian zhang on 4/26/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "APointCloud.h"
#include "BaseBuffer.h"
#include <tetrahedron_math.h>
APointCloud::APointCloud() 
{
	m_points = new BaseBuffer;
	m_numPoints = 0;
}

APointCloud::~APointCloud() 
{
	delete m_points;
}

const TypedEntity::Type APointCloud::type() const
{ return TPointCloud; }

const unsigned APointCloud::numPoints() const
{ return m_numPoints; }

Vector3F * APointCloud::points() const
{ return (Vector3F *)m_points->data(); }

void APointCloud::create(unsigned n)
{	
	m_numPoints = n;
	m_points->create(n * 12);
}

const unsigned APointCloud::numComponents() const
{ return numPoints(); }

const BoundingBox APointCloud::calculateBBox() const
{
	BoundingBox b;
	unsigned i=0;
	for(; i< m_numPoints; i++)
		b.expandBy(points()[i], 1e-6f);
		
	return b;
}

const BoundingBox APointCloud::calculateBBox(unsigned icomponent) const
{
	BoundingBox b;
	b.expandBy(points()[icomponent], 1e-6f);
	return b;
}

bool APointCloud::intersectBox(const BoundingBox & box)
{
	BoundingBox b = calculateBBox();
	return b.intersect(box);
}

bool APointCloud::intersectTetrahedron(const Vector3F * tet)
{
	unsigned i;
	for(i=0; i< m_numPoints; i++)
		if(intersectTetrahedron(i, tet)) return true;
	
	return false;
}

bool APointCloud::intersectBox(unsigned icomponent, const BoundingBox & box)
{ return box.isPointInside(points()[icomponent]); }

bool APointCloud::intersectTetrahedron(unsigned icomponent, const Vector3F * tet)
{ return pointInsideTetrahedronTest(points()[icomponent], tet); }
