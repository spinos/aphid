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
	m_radius = new BaseBuffer;
	m_numPoints = 0;
}

APointCloud::~APointCloud() 
{
	delete m_points;
	delete m_radius;
}

const TypedEntity::Type APointCloud::type() const
{ return TPointCloud; }

const unsigned APointCloud::numPoints() const
{ return m_numPoints; }

Vector3F * APointCloud::points() const
{ return (Vector3F *)m_points->data(); }

float * APointCloud::pointRadius() const
{ return (float *)m_radius->data(); }

void APointCloud::create(unsigned n)
{	
	m_numPoints = n;
	m_points->create(n * 12);
	m_radius->create(n * 4);
}

const unsigned APointCloud::numComponents() const
{ return numPoints(); }

const BoundingBox APointCloud::calculateBBox() const
{
	BoundingBox b;
	unsigned i=0;
	for(; i< m_numPoints; i++)
		b.expandBy(points()[i], pointRadius()[i]);
		
	return b;
}

const BoundingBox APointCloud::calculateBBox(unsigned icomponent) const
{
	BoundingBox b;
	b.expandBy(points()[icomponent], pointRadius()[icomponent]);
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

void APointCloud::copyPointsFrom(Vector3F * src)
{ m_points->copyFrom(src, numPoints() * 12); }
//:~