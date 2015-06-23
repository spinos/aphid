/*
 *  GeometryArray.cpp
 *  
 *
 *  Created by jian zhang on 4/25/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "GeometryArray.h"
#include <BezierCurve.h>
GeometryArray::GeometryArray() 
{
	m_geos = 0;
	m_numGeometies = 0;
}

GeometryArray::~GeometryArray()
{
	delete m_geos;
}

const TypedEntity::Type GeometryArray::type() const
{ return TGeometryArray; }

void GeometryArray::create(unsigned n)
{
	m_numGeometies = n;
	m_geos = new Geometry *[n];
}

void GeometryArray::setGeometry(Geometry * geo, unsigned i)
{ m_geos[i] = geo; }

void GeometryArray::setNumGeometries(unsigned n)
{ m_numGeometies = n; }

const unsigned GeometryArray::numComponents() const
{ return m_numGeometies; }

void GeometryArray::destroyGeometries()
{
	unsigned i=0;
	for(; i < m_numGeometies; i++)
		delete m_geos[i];
}

const BoundingBox GeometryArray::calculateBBox() const
{
	BoundingBox b;
	unsigned i=0;
	for(; i < m_numGeometies; i++) {
		b.expandBy(calculateBBox(i));
	}
	return b;
}

const BoundingBox GeometryArray::calculateBBox(unsigned icomponent) const
{
	return m_geos[icomponent]->calculateBBox();
}
	
Geometry * GeometryArray::geometry(unsigned icomponent) const
{ return m_geos[icomponent]; }

const unsigned GeometryArray::numGeometries() const
{ return m_numGeometies; }

bool GeometryArray::intersectRay(const Ray * r)
{
	unsigned i=0;
	for(; i < m_numGeometies; i++)
		if(intersectRay(i, r)) return true;
	return false;
}

bool GeometryArray::intersectRay(unsigned icomponent, const Ray * r)
{
	return geometry(icomponent)->intersectRay(r);
}
