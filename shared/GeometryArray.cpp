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
	unsigned i=0;
	for(; i < m_numGeometies; i++)
		delete m_geos[i];
		
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

const unsigned GeometryArray::numComponents() const
{ return m_numGeometies; }

const BoundingBox GeometryArray::calculateBBox() const
{
	if(m_componentType == TUnknown) return Geometry::calculateBBox();
	BoundingBox b;
	unsigned i=0;
	for(; i < m_numGeometies; i++) {
		b.expandBy(calculateBBox(i));
	}
	return b;
}

const BoundingBox GeometryArray::calculateBBox(unsigned icomponent) const
{
	if(m_componentType == TBezierCurve) 
		return ((BezierCurve *)m_geos[icomponent])->calculateBBox();
	
	return BoundingBox();
}
	
Geometry * GeometryArray::geometry(unsigned icomponent) const
{ return m_geos[icomponent]; }

const unsigned GeometryArray::numGeometies() const
{ return m_numGeometies; }

void GeometryArray::setComponentType(Type t)
{ m_componentType = t; }

TypedEntity::Type GeometryArray::componentType() const
{ return m_componentType; }
