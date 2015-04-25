/*
 *  Geometry.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/23/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "Geometry.h"
Geometry::Geometry() {}
Geometry::~Geometry() {}

void Geometry::setBBox(const BoundingBox &bbox)
{ m_bbox = bbox; }

const BoundingBox & Geometry::getBBox() const
{ return m_bbox; }

BoundingBox * Geometry::bbox()
{ return &m_bbox; }

void Geometry::updateBBox(const BoundingBox & box)
{ m_bbox.expandBy(box); }

const unsigned Geometry::numComponents() const 
{ return 0; }

BoundingBox const Geometry::calculateBBox() const 
{ BoundingBox b; return b; }

const BoundingBox Geometry::calculateBBox(unsigned icomponent) const
{ BoundingBox b; return b; }