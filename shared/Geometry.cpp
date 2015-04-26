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

const unsigned Geometry::numComponents() const 
{ return 0; }

BoundingBox const Geometry::calculateBBox() const 
{ return BoundingBox(); }

const BoundingBox Geometry::calculateBBox(unsigned icomponent) const
{ return BoundingBox(); }

bool Geometry::intersectBox(const BoundingBox & box)
{ return false; }

bool Geometry::intersectTetrahedron(const Vector3F * tet)
{ return false; }

bool Geometry::intersectBox(unsigned icomponent, const BoundingBox & box)
{ return false; }

bool Geometry::intersectTetrahedron(unsigned icomponent, const Vector3F * tet)
{ return false; }