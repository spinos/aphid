/*
 *  Boundary.cpp
 *  aphid
 *
 *  Created by jian zhang on 4/26/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "Boundary.h"

Boundary::Boundary() {}

void Boundary::setBBox(const BoundingBox &bbox)
{ m_bbox = bbox; }

const BoundingBox & Boundary::getBBox() const
{ return m_bbox; }

BoundingBox * Boundary::bbox()
{ return &m_bbox; }

void Boundary::updateBBox(const BoundingBox & box)
{ m_bbox.expandBy(box); }