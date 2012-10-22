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

void Geometry::setBBox(const BoundingBox &bbox)
{
	m_bbox = bbox;
}

const BoundingBox Geometry::getBBox() const
{
	return m_bbox;
}