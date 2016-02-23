/*
 *  Primitive.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/16/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include "Primitive.h"
#include "Geometry.h"

namespace aphid {

Primitive::Primitive() {}

void Primitive::setGeometry(Geometry * data)
{
	m_geometry = data;
}

Geometry *Primitive::getGeometry() const
{
	return m_geometry;
}

void Primitive::setComponentIndex(const unsigned &idx)
{
	m_componentIndex = idx;
}

const unsigned & Primitive::getComponentIndex() const
{
	return m_componentIndex;
}

}