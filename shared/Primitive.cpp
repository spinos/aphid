/*
 *  Primitive.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/16/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include <TypedEntity.h>
#include "Primitive.h"

Primitive::Primitive() {}

void Primitive::setGeometry(char * data)
{
	m_geometry = data;
}

char *Primitive::getGeometry() const
{
	return m_geometry;
}

void Primitive::setComponentIndex(const unsigned &idx)
{
	m_componentIndex = idx;
}

const unsigned Primitive::getComponentIndex() const
{
	return m_componentIndex;
}

bool Primitive::isMeshGeometry() const
{
	return ((TypedEntity *)m_geometry)->isMesh();
}
